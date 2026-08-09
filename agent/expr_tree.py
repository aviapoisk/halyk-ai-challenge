"""Expression-tree validator + evaluator (shadow / tree engines).

Flat node list with integer child refs — no recursive JSON schema.
Invalid trees fall back to the enum path (caller clears expr_* + flags).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from agent.s4_engine import (
    CovenantResult,
    _amount_col,
    _cat_list,
    _currency_mix,
    _money_minus_tokens,
    _off_ledger_sum,
    _one_off_addbacks,
    _related_party_names,
    _restricted_transfer_den_extra,
    _sum_cats,
    looks_like_dscr,
    looks_like_ebitda_over_opex_lines,
    looks_like_money_linear_metric,
    looks_like_net_debt,
    looks_like_ratio_language,
    looks_like_total_debt_leverage,
    related_party_match_mask,
    snapshot_as_of,
    verdict,
)
from agent.schemas import CovenantSpec, EBITDA_OPEX_CATEGORIES, ExprNode, RelatedParty

_MAX_DEPTH = 4
_BINARY = {"sub", "div"}
_NARY = {"add", "max", "min"}
_LEAVES = {"const", "agg", "related_outflows", "group_capex", "off_ledger"}


@dataclass
class ExprContext:
    group_capex: float | None = None
    related_parties: list[RelatedParty] | None = None
    related_threshold_pct: float | None = None
    off_ledger: list | None = None
    period_end: str | None = None
    materiality_threshold_usd: float | None = None
    adj_ebitda: bool = False


def validate_expr_tree(
    nodes: list[ExprNode] | None,
    root: int | None,
) -> tuple[bool, list[str]]:
    """Return (ok, flags). On failure caller should null out expr fields."""
    flags: list[str] = []
    if nodes is None or root is None:
        return False, ["expr missing nodes or root"]
    by_id: dict[int, ExprNode] = {}
    for n in nodes:
        if n.id in by_id:
            flags.append(f"expr duplicate id={n.id}")
            return False, flags
        by_id[n.id] = n
    if root not in by_id:
        flags.append(f"expr root {root} not in nodes")
        return False, flags

    # Acyclicity + depth via DFS.
    visiting: set[int] = set()
    visited: set[int] = set()

    def dfs(nid: int, depth: int) -> bool:
        if depth > _MAX_DEPTH:
            flags.append(f"expr depth > {_MAX_DEPTH} at id={nid}")
            return False
        if nid in visiting:
            flags.append(f"expr cycle at id={nid}")
            return False
        if nid in visited:
            return True
        if nid not in by_id:
            flags.append(f"expr missing child id={nid}")
            return False
        node = by_id[nid]
        visiting.add(nid)
        for c in node.children:
            if not dfs(c, depth + 1):
                return False
        visiting.remove(nid)
        visited.add(nid)
        return True

    if not dfs(root, 0):
        return False, flags

    for nid, node in by_id.items():
        if node.op in _LEAVES:
            if node.children:
                flags.append(f"expr leaf id={nid} op={node.op} has children")
                return False, flags
            if node.op == "const" and node.value is None:
                flags.append(f"expr const id={nid} missing value")
                return False, flags
            if node.op == "agg" and not node.categories:
                flags.append(f"expr agg id={nid} missing categories")
                return False, flags
            if node.op == "off_ledger" and not node.categories:
                flags.append(f"expr off_ledger id={nid} missing categories")
                return False, flags
        elif node.op in _BINARY:
            if len(node.children) != 2:
                flags.append(f"expr {node.op} id={nid} needs 2 children")
                return False, flags
        elif node.op in _NARY:
            if len(node.children) < 1:
                flags.append(f"expr {node.op} id={nid} needs ≥1 children")
                return False, flags
        else:
            flags.append(f"expr unknown op={node.op} id={nid}")
            return False, flags
    return True, flags


def sanitize_expr_on_spec(spec: CovenantSpec) -> tuple[CovenantSpec, list[str]]:
    """Clear invalid expr_* so the enum path remains authoritative."""
    if not spec.expr_nodes and spec.expr_root is None:
        return spec, []
    ok, flags = validate_expr_tree(spec.expr_nodes, spec.expr_root)
    if ok:
        return spec, []
    return spec.model_copy(update={"expr_nodes": None, "expr_root": None}), [
        f"EXPR_INVALID: {f}" for f in flags
    ]


def _agg_value(
    period_df: pd.DataFrame,
    categories: dict[str, str],
    cats: list[str],
    sign: str,
) -> tuple[float, list[str]]:
    raw, ids = _sum_cats(period_df, categories, cats)
    if sign == "signed":
        return float(raw), ids
    if sign == "abs":
        return abs(float(raw)), ids
    col = _amount_col(period_df)
    if period_df.empty or not cats:
        return 0.0, []
    wanted = set(cats)
    mask = period_df["txn_id"].map(lambda t: categories.get(str(t)) in wanted)
    sub = period_df[mask]
    if sign == "outflows":
        pay = sub[sub[col] < 0]
        return float(-pay[col].sum()), list(pay["txn_id"].astype(str))
    if sign == "inflows":
        inc = sub[sub[col] > 0]
        return float(inc[col].sum()), list(inc["txn_id"].astype(str))
    return abs(float(raw)), ids


def evaluate_expr(
    spec: CovenantSpec,
    period_df: pd.DataFrame,
    categories: dict[str, str],
    ctx: ExprContext,
) -> CovenantResult:
    """Recursive evaluation of a validated flat expression tree."""
    nodes = {n.id: n for n in (spec.expr_nodes or [])}
    root = spec.expr_root
    assert root is not None and root in nodes

    contrib: list[str] = []
    flags: list[str] = []
    springing_num: float | None = None

    def eval_node(nid: int) -> float:
        nonlocal springing_num
        node = nodes[nid]
        if node.op == "const":
            return float(node.value or 0.0)
        if node.op == "group_capex":
            if ctx.group_capex is None:
                flags.append("expr group_capex missing")
                return 0.0
            return float(ctx.group_capex)
        if node.op == "off_ledger":
            return _off_ledger_sum(ctx.off_ledger, node.categories or [])
        if node.op == "agg":
            frame = period_df
            if node.aggregation == "as_of_balance" and ctx.period_end:
                frame = snapshot_as_of(period_df, ctx.period_end)
            cats = list(node.categories or [])
            val, ids = _agg_value(frame, categories, cats, node.sign)
            # W3.6: mirror enum `_restricted_transfer_den_extra` — Restricted
            # transfers remapped to `other` still belong in total-capex den.
            if (
                "capex" in cats
                and "unrestricted_subsidiary_transfer" in cats
                and node.sign == "abs"
            ):
                extra, extra_ids = _restricted_transfer_den_extra(
                    frame, categories, set(ids)
                )
                if extra:
                    val = float(val) + float(extra)
                    ids = list(ids) + list(extra_ids)
                    flags.append(f"expr den +restricted transfers={extra:.2f}")
            # W2.6: Adj.EBITDA — exclude one-off opex (≥ materiality) from the opex abs leaf.
            if (
                ctx.adj_ebitda
                and ctx.materiality_threshold_usd is not None
                and node.sign == "abs"
                and node.categories
                and set(node.categories) <= set(EBITDA_OPEX_CATEGORIES)
            ):
                add_total, add_ids = _one_off_addbacks(
                    frame, categories, ctx.materiality_threshold_usd
                )
                if add_total:
                    val = max(0.0, float(val) - float(add_total))
                    flags.append(
                        f"expr Adj opex reduced by add-backs={add_total:.2f} n={len(add_ids)}"
                    )
                    contrib.extend(add_ids)
            contrib.extend(ids)
            return val
        if node.op == "related_outflows":
            if not ctx.related_parties:
                flags.append("expr related_outflows missing KYC")
                return 0.0
            # Null thr% → explicit affiliates only (same as enum _related_party_names).
            names = _related_party_names(ctx.related_parties, ctx.related_threshold_pct)
            mask, mflags = related_party_match_mask(period_df, names)
            flags.extend(mflags)
            col = _amount_col(period_df)
            pay = period_df[mask & (period_df[col] < 0)]
            contrib.extend(list(pay["txn_id"].astype(str)))
            return float(-pay[col].sum())
        kids = [eval_node(c) for c in node.children]
        if node.op == "add":
            return float(sum(kids))
        if node.op == "sub":
            return float(kids[0] - kids[1])
        if node.op == "div":
            num, den = kids[0], kids[1]
            if nid == root:
                springing_num = abs(float(num))
            if den <= 0:
                flags.append("expr div denominator <= 0 — undefined")
                return max(abs(float(num)), 0.01)
            return float(num) / float(den)
        if node.op == "max":
            return float(max(kids)) if kids else 0.0
        if node.op == "min":
            return float(min(kids)) if kids else 0.0
        flags.append(f"expr unknown op {node.op}")
        return 0.0

    value = eval_node(root)
    status = verdict(value, spec.comparator, spec.threshold)

    actual = round(abs(value), 2) if abs(value) > 0 else 0.01
    return CovenantResult(
        clause=spec.clause,
        status=status,
        actual=actual,
        contributing_txn_ids=sorted(set(contrib)),
        flags=flags,
        inputs={
            "expr_value_raw": float(value),
            "numerator": springing_num,
            "springing_numerator": springing_num,
            "period_row_count": int(len(period_df)),
            "currency_mix": _currency_mix(period_df),
            "engine": "tree",
        },
    )


def synthesize_expr_from_enum(spec: CovenantSpec) -> CovenantSpec | None:
    """Build a flat tree that mirrors the enum evaluator (for shadow/golden).

    Dual-path with s4 evaluate_*: when changing a formula, update both.
    """
    # Tier-1 shapes that the flat tree cannot express — stay on enum path.
    if getattr(spec, "description_bucket", None):
        return None
    if getattr(spec, "pct_of_group_capex", False):
        return None
    if getattr(spec, "quarter_reduce", None):
        return None
    if getattr(spec, "springing_mode", None) in {
        "own_capex_above",
        "trigger_above",
        "min_quarter_revenue_below",
        "leverage_above",
        "payroll_ratio_above",
    }:
        return None
    if getattr(spec, "ebitda_plus_categories", None):
        return None
    quote0 = f"{spec.quote or ''} {spec.formula_description or ''}".casefold()
    if looks_like_dscr(quote0):
        return None
    nodes: list[ExprNode] = []
    nxt = 0

    def add(op: str, **kw: Any) -> int:
        nonlocal nxt
        nid = nxt
        nxt += 1
        nodes.append(ExprNode(id=nid, op=op, **kw))  # type: ignore[arg-type]
        return nid

    t = spec.covenant_type
    quote = f"{spec.quote or ''} {spec.formula_description or ''}".casefold()
    has_ebitda = "ebitda" in quote

    if t == "sum_revenue":
        root = add("agg", categories=["revenue"], sign="signed")
    elif t == "sum_category":
        cats = _cat_list(spec.filter_categories, spec.filter_category)
        if not cats:
            return None
        # Enum path adds matching off_ledger amounts — mirror that here.
        agg_id = add("agg", categories=cats, sign="abs", aggregation=spec.aggregation)
        ol_id = add("off_ledger", categories=cats)
        root = add("add", children=[agg_id, ol_id])
    elif t == "max_category":
        cats = _cat_list(spec.filter_categories, spec.filter_category)
        if not cats:
            return None
        child_ids = [add("agg", categories=[c], sign="abs") for c in cats]
        root = add("max", children=child_ids)
    elif t == "revenue_minus_max_category":
        cats = _cat_list(spec.filter_categories, spec.filter_category)
        if not cats:
            return None
        rev = add("agg", categories=["revenue"], sign="signed")
        child_ids = [add("agg", categories=[c], sign="abs") for c in cats]
        mx = add("max", children=child_ids)
        root = add("sub", children=[rev, mx])
    elif t == "ratio_generic":
        num_cats = _cat_list(spec.numerator_categories, spec.numerator_category)
        den_cats = _cat_list(spec.denominator_categories, spec.denominator_category)
        # Strip garbage residual "other" — never a formula article.
        den_cats = [c for c in den_cats if c != "other"]
        num_cats = [c for c in num_cats if c != "other"]

        # Absolute $ metrics: synthesize sum / sub, not div (tree engine prefers this path).
        thr = float(spec.threshold)
        if looks_like_money_linear_metric(quote, thr):
            minus = _money_minus_tokens(quote)
            true_ebitda_ratio = has_ebitda and not minus and (
                "долг" in quote
                or "debt" in quote
                or "leverage" in quote
                or "покрытия" in quote
                or "coverage" in quote
                or "марж" in quote
                or "/ ebitda" in quote
                or "/ebitda" in quote
                or "к ebitda" in quote
            )
            if not true_ebitda_ratio:
                if minus and num_cats and den_cats:
                    left_cats, right_cats = list(num_cats), list(den_cats)
                    if "revenue" in right_cats and "capex" in left_cats:
                        left_cats, right_cats = right_cats, left_cats
                    elif (
                        "financing_proceeds" in right_cats
                        and any(c in left_cats for c in ("interest_expense", "taxes"))
                    ):
                        left_cats, right_cats = right_cats, left_cats
                    left = add("agg", categories=left_cats, sign="abs")
                    right = add("agg", categories=right_cats, sign="abs")
                    root = add("sub", children=[left, right])
                    return spec.model_copy(update={"expr_nodes": nodes, "expr_root": root})
                filt = _cat_list(spec.filter_categories, spec.filter_category) or []
                sum_cats = num_cats or filt or (den_cats if not has_ebitda else [])
                if sum_cats and (not den_cats or not num_cats or not looks_like_ratio_language(quote)):
                    root = add("agg", categories=sum_cats if num_cats or filt else den_cats, sign="abs")
                    return spec.model_copy(update={"expr_nodes": nodes, "expr_root": root})

        # Explicit category dens that are not EBITDA-shape (capex, transfers, …) win over
        # incidental "EBITDA" tokens in formula_description disclaimers
        # («not Group capex to Borrower EBITDA»).
        _ebitda_den_ok = {"interest_expense", "revenue", "financing", "rent", "utilities", "insurance"}
        ebitda_shape_den = not den_cats or set(den_cats) <= _ebitda_den_ok
        use_ebitda = has_ebitda and ebitda_shape_den

        # EBITDA (incl. Adjusted) → sub(revenue, operating_expense), never abs-sum of cats.
        if use_ebitda:
            rev = add("agg", categories=["revenue"], sign="signed")
            opex = add("agg", categories=list(EBITDA_OPEX_CATEGORIES), sign="abs")
            ebitda = add("sub", children=[rev, opex])
            adj = "скорректирован" in quote or "adjusted ebitda" in quote
            interest_den = den_cats == ["interest_expense"] or set(den_cats) == {"interest_expense"}
            dscr = looks_like_dscr(quote) or (
                not num_cats and set(den_cats or []) == {"interest_expense", "financing"}
            )
            opex_line_den = bool(den_cats) and set(den_cats) <= {"rent", "utilities", "insurance"}
            icr = (
                not dscr
                and (
                    "покрытия процентов" in quote
                    or "покрытие процентов" in quote
                    or "interest coverage" in quote
                    or (not num_cats and interest_den)
                )
            )
            if dscr:
                den_ds = add("agg", categories=["interest_expense", "financing"], sign="abs")
                root = add("div", children=[ebitda, den_ds])
            elif icr or interest_den:
                den_int = add("agg", categories=["interest_expense"], sign="abs")
                root = add("div", children=[ebitda, den_int])
            elif (opex_line_den or looks_like_ebitda_over_opex_lines(quote)) and not num_cats:
                dens = den_cats or ["rent", "utilities", "insurance"]
                den_op = add("agg", categories=list(dens), sign="abs")
                root = add("div", children=[ebitda, den_op])
            elif looks_like_net_debt(quote) or (
                num_cats and set(num_cats) == {"financing_proceeds", "financing"}
            ):
                drawn = add("agg", categories=["financing_proceeds"], sign="abs")
                repay = add("agg", categories=["financing"], sign="abs")
                net = add("sub", children=[drawn, repay])
                root = add("div", children=[net, ebitda])
            elif looks_like_total_debt_leverage(quote) and (
                not num_cats or num_cats == ["financing_proceeds"]
            ):
                num = add("agg", categories=["financing_proceeds"], sign="abs")
                root = add("div", children=[num, ebitda])
            # Adj. EBITDA / Revenue (or EBITDA / Revenue).
            elif adj or (
                "выручк" in quote
                and ("ebitda" in quote)
                and not num_cats
                and (not den_cats or den_cats == ["revenue"])
                and not looks_like_total_debt_leverage(quote)
                and not looks_like_dscr(quote)
            ):
                den_rev = add("agg", categories=["revenue"], sign="abs")
                root = add("div", children=[ebitda, den_rev])
            elif num_cats:
                # financing_proceeds / EBITDA, (taxes+utilities)/EBITDA, etc.
                num = add("agg", categories=num_cats, sign="abs")
                root = add("div", children=[num, ebitda])
            elif den_cats == ["revenue"] or (not den_cats and "выручк" in quote):
                den_rev = add("agg", categories=["revenue"], sign="abs")
                root = add("div", children=[ebitda, den_rev])
            else:
                # Numerator unknown — cannot invent.
                return None
        else:
            if not num_cats or not den_cats:
                return None
            # Unrestricted transfers / total capex: den includes transfers + capex.
            if "unrestricted_subsidiary_transfer" in num_cats and "capex" in den_cats:
                if "unrestricted_subsidiary_transfer" not in den_cats:
                    den_cats = list(den_cats) + ["unrestricted_subsidiary_transfer"]
            num = add("agg", categories=num_cats, sign="abs")
            den = add("agg", categories=den_cats, sign="abs")
            root = add("div", children=[num, den])
    elif t == "ratio_group_capex_to_ebitda":
        from agent.s4_engine import is_absolute_group_capex_ceiling

        # Absolute Group-capex ceilings (incl. outside-borrower) stay on the enum path.
        if is_absolute_group_capex_ceiling(spec):
            return None
        g = add("group_capex")
        rev = add("agg", categories=["revenue"], sign="signed")
        opex = add("agg", categories=list(EBITDA_OPEX_CATEGORIES), sign="abs")
        ebitda = add("sub", children=[rev, opex])
        root = add("div", children=[g, ebitda])
    elif t == "sum_related_party_payments":
        # Absolute related outflows; ratio-of-revenue handled by enum path when threshold < 1.
        if spec.threshold < 1.0:
            return None
        root = add("related_outflows")
    else:
        return None

    return spec.model_copy(update={"expr_nodes": nodes, "expr_root": root})
