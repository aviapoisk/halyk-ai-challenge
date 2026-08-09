"""B10 — clause carve-outs: verify quotes, apply exclusions, suspect empty lists."""

from __future__ import annotations

import re

import pandas as pd

from agent.normalize import normalize_company_name
from agent.schemas import CarveOut, CovenantSpec
from agent.threshold_check import quote_in_document

# Words that suggest a carve-out; used for CARVEOUT_SUSPECTED / CARVE_OUT_LANGUAGE (no auto-exclusion).
_CARVEOUT_HINT = re.compile(
    r"за\s+исключением|кроме|не\s+включаются|исключая|исключен|"
    r"не\s+применя|не\s+распространяется|"
    r"\bexcluding\b|\bother\s+than\b|carve[\s-]?out",
    re.IGNORECASE,
)


def clause_has_carve_out_language(text: str) -> bool:
    """True when clause/quote text hints at contractual exclusions."""
    return bool(text and _CARVEOUT_HINT.search(text))


def suspect_carve_outs(spec: CovenantSpec) -> list[str]:
    """Flag when the clause text hints at exclusions but carve_outs is empty."""
    if spec.carve_outs:
        return []
    hay = f"{spec.quote or ''} {spec.formula_description or ''}"
    if clause_has_carve_out_language(hay):
        return ["CARVEOUT_SUSPECTED"]
    return []


def verify_carve_out_quotes(
    spec: CovenantSpec,
    document_text: str,
    *,
    section_text: str | None = None,
) -> tuple[CovenantSpec, list[str]]:
    """Drop carve-outs whose quote is not found in the document (prefer section_text).

    Hallucinated exclusions silently shrink aggregates — refuse them.
    """
    flags: list[str] = []
    if not spec.carve_outs:
        return spec, flags
    kept: list[CarveOut] = []
    primary = section_text if section_text else document_text
    for co in spec.carve_outs:
        found = quote_in_document(co.quote, primary) or (
            section_text is not None and quote_in_document(co.quote, document_text)
        )
        if found:
            kept.append(co)
        else:
            flags.append(f"CARVEOUT_QUOTE_NOT_FOUND kind={co.kind} value={co.value!r}")
    if len(kept) != len(spec.carve_outs):
        return spec.model_copy(update={"carve_outs": kept}), flags
    return spec, flags


def _row_matches(co: CarveOut, row: pd.Series, categories: dict[str, str]) -> bool:
    tid = str(row.get("txn_id", ""))
    if co.kind == "txn_id":
        return tid == co.value.strip()
    if co.kind == "category":
        want = co.value.strip()
        # Unknown / hallucinated category labels (e.g. restricted_subsidiary_transfer
        # not in CategoryName) → fall back to description/counterparty keyword match.
        from typing import get_args

        from agent.schemas import CategoryName

        known = set(get_args(CategoryName))
        if known and want not in known:
            desc = str(row.get("description", "") or "").casefold()
            cp = str(row.get("counterparty", "") or "").casefold()
            needle = want.replace("_", " ").casefold()
            # Restricted subsidiary transfers: match «ограниченн» / restricted in text.
            if "restricted" in needle or "ограниченн" in needle:
                return "ограниченн" in desc or "restricted" in desc or "ограниченн" in cp
            return needle in desc or needle in cp
        return categories.get(tid) == want
    if co.kind == "counterparty":
        cp = str(row.get("counterparty", "") or "")
        return normalize_company_name(cp) == normalize_company_name(co.value)
    if co.kind == "description_keyword":
        desc = str(row.get("description", "") or "").casefold()
        return co.value.casefold() in desc
    if co.kind == "amount_below":
        thr = co.amount
        if thr is None:
            try:
                thr = float(co.value)
            except ValueError:
                return False
        col = "amount_usd" if "amount_usd" in row.index else "amount"
        try:
            amt = abs(float(row[col]))
        except (TypeError, ValueError):
            return False
        return amt < abs(float(thr))
    return False


def apply_carve_outs(
    spec: CovenantSpec,
    period_df: pd.DataFrame,
    categories: dict[str, str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Filter period_df by verified carve_outs. Returns (df, excluded_txn_ids, flags).

    Known limitation: COMPLIANT-via-carve-out does not feed evidence flip-test
    (flip currently targets BREACH cells only).
    """
    flags: list[str] = []
    flags.extend(suspect_carve_outs(spec))
    if not spec.carve_outs or period_df.empty:
        return period_df, [], flags

    exclude_mask = pd.Series(False, index=period_df.index)
    for co in spec.carve_outs:
        # Restricted-subsidiary carve-outs are applied via pledge category remap
        # (numerator-only); excluding the counterparty from period_df would also
        # drop them from the capital-outflows denominator.
        if co.kind == "category" and (
            "restricted" in co.value.casefold() or "ограниченн" in co.value.casefold()
        ):
            flags.append(
                f"CARVEOUT_DEFERRED_TO_PLEDGE kind={co.kind} value={co.value!r}"
            )
            continue
        matched = period_df.apply(lambda r, c=co: _row_matches(c, r, categories), axis=1)
        if not bool(matched.any()):
            flags.append(f"CARVEOUT_MATCHED_NOTHING kind={co.kind} value={co.value!r}")
        exclude_mask = exclude_mask | matched

    excluded = period_df.loc[exclude_mask]
    excluded_ids = sorted(excluded["txn_id"].astype(str).tolist()) if not excluded.empty else []
    if excluded_ids:
        col = "amount_usd" if "amount_usd" in excluded.columns else "amount"
        excl_sum = float(excluded[col].abs().sum()) if col in excluded.columns else 0.0
        flags.append(f"carve_out_excluded n={len(excluded_ids)} abs_sum={excl_sum:.2f}")
    kept = period_df.loc[~exclude_mask].copy()
    return kept, excluded_ids, flags


def carve_out_status_flip_flag(
    status_without: str,
    status_with: str,
    excluded_ids: list[str],
    excluded_abs_sum: float,
) -> str | None:
    if not excluded_ids or status_without == status_with:
        return None
    return (
        f"CARVEOUT_FLIPPED_STATUS {status_without}->{status_with} "
        f"excluded_n={len(excluded_ids)} abs_sum={excluded_abs_sum:.2f}"
    )
