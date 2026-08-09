"""Expression-tree validator + golden enum parity — no LLM."""

from __future__ import annotations

import pandas as pd

from agent.expr_tree import (
    ExprContext,
    evaluate_expr,
    synthesize_expr_from_enum,
    validate_expr_tree,
)
from agent.s4_engine import evaluate_sum_category, evaluate_sum_revenue, prepare_aggregation_frame, snapshot_as_of
from agent.schemas import CovenantSpec, ExprNode


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_validate_rejects_cycle():
    nodes = [
        ExprNode(id=0, op="add", children=[1]),
        ExprNode(id=1, op="add", children=[0]),
    ]
    ok, flags = validate_expr_tree(nodes, 0)
    assert not ok
    assert any("cycle" in f for f in flags)


def test_validate_rejects_bad_refs():
    nodes = [ExprNode(id=0, op="div", children=[1, 2])]
    ok, flags = validate_expr_tree(nodes, 0)
    assert not ok


def test_validate_rejects_depth():
    nodes = [
        ExprNode(id=0, op="add", children=[1]),
        ExprNode(id=1, op="add", children=[2]),
        ExprNode(id=2, op="add", children=[3]),
        ExprNode(id=3, op="add", children=[4]),
        ExprNode(id=4, op="add", children=[5]),
        ExprNode(id=5, op="const", value=1.0),
    ]
    ok, flags = validate_expr_tree(nodes, 0)
    assert not ok
    assert any("depth" in f for f in flags)


def test_golden_sum_revenue_tree_matches_enum():
    df = _df(
        [
            {
                "txn_id": "t1",
                "date": "2025-06-01",
                "account_id": "A",
                "counterparty": "X",
                "description": "d",
                "amount": 1500.0,
                "currency": "USD",
                "amount_usd": 1500.0,
            }
        ]
    )
    cats = {"t1": "revenue"}
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="sum_revenue",
        comparator="min",
        threshold=1000.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="q",
    )
    enum_r = evaluate_sum_revenue(spec, df, cats)
    tree_spec = synthesize_expr_from_enum(spec)
    assert tree_spec is not None
    tree_r = evaluate_expr(tree_spec, df, cats, ExprContext())
    assert tree_r.status == enum_r.status
    assert abs(tree_r.actual - enum_r.actual) < 0.011


def test_golden_sum_category_tree_matches_enum():
    df = _df(
        [
            {
                "txn_id": "t1",
                "date": "2025-06-01",
                "account_id": "A",
                "counterparty": "X",
                "description": "d",
                "amount": -500.0,
                "currency": "USD",
                "amount_usd": -500.0,
            }
        ]
    )
    cats = {"t1": "capex"}
    spec = CovenantSpec(
        clause="6.2",
        covenant_type="sum_category",
        comparator="max",
        threshold=1000.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="q",
        filter_category="capex",
    )
    enum_r = evaluate_sum_category(spec, df, cats)
    tree_spec = synthesize_expr_from_enum(spec)
    assert tree_spec is not None
    tree_r = evaluate_expr(tree_spec, df, cats, ExprContext())
    assert tree_r.status == enum_r.status
    assert abs(tree_r.actual - enum_r.actual) < 0.011


def test_springing_via_div_numerator_keeps_actual():
    df = _df(
        [
            {
                "txn_id": "n1",
                "date": "2025-06-01",
                "account_id": "A",
                "counterparty": "X",
                "description": "fin",
                "amount": 100.0,
                "currency": "USD",
                "amount_usd": 100.0,
            },
            {
                "txn_id": "d1",
                "date": "2025-06-01",
                "account_id": "A",
                "counterparty": "Y",
                "description": "cap",
                "amount": -1000.0,
                "currency": "USD",
                "amount_usd": -1000.0,
            },
        ]
    )
    cats = {"n1": "financing_proceeds", "d1": "capex"}
    spec = CovenantSpec(
        clause="6.3",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=0.5,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="q",
        numerator_category="financing_proceeds",
        denominator_category="capex",
        springing_min_numerator=500.0,
    )
    tree_spec = synthesize_expr_from_enum(spec)
    assert tree_spec is not None
    tree_spec = tree_spec.model_copy(update={"springing_min_numerator": 500.0})
    result = evaluate_expr(tree_spec, df, cats, ExprContext())
    assert result.status == "COMPLIANT"
    assert result.actual > 0.01  # real ratio kept, not stubbed to 0.01 from zero


def test_as_of_balance_cumulative():
    df = _df(
        [
            {
                "txn_id": "a",
                "date": "2025-01-01",
                "account_id": "A",
                "counterparty": "X",
                "description": "d",
                "amount": 10.0,
                "currency": "USD",
            },
            {
                "txn_id": "b",
                "date": "2025-06-30",
                "account_id": "A",
                "counterparty": "X",
                "description": "d",
                "amount": 20.0,
                "currency": "USD",
            },
            {
                "txn_id": "c",
                "date": "2025-12-31",
                "account_id": "A",
                "counterparty": "X",
                "description": "d",
                "amount": 30.0,
                "currency": "USD",
            },
            {
                "txn_id": "d",
                "date": "2026-01-01",
                "account_id": "A",
                "counterparty": "X",
                "description": "d",
                "amount": 99.0,
                "currency": "USD",
            },
        ]
    )
    snap = snapshot_as_of(df, "2025-12-31")
    assert set(snap["txn_id"]) == {"a", "b", "c"}
    spec = CovenantSpec(
        clause="6.9",
        covenant_type="sum_category",
        comparator="max",
        threshold=1e9,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="q",
        filter_category="other",
        aggregation="as_of_balance",
    )
    period = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")]
    out, flags = prepare_aggregation_frame(spec, period, full_df=df)
    assert "cumulative_rows=3" in flags[0]
    assert "exact_day_rows=1" in flags[0]
    assert set(out["txn_id"]) == {"a", "b", "c"}


def test_synthesize_unrestricted_ratio_ignores_ebitda_disclaimer():
    """formula_description may say «not Group capex to Borrower EBITDA» — den must stay capex."""
    from agent.s4_engine import refine_covenant_spec

    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=0.15,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote=(
            "капитальных активов, переданных Неограниченным дочерним организациям, "
            "не превышала 0.15x совокупных капитальных затрат Заёмщика"
        ),
        formula_description=(
            "ratio of capital assets transferred to Unrestricted Subsidiaries "
            "to Borrower's total capital expenditures "
            "(not Group capex to Borrower EBITDA)"
        ),
        numerator_categories=[],
        denominator_categories=[],
    )
    spec = refine_covenant_spec(spec)
    tree = synthesize_expr_from_enum(spec)
    assert tree is not None
    den = next(
        n
        for n in tree.expr_nodes
        if n.id == tree.expr_nodes[tree.expr_root].children[1]
    )
    assert den.op == "agg"
    assert "capex" in den.categories
    assert "unrestricted_subsidiary_transfer" in den.categories
    assert den.op != "sub"
