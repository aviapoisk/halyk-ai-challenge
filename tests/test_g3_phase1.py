"""Unit tests for quarter / as-of quote parsers — no LLM."""

from agent.normalize import as_of_balance_from_quote, quarter_period_from_quote
from agent.s4_engine import (
    heuristic_category,
    is_covenant_relevant_heuristic,
    refine_covenant_spec,
)
from agent.schemas import CovenantSpec, OffLedgerAddition
from agent.expr_tree import ExprContext, evaluate_expr, synthesize_expr_from_enum
import pandas as pd


def test_quarter_period_fourth_ru():
    q = "Выручку за четвёртый финансовый квартал периода, оканчивающегося 2025-12-31"
    assert quarter_period_from_quote(q, "2025-12-31") == ("2025-10-01", "2025-12-31")


def test_quarter_period_q2_en():
    q = "Revenue for the second fiscal quarter ending 2025-06-30"
    assert quarter_period_from_quote(q) == ("2025-04-01", "2025-06-30")


def test_as_of_balance_detects_ru():
    assert as_of_balance_from_quote("Совокупные обязательства по персоналу по состоянию на 2025-12-31")


def test_refine_sets_quarter_and_as_of():
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="sum_revenue",
        comparator="min",
        threshold=3_500_000.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="Выручку за четвёртый финансовый квартал периода, оканчивающегося 2025-12-31 не менее 3500000",
    )
    out = refine_covenant_spec(spec)
    assert out.period_start == "2025-10-01"
    assert out.period_end == "2025-12-31"


def test_refine_ebitda_strips_other_den():
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=1.7,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="отношение поступлений по финансированию к EBITDA не более 1.70x",
        numerator_categories=["financing_proceeds"],
        denominator_categories=["other"],
        springing_min_numerator=4_000_000.0,
    )
    out = refine_covenant_spec(spec)
    assert out.denominator_categories == []
    assert out.numerator_categories == ["financing_proceeds"]
    synth = synthesize_expr_from_enum(out)
    assert synth is not None
    assert synth.expr_nodes is not None
    ops = {n.op for n in synth.expr_nodes}
    assert "sub" in ops and "div" in ops


def test_retainer_heuristic_not_opex():
    assert heuristic_category("Management advisory retainer 2025", -100.0) == "other"
    assert not is_covenant_relevant_heuristic("Brand campaign exhibition — Almaty office, instalment 3", cp_token_freq=40, desc_tail=1)
    assert is_covenant_relevant_heuristic("Plant servicing and operating costs 2025")


def test_tree_sum_category_includes_off_ledger():
    df = pd.DataFrame(
        [
            {
                "txn_id": "t1",
                "date": pd.Timestamp("2025-06-01"),
                "account_id": "A",
                "counterparty": "X",
                "description": "payroll",
                "amount": -100.0,
                "currency": "USD",
                "amount_usd": -100.0,
            }
        ]
    )
    cats = {"t1": "payroll"}
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="sum_category",
        comparator="max",
        threshold=150.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="payroll",
        filter_categories=["payroll"],
    )
    tree_spec = synthesize_expr_from_enum(spec)
    assert tree_spec is not None
    assert any(n.op == "off_ledger" for n in (tree_spec.expr_nodes or []))
    ctx = ExprContext(
        off_ledger=[OffLedgerAddition(amount=50.0, category="payroll", quote="severance")],
        period_end="2025-12-31",
    )
    result = evaluate_expr(tree_spec, df, cats, ctx)
    assert result.actual == 150.0
    assert result.status == "COMPLIANT"
