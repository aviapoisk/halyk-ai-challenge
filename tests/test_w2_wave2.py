"""Wave-2 unit tests — ICR, group refine, add-backs, pledge, include_in_period. No LLM."""

from __future__ import annotations

import pandas as pd

from agent.expr_tree import synthesize_expr_from_enum
from agent.s2_classify import auditor_doc_is_final, auditor_doc_is_interim_or_draft
from agent.s4_engine import (
    apply_pledge_based_transfer_categories,
    apply_reclassifications,
    evaluate_ratio_generic,
    fill_missing_amounts_from_reclassifications,
    recategorize_filled_amount_rows,
    refine_covenant_spec,
    _one_off_addbacks,
)
from agent.schemas import CovenantSpec, KYCFacts, RelatedParty, Reclassification


def test_draft_markers_include_proekt():
    assert auditor_doc_is_interim_or_draft(
        "ПРОЕКТ. НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ. Заменена окончательным отчётом."
    )
    assert not auditor_doc_is_final(
        "ПРОЕКТ. НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ."
    )


def test_refine_group_capex_from_quote():
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=9.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote=(
            "отношение совокупных капитальных затрат Группы к EBITDA Заёмщика "
            "не превышало 9.00x"
        ),
        numerator_categories=["capex"],
        denominator_categories=["revenue", "operating_expense"],
    )
    out = refine_covenant_spec(spec)
    assert out.covenant_type == "ratio_group_capex_to_ebitda"


def test_refine_and_synth_interest_coverage():
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="min",
        threshold=1.5,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="обеспечивает покрытие процентов (EBITDA к процентным расходам) не менее 1.50x",
        formula_description="EBITDA / interest_expense",
        denominator_categories=["interest_expense"],
    )
    out = refine_covenant_spec(spec)
    assert out.denominator_categories == ["interest_expense"]
    assert not out.numerator_categories
    syn = synthesize_expr_from_enum(out)
    assert syn is not None
    ops = [n.op for n in syn.expr_nodes or []]
    assert "div" in ops


def test_evaluate_interest_coverage_ratio():
    df = pd.DataFrame(
        {
            "txn_id": ["r1", "o1", "i1", "i2"],
            "amount": [10_000.0, -4_000.0, -500.0, -1_500.0],
            "amount_usd": [10_000.0, -4_000.0, -500.0, -1_500.0],
            "description": ["rev", "opex", "interest a", "interest b"],
            "counterparty": ["A", "B", "C", "D"],
            "currency": ["USD"] * 4,
        }
    )
    cats = {
        "r1": "revenue",
        "o1": "operating_expense",
        "i1": "interest_expense",
        "i2": "interest_expense",
    }
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="min",
        threshold=1.5,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="покрытие процентов EBITDA",
        formula_description="EBITDA / interest",
        denominator_categories=["interest_expense"],
    )
    res = evaluate_ratio_generic(spec, df, cats)
    # (10000-4000)/(500+1500) = 3.0
    assert res.actual == 3.0
    assert res.status == "COMPLIANT"


def test_adj_ebitda_addbacks_above_materiality():
    df = pd.DataFrame(
        {
            "txn_id": ["r1", "o1", "flood", "silt"],
            "amount": [7_000_000.0, -5_000_000.0, -481_247.63, -251_338.94],
            "amount_usd": [7_000_000.0, -5_000_000.0, -481_247.63, -251_338.94],
            "description": [
                "revenue",
                "plant opex",
                "Ilek Restoration flood remediation",
                "Zhaiyk Dredging silt removal",
            ],
            "counterparty": ["A", "B", "C", "D"],
            "currency": ["USD"] * 4,
        }
    )
    cats = {t: "operating_expense" for t in ["o1", "flood", "silt"]}
    cats["r1"] = "revenue"
    total, ids = _one_off_addbacks(df, cats, 300_000.0)
    assert "flood" in ids
    assert "silt" not in ids
    assert abs(total - 481_247.63) < 0.01

    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="min",
        threshold=0.28,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="отношение Скорректированной EBITDA к Выручке не менее 0.28x",
        formula_description="Adjusted EBITDA / Revenue",
        denominator_categories=["revenue"],
    )
    res = evaluate_ratio_generic(spec, df, cats, materiality_threshold_usd=300_000.0)
    # plain (7M - 5M - 481k - 251k)/7M vs adj adds flood back
    assert res.actual > 0.20


def test_include_in_period_applies_to_category():
    full = pd.DataFrame(
        {
            "txn_id": ["t1", "t2"],
            "amount": [100.0, -486_204.19],
            "amount_usd": [100.0, -486_204.19],
            "date": ["2025-06-01", "2024-12-15"],
            "description": ["rev", "Mineral extraction tax"],
            "counterparty": ["A", "State"],
            "currency": ["USD", "USD"],
        }
    )
    period = full[full["txn_id"] == "t1"].copy()
    cats = {"t1": "revenue", "t2": "other"}
    recs = [
        Reclassification(
            action="include_in_period",
            txn_id="t2",
            amount=486_204.19,
            to_category="taxes",
            quote="TXN t2 расход MET",
        )
    ]
    new_cats, out, flags, applied, _ = apply_reclassifications(
        cats, period, recs, full_df=full
    )
    assert "t2" in set(out["txn_id"].astype(str))
    assert new_cats["t2"] == "taxes"
    assert applied == ["t2"]


def test_recategorize_filled_tax_row():
    df = pd.DataFrame(
        {
            "txn_id": ["t2"],
            "amount": [float("nan")],
            "description": ["Mineral extraction tax assessment"],
            "counterparty": ["State"],
        }
    )
    recs = [
        Reclassification(
            action="include_in_period",
            txn_id="t2",
            amount=486_204.19,
            to_category=None,
            quote="фактическая сумма операции составляет $486,204.19 (расход).",
        )
    ]
    filled, _ = fill_missing_amounts_from_reclassifications(df, recs)
    assert filled.loc[0, "amount"] < 0
    cats, flags = recategorize_filled_amount_rows(filled, {"t2": "other"}, recs)
    assert cats["t2"] == "taxes"
    assert any("taxes" in f for f in flags)


def test_pledge_remaps_restricted_transfer():
    df = pd.DataFrame(
        {
            "txn_id": ["u1", "r1"],
            "amount": [-418_204.37, -302_118.64],
            "amount_usd": [-418_204.37, -302_118.64],
            "description": [
                "Transfer of capital assets to subsidiary",
                "Transfer of capital assets to subsidiary",
            ],
            "counterparty": [
                "Zhezkazgan Processing Holdings LLP",
                "Zhezkazgan Conveyor Assets LLP",
            ],
            "currency": ["USD", "USD"],
        }
    )
    cats = {
        "u1": "unrestricted_subsidiary_transfer",
        "r1": "unrestricted_subsidiary_transfer",
    }
    kyc = KYCFacts(
        borrower_name="Zhezkazgan Mining Services JSC",
        account_id="ACC-7809",
        related_party_threshold_pct=25.0,
        unrestricted_pledge_threshold_pct=50.0,
        counterparties=[
            RelatedParty(
                counterparty_name="Zhezkazgan Processing Holdings LLP",
                voting_rights_pct=100.0,
                pledged_assets_pct=11.4,
                quote="Processing 11.4%",
            ),
            RelatedParty(
                counterparty_name="Zhezkazgan Conveyor Assets LLP",
                voting_rights_pct=100.0,
                pledged_assets_pct=87.6,
                quote="Conveyor 87.6%",
            ),
        ],
    )
    new_cats, flags, remapped = apply_pledge_based_transfer_categories(df, cats, kyc)
    assert new_cats["r1"] == "other"
    assert new_cats["u1"] == "unrestricted_subsidiary_transfer"
    assert flags
    assert "u1" in remapped
    assert "r1" not in remapped
