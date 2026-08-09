"""Wave-3 unit tests — draft gate, consolidate bind, refine quote-miss, payroll, addbacks, tree den."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.expr_tree import ExprContext, evaluate_expr, synthesize_expr_from_enum
from agent.s2_classify import (
    auditor_doc_is_final,
    auditor_doc_is_interim_or_draft,
    classify_doc_type,
    find_group_capex_document,
)
from agent.s3_extract import _reclass_missing_to_category
from agent.s4_engine import (
    apply_reclassifications,
    apply_relevance_and_retainer_filters,
    borrower_own_capex,
    evaluate_ratio_generic,
    infer_to_category_from_quote,
    make_stub_cell,
    refine_covenant_spec,
    scrape_materiality_from_notes,
    spec_incomplete,
    _one_off_addbacks,
)
from agent.run import _stub_cell
from agent.schemas import AuditorNotesFacts, CovenantSpec, IngestedDocument, PageText, Reclassification


def _doc(doc_id: str, text: str) -> IngestedDocument:
    return IngestedDocument(
        doc_id=doc_id,
        path=f"/tmp/{doc_id}.pdf",
        sha256="0" * 64,
        pages=[
            PageText(
                page_number=1,
                raw_text=text,
                normalized_text=text,
                char_count=len(text),
                is_scan=False,
            )
        ],
    )


def test_w34_procedural_promezhutochn_is_not_draft():
    text = (
        "Аудиторские процедуры промежуточного и завершающего этапов. "
        "Окончательная позиция: операция переклассифицирована как процентные расходы."
    )
    assert not auditor_doc_is_interim_or_draft(text)
    assert auditor_doc_is_final(text)


def test_w34_proekt_status_still_draft():
    text = "ПРОЕКТ. НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ. Заменена окончательным отчётом."
    assert auditor_doc_is_interim_or_draft(text)
    assert not auditor_doc_is_final(text)


def test_w34_infer_interest_from_quote():
    q = "Операция переклассифицирована как Процентные расходы на сумму $592,296.10"
    assert infer_to_category_from_quote(q) == "interest_expense"


def test_w34_reclass_bind_by_counterparty_amount():
    df = pd.DataFrame(
        {
            "txn_id": ["TXN-B1-0020"],
            "amount": [-592_296.10],
            "amount_usd": [-592_296.10],
            "description": ["Irtysh advisory fee"],
            "counterparty": ["Irtysh Capital LLP"],
            "currency": ["USD"],
        }
    )
    cats = {"TXN-B1-0020": "operating_expense"}
    recs = [
        Reclassification(
            action="reclassify",
            txn_id=None,
            amount=592_296.10,
            counterparty="Irtysh Capital LLP",
            to_category=None,
            quote="переклассифицирована как Процентные расходы",
        )
    ]
    new_cats, _, flags, applied, unmatched = apply_reclassifications(cats, df, recs)
    assert not unmatched
    assert new_cats["TXN-B1-0020"] == "interest_expense"
    assert applied == ["TXN-B1-0020"]
    assert any("applied" in f for f in flags)


def test_w31_consolidated_beats_auditor_report_phrase():
    text = (
        "Auditor's Report\n"
        "CONSOLIDATED ANNUAL REPORT — Sarybel Energy Holding JSC\n"
        "Segment conducted through Ekibastuz Power Services JSC"
    )
    doc = _doc("parent", text)
    assert classify_doc_type(doc) == "consolidated_parent_report"
    found = find_group_capex_document([doc], "Ekibastuz Power Services JSC")
    assert found is not None
    assert found.doc_id == "parent"


def test_v4_own_capex_fallback_and_group_stub_never_threshold():
    df = pd.DataFrame(
        {
            "txn_id": ["c1", "r1"],
            "amount": [-1_204_663.28, 8_214_663.28],
            "amount_usd": [-1_204_663.28, 8_214_663.28],
            "description": ["Purchase of equipment", "sales"],
            "counterparty": ["A", "B"],
            "currency": ["USD", "USD"],
        }
    )
    cats = {"c1": "capex", "r1": "revenue"}
    own, ids = borrower_own_capex(df, cats)
    assert own == pytest.approx(1_204_663.28)
    assert ids == ["c1"]
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_group_capex_to_ebitda",
        comparator="max",
        threshold=9.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="Группы Capex / EBITDA",
    )
    stub = _stub_cell(spec)
    assert stub["actual"] != 9.0
    assert stub["actual"] != abs(float(spec.threshold))
    # Plain threshold stub still works for other types.
    other = make_stub_cell(threshold=9.0)
    assert other["actual"] == 9.0


def test_w33_refine_rp_despite_quote_miss():
    spec = CovenantSpec(
        clause="6.3",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=0.04,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="платежи аффилированным лицам не превышают 4% выручки",
        formula_description="related party payments / revenue",
        numerator_categories=["other"],
        denominator_categories=["revenue"],
    )
    out = refine_covenant_spec(spec, document_text="Loan agreement without that sentence.")
    assert out.covenant_type == "sum_related_party_payments"
    assert not out.numerator_categories


def test_w32_payroll_prior_and_retainer_not_payroll():
    df = pd.DataFrame(
        {
            "txn_id": ["p1", "r1", "c1", "s1"],
            "amount": [-100.0, -50.0, -200.0, 300.0],
            "description": [
                "Rotational staff payroll settlement Q4",
                "Management advisory retainer",
                "Purchase of kiln equipment Q2",
                "Throughput sales settlement March",
            ],
            "counterparty": ["A", "B", "C", "D"],
            "currency": ["USD", "USD", "USD", "USD"],
        }
    )
    cats = {
        "p1": "operating_expense",
        "r1": "payroll",
        "c1": "operating_expense",
        "s1": "other",
    }
    new_cats, _, flags = apply_relevance_and_retainer_filters(df, cats, {})
    assert new_cats["p1"] == "payroll"
    assert new_cats["r1"] == "other"
    assert new_cats["c1"] == "capex"
    assert new_cats["s1"] == "revenue"
    assert any("payroll_description_prior" in f for f in flags)


def test_v4_infer_to_category_requires_reclass_verb():
    assert infer_to_category_from_quote(
        "Операция переклассифицирована как процентные расходы."
    ) == "interest_expense"
    # Amount-fill scrape quote without reclass verb must not invent to_category.
    assert (
        infer_to_category_from_quote(
            "TXN-P7-0033 amount $91,447.35. Elsewhere: услуги как процентные расходы в 2026."
        )
        is None
    )


def test_v4_reclass_missing_to_category_skips_referral_and_inferable():
    facts = AuditorNotesFacts(
        borrower_name="X JSC",
        reclassifications_present=True,
        reclassification_notes=[],
        reclassifications=[
            Reclassification(
                action="reclassify",
                status="referred_elsewhere",
                quote="вывод изложен в отчёте № AR-2025-0634, не повторяется",
            ),
            Reclassification(
                action="reclassify",
                status="applied",
                quote="Операция переклассифицирована как процентные расходы.",
            ),
            Reclassification(
                action="reclassify",
                status="applied",
                quote="ссылка без категории",
                to_category=None,
            ),
        ],
    )
    bad = _reclass_missing_to_category(facts)
    assert len(bad) == 1
    assert "ссылка" in (bad[0].quote or "")


def test_v4_spec_incomplete_skips_when_ebitda_quote_repairable():
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=0.35,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="Скорректированная EBITDA к выручке",
        numerator_categories=[],
        denominator_categories=[],
    )
    flags = spec_incomplete(spec)
    assert flags == []


def test_v4_scrape_materiality_from_one_off_rule():
    notes = [
        "Разовые статьи ≥ $300,000; меньшие к EBITDA не прибавляются. "
        "Ilek Canal Restoration 481,247.63 / Aral Freight 342,905.28"
    ]
    assert scrape_materiality_from_notes(notes) == pytest.approx(300_000.0)


def test_w35_arbitration_servicing_is_one_off():
    df = pd.DataFrame(
        {
            "txn_id": ["arb", "silt"],
            "amount": [-342_905.28, -251_338.94],
            "amount_usd": [-342_905.28, -251_338.94],
            "description": [
                "Aral Freight arbitration and legal servicing",
                "Zhaiyk Dredging silt removal",
            ],
            "counterparty": ["A", "B"],
            "currency": ["USD", "USD"],
        }
    )
    cats = {"arb": "operating_expense", "silt": "operating_expense"}
    total, ids = _one_off_addbacks(df, cats, 300_000.0)
    assert "arb" in ids
    assert "silt" not in ids
    assert abs(total - 342_905.28) < 0.01


def test_w36_tree_den_includes_restricted_transfer():
    from agent.carveouts import apply_carve_outs
    from agent.schemas import CarveOut

    df = pd.DataFrame(
        {
            "txn_id": ["u1", "c1", "r1"],
            "amount": [-418_204.37, -1_204_663.28, -302_118.64],
            "amount_usd": [-418_204.37, -1_204_663.28, -302_118.64],
            "description": [
                "Transfer to unrestricted subsidiary",
                "Capex plant",
                "Transfer of assets to subsidiary Processing",
            ],
            "counterparty": ["U", "V", "R"],
            "currency": ["USD"] * 3,
        }
    )
    cats = {
        "u1": "unrestricted_subsidiary_transfer",
        "c1": "capex",
        "r1": "other",  # Restricted remapped off unrestricted — still in den via den-extra
    }
    spec = CovenantSpec(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=0.25,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="отношение передач Неограниченным к совокупным капитальным затратам; "
        "передача Ограниченным в расчёт не включается",
        numerator_categories=["unrestricted_subsidiary_transfer"],
        denominator_categories=["capex", "unrestricted_subsidiary_transfer"],
        carve_outs=[
            CarveOut(kind="category", value="Restricted / Ограниченные", quote="в расчёт не включается")
        ],
    )
    kept, excluded, cflags = apply_carve_outs(spec, df, cats)
    assert excluded == []  # Restricted must not drop from period_df (den needs them)
    assert any("CARVEOUT_DEFERRED_TO_PLEDGE" in f for f in cflags)
    assert len(kept) == 3
    syn = synthesize_expr_from_enum(spec)
    assert syn is not None
    res = evaluate_expr(syn, df, cats, ExprContext())
    assert res.actual == pytest.approx(0.22, abs=0.011)
    enum = evaluate_ratio_generic(spec, df, cats)
    assert enum.actual == pytest.approx(0.22, abs=0.011)
    assert abs(res.actual - enum.actual) < 0.011
