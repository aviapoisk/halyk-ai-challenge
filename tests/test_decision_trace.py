"""Unit tests for decision-trace builders — no LLM."""

from __future__ import annotations

from agent.decision_trace import (
    build_auditor_section,
    build_binding,
    build_cell_trace,
    build_loan_section,
    diagnose_cell,
    empty_case_trace,
)
from agent.schemas import (
    CovenantSpec,
    DocumentLinkage,
    IngestedDocument,
    LoanAgreementFacts,
    PageText,
    Reclassification,
)


def _doc(doc_id: str) -> IngestedDocument:
    return IngestedDocument(
        doc_id=doc_id,
        path=f"{doc_id}.pdf",
        sha256="x",
        pages=[PageText(page_number=1, raw_text="t", normalized_text="t", char_count=1, is_scan=False)],
    )


def _loan(*clauses: str) -> LoanAgreementFacts:
    covenants = [
        CovenantSpec(
            clause=c,
            covenant_type="sum_revenue",
            comparator="min",
            threshold=1.0,
            period_start="2025-01-01",
            period_end="2025-12-31",
            quote="q",
        )
        for c in clauses
    ]
    return LoanAgreementFacts(
        borrower_name="Test JSC",
        account_id="ACC-7201",
        contract_date="2025-01-01",
        effective_from="2025-01-01",
        effective_to="2025-12-31",
        is_superseded=False,
        covenants=covenants,
    )


class TestBuildLoanSection:
    def test_missing_clauses_listed(self):
        loan = _loan("6.3")
        section = build_loan_section(loan, ["6.1", "6.2", "6.3"])
        assert [c["clause"] for c in section["covenants_extracted"]] == ["6.3"]
        assert section["covenants_extracted"][0]["quote"] == "q"
        assert section["clauses_missing_from_extract"] == ["6.1", "6.2"]


class TestBuildBinding:
    def test_docs_list(self):
        loan = _loan("6.1")
        linkages = [
            DocumentLinkage(doc_id="loan1", scenario_id="B1", link_channel="account_anchor"),
            DocumentLinkage(doc_id="aud1", scenario_id="B1", link_channel="company_name"),
        ]
        by_type = {"loan_agreement": [_doc("loan1")], "auditor_notes": [_doc("aud1")]}
        binding = build_binding("B1", linkages, by_type, loan)
        assert binding["account_id"] == "ACC-7201"
        assert {d["doc_id"] for d in binding["docs"]} == {"loan1", "aud1"}
        assert next(d for d in binding["docs"] if d["doc_id"] == "loan1")["channel"] == "account_anchor"


class TestBuildAuditorSection:
    def test_applied_flag(self):
        recs = [
            Reclassification(
                action="reclassify",
                txn_id="TXN-B1-0020",
                from_category="other",
                to_category="operating_expense",
                quote="q",
            ),
            Reclassification(action="exclude_from_period", txn_id="TXN-MISSING", quote="gone"),
        ]
        section = build_auditor_section("aud1", recs, applied_keys=["TXN-B1-0020"], unmatched_indices=[1])
        assert section["reclassifications"][0]["applied"] is True
        assert section["reclassifications"][1]["applied"] is False


class TestBuildCellTrace:
    def test_stub_diagnosis(self):
        cell = build_cell_trace(
            {"status": "COMPLIANT", "actual": 1.0, "evidence_txn_id": None},
            stub_reason="clause not computed; stubbed",
        )
        assert cell["outcome"] == "stubbed"
        assert cell["diagnosis"] == "clause absent from extract → stubbed"
        assert diagnose_cell(cell) == "clause absent from extract → stubbed"

    def test_empty_case_has_stub_cells(self):
        case = empty_case_trace("B1", "unprocessed", ["no loan"], ["6.1", "6.2"])
        assert case["status"] == "unprocessed"
        assert set(case["cells"]) == {"6.1", "6.2"}
        assert case["cells"]["6.1"]["outcome"] == "stubbed"
