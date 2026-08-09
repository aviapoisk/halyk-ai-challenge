from agent.s2_classify import classify_doc_type, find_group_capex_document
from agent.schemas import IngestedDocument, PageText


def make_doc(doc_id: str, text: str) -> IngestedDocument:
    return IngestedDocument(
        doc_id=doc_id,
        path=f"/tmp/{doc_id}.pdf",
        sha256="0" * 64,
        pages=[PageText(page_number=1, raw_text=text, normalized_text=text, char_count=len(text), is_scan=False)],
    )


class TestClassifyDocType:
    def test_loan_agreement(self):
        doc = make_doc("a", "ДОГОВОР БАНКОВСКОГО ЗАЙМА между Заёмщиком и Банком")
        assert classify_doc_type(doc) == "loan_agreement"

    def test_kyc_dossier(self):
        doc = make_doc("b", 'Досье «Знай своего клиента» (KYC)')
        assert classify_doc_type(doc) == "kyc_dossier"

    def test_auditor_notes(self):
        doc = make_doc("c", "АУДИТОРСКОЕ ДЕЛО №ACC-0001/2025")
        assert classify_doc_type(doc) == "auditor_notes"

    def test_consolidated_parent_report(self):
        doc = make_doc("d", "CONSOLIDATED ANNUAL REPORT — Some Holding JSC")
        assert classify_doc_type(doc) == "consolidated_parent_report"

    def test_unclassified_returns_none(self):
        doc = make_doc("e", "мусорный документ без опознавательных фраз")
        assert classify_doc_type(doc) is None

    def test_loan_agreement_wins_over_later_patterns(self):
        """Priority contract: first matching pattern in _PATTERNS wins."""
        doc = make_doc(
            "x",
            "ДОГОВОР БАНКОВСКОГО ЗАЙМА\nAlso mentions CONSOLIDATED group report and KYC dossier",
        )
        assert classify_doc_type(doc) == "loan_agreement"


class TestFindGroupCapexDocument:
    def test_matches_across_line_break_in_borrower_name(self):
        """Regression: PyMuPDF can wrap a company name across a line break inside a
        paragraph ("Ekibastuz Power\\nServices JSC"); the lookup must not require the
        name to appear on a single unbroken line."""
        text = "CONSOLIDATED REPORT. Segment is conducted through Ekibastuz Power\nServices JSC, which operates."
        doc = make_doc("h", text)
        found = find_group_capex_document([doc], "Ekibastuz Power Services JSC")
        assert found is not None
        assert found.doc_id == "h"

    def test_no_match_returns_none(self):
        doc = make_doc("i", "CONSOLIDATED REPORT mentioning Some Other Company JSC only.")
        assert find_group_capex_document([doc], "Ekibastuz Power Services JSC") is None

    def test_ignores_non_consolidated_docs(self):
        doc = make_doc("j", "ДОГОВОР БАНКОВСКОГО ЗАЙМА mentioning Ekibastuz Power Services JSC.")
        assert find_group_capex_document([doc], "Ekibastuz Power Services JSC") is None
