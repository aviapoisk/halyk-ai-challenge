"""Clause-key mismatch warnings: template keys vs S3 extract (no LLM)."""

from __future__ import annotations

from agent.progress import NullSink, QuietSink
from agent.run import _report_clause_key_mismatch
from agent.schemas import CovenantSpec, LoanAgreementFacts


def _loan(*clauses: str, quotes: dict[str, str] | None = None) -> LoanAgreementFacts:
    quotes = quotes or {}
    covenants = [
        CovenantSpec(
            clause=c,
            covenant_type="sum_revenue",
            comparator="min",
            threshold=1.0,
            period_start="2025-01-01",
            period_end="2025-12-31",
            quote=quotes.get(c, f"verbatim {c}"),
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


def test_mismatch_alerts_missing_and_orphan():
    sink = NullSink()
    loan = _loan("6.1", "6.2", quotes={"6.1": "Пункт 6.1 Capex…", "6.2": "Пункт 6.2 EBITDA…"})
    flags = _report_clause_key_mismatch(
        sink,
        scenario_id="P5",
        template_clauses=["7.1", "7.2"],
        loan_facts=loan,
        loan_doc_ids=["abc123.pdf"],
    )
    assert flags == ["clause-key mismatch: missing=['7.1', '7.2'] orphan=['6.1', '6.2']"]

    codes = [a["code"] for a in sink.tracker.alerts]
    assert codes.count("CLAUSE_MISSING_FROM_EXTRACT") == 2
    assert codes.count("CLAUSE_ORPHAN_EXTRACT") == 2
    assert all(a["severity"] == "warning" for a in sink.tracker.alerts)
    assert all(a["docs"] == ["abc123.pdf"] for a in sink.tracker.alerts)

    orphans = [a for a in sink.tracker.alerts if a["code"] == "CLAUSE_ORPHAN_EXTRACT"]
    by_clause = {a["clause"]: a for a in orphans}
    assert by_clause["6.1"]["quote_preview"] == "Пункт 6.1 Capex…"
    assert "sum_revenue" in by_clause["6.1"]["formula_description"]

    log = "\n".join(sink.tracker.log_lines)
    assert "[CLAUSE_KEY_MISMATCH] P5" in log
    assert "quote='Пункт 6.1 Capex…'" in log
    assert "missing P5/7.1" in log


def test_matched_template_no_alerts():
    sink = NullSink()
    loan = _loan("6.1", "6.2", "6.3")
    flags = _report_clause_key_mismatch(
        sink,
        scenario_id="B1",
        template_clauses=["6.1", "6.2", "6.3"],
        loan_facts=loan,
        loan_doc_ids=["loan.pdf"],
    )
    assert flags == []
    assert sink.tracker.alerts == []


def test_quiet_sink_prints_warning_prefix(capsys):
    sink = QuietSink()
    loan = _loan("6.1")
    _report_clause_key_mismatch(
        sink,
        scenario_id="P3",
        template_clauses=["7.1"],
        loan_facts=loan,
        loan_doc_ids=["d1.pdf"],
    )
    out = capsys.readouterr().out
    assert "WARNING CLAUSE_MISSING_FROM_EXTRACT" in out
    assert "WARNING CLAUSE_ORPHAN_EXTRACT" in out
    assert "docs=d1.pdf" in out
