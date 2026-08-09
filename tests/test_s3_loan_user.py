"""Unit tests for S3 loan-agreement user prompt builder and heading-anchor system text."""

from agent.s3_extract import _LOAN_AGREEMENT_SYSTEM, _loan_agreement_user


def test_loan_agreement_user_without_clauses_returns_doc_text() -> None:
    doc = "Agreement body text."
    assert _loan_agreement_user(doc) == doc
    assert _loan_agreement_user(doc, None) == doc
    assert _loan_agreement_user(doc, []) == doc


def test_loan_agreement_user_sorts_keys_and_forbids_fabrication() -> None:
    doc = "Agreement body text."
    out = _loan_agreement_user(doc, ["6.2", "6.1"])
    assert out.startswith("Template expects clause keys: 6.1, 6.2.")
    assert "Do NOT invent a clause for a template key that is not in the document." in out
    assert out.endswith(doc)
    assert "\n\n" + doc in out


def test_loan_agreement_system_uses_semantic_hint_not_title_filter() -> None:
    assert "Section titles are a HINT, not a filter" in _LOAN_AGREEMENT_SYSTEM
    assert "Extract only what the article titled" not in _LOAN_AGREEMENT_SYSTEM
    assert "«Финансовые ковенанты»" in _LOAN_AGREEMENT_SYSTEM
