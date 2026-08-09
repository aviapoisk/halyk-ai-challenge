"""Unit tests for threshold / quote helpers (no LLM)."""

from agent.threshold_check import quote_in_document, threshold_appears, threshold_forms


def test_threshold_forms_include_common_spellings():
    forms = threshold_forms(1_000_000.0)
    assert "1000000" in forms
    assert "1,000,000" in forms or "1 000 000" in forms


def test_threshold_appears():
    assert threshold_appears(3.5, "коэффициент не более 3.5x")
    assert threshold_appears(1000000, "лимит 1,000,000 USD")
    assert not threshold_appears(9.0, "лимит три процента без цифры")


def test_quote_in_document():
    doc = "Аудитор рассмотрел сумму 100 и сохранил классификацию."
    assert quote_in_document("сохранил классификацию", doc)
    assert not quote_in_document("вымышленная цитата которой нет", doc)
