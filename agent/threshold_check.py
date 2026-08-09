"""Helpers: threshold text presence + quote verification against document text."""

from __future__ import annotations

from agent.normalize import fold_for_substring


def _fold(text: str) -> str:
    return fold_for_substring(text)


def threshold_forms(value: float) -> list[str]:
    """Plausible spellings of a numeric threshold as they appear in contracts."""
    forms = {
        f"{value:,.0f}",
        f"{value:,.2f}",
        f"{value:.0f}",
        f"{value:.2f}",
        f"{value:g}",
        f"{value * 100:.0f}%",
        f"{value * 100:.2f}%",
        f"{value}x",
        f"{value:.2f}x",
        f"{value:.1f}x",
    }
    forms |= {f.replace(",", " ") for f in forms} | {f.replace(",", "") for f in forms}
    forms |= {f.replace(".", ",") for f in list(forms)}
    return sorted(forms)


def threshold_appears(threshold: float, clause_text: str) -> bool:
    haystack = _fold(clause_text)
    return any(form in haystack for form in threshold_forms(threshold) if form)


def quote_in_document(quote: str, document_text: str) -> bool:
    """True if quote (folded) is a substring of the document text (folded)."""
    q = _fold(quote or "").strip()
    if len(q) < 8:
        return False
    return q in _fold(document_text or "")
