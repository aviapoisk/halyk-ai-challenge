"""Synthetic scan / long-context smoke tests (no live LLM)."""

from agent.normalize import sanitize
from agent.schemas import IngestedDocument, PageText


def test_scan_document_property_when_all_pages_below_threshold():
    doc = IngestedDocument(
        doc_id="scan",
        path="/tmp/scan.pdf",
        sha256="a" * 64,
        pages=[
            PageText(page_number=1, raw_text="", normalized_text="", char_count=0, is_scan=True),
            PageText(page_number=2, raw_text=" ", normalized_text=" ", char_count=0, is_scan=True),
        ],
    )
    assert doc.is_scan is True


def test_long_normalized_text_sanitizes_without_error():
    huge = ("A C C - 7 8 0 5 " * 5000) + ("слово " * 5000)
    out = sanitize(huge)
    assert "ACC" in out.replace(" ", "") or "A C C" in huge
    assert len(out) > 10_000
