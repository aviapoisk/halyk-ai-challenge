"""Unit tests for S2 document inventory builders — no LLM."""

from __future__ import annotations

from agent.doc_inventory import (
    build_s2_entries,
    format_s2_unlinked_log_line,
    s2_summary,
)
from agent.schemas import DocumentLinkage, IngestedDocument, PageText


def _page(n: int, chars: int, is_scan: bool = False, text: str | None = None) -> PageText:
    body = text if text is not None else ("x" * chars)
    return PageText(
        page_number=n,
        raw_text=body,
        normalized_text=body,
        char_count=len(body.strip()),
        is_scan=is_scan,
    )


def _doc(doc_id: str, pages: list[PageText], text: str | None = None) -> IngestedDocument:
    if text is not None:
        pages = [
            PageText(
                page_number=1,
                raw_text=text,
                normalized_text=text,
                char_count=len(text.strip()),
                is_scan=False,
            )
        ]
    return IngestedDocument(doc_id=doc_id, path=f"/tmp/{doc_id}.pdf", sha256="abc" + doc_id, pages=pages)


def test_build_s2_entries_unlinked():
    docs = {
        "a1": _doc("a1", [], text="ДОГОВОР БАНКОВСКОГО ЗАЙМА " + ("x" * 100)),
        "b2": _doc("b2", [_page(1, 60)]),
    }
    linkages = [
        DocumentLinkage(
            doc_id="b2",
            account_anchor=None,
            company_name="Noise Co",
            scenario_id=None,
            link_channel=None,
        ),
        DocumentLinkage(
            doc_id="a1",
            account_anchor="ACC-7805",
            company_name="Borrower JSC",
            scenario_id="P5",
            link_channel="account_anchor",
        ),
    ]
    all_entries, unlinked = build_s2_entries(docs, linkages)
    assert [e["doc_id"] for e in all_entries] == ["a1", "b2"]
    assert all_entries[0]["scenario_id"] == "P5"
    assert all_entries[0]["type"] == "loan_agreement"
    assert len(unlinked) == 1
    assert unlinked[0]["doc_id"] == "b2"
    assert format_s2_unlinked_log_line(unlinked[0]).startswith("[S2] unlinked b2")
    summary = s2_summary(all_entries, unlinked)
    assert summary["n_linked"] == 1
    assert summary["n_unlinked"] == 1
    assert summary["by_channel"]["account_anchor"] == 1
    assert summary["by_channel"]["unlinked"] == 1
