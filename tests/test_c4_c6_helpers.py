"""C4 related-party core-name + C6 auditor merge — no LLM."""

from __future__ import annotations

import pandas as pd

from agent.run import _merge_auditor_facts
from agent.s4_engine import related_party_match_mask
from agent.schemas import AuditorNotesFacts, IngestedDocument, PageText, Reclassification
from agent.sections import longest_section_around_anchor


def _page(text: str) -> PageText:
    return PageText(
        page_number=1,
        raw_text=text,
        normalized_text=text,
        char_count=len(text),
        is_scan=False,
    )


def test_related_core_name_unique_match():
    df = pd.DataFrame(
        [
            {
                "txn_id": "t1",
                "date": "2025-01-01",
                "counterparty": "Acme Trading",
                "description": "p",
                "amount": -100.0,
                "amount_usd": -100.0,
                "currency": "USD",
                "account_id": "A",
            }
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    names = {"acme trading llp"}
    mask, flags = related_party_match_mask(df, names)
    assert bool(mask.iloc[0])
    assert not any("COLLISION" in f for f in flags)


def test_related_core_name_collision_no_match():
    df = pd.DataFrame(
        [
            {
                "txn_id": "t1",
                "date": "2025-01-01",
                "counterparty": "Acme LLP",
                "description": "p",
                "amount": -100.0,
                "amount_usd": -100.0,
                "currency": "USD",
                "account_id": "A",
            },
            {
                "txn_id": "t2",
                "date": "2025-01-01",
                "counterparty": "Acme JSC",
                "description": "p",
                "amount": -50.0,
                "amount_usd": -50.0,
                "currency": "USD",
                "account_id": "A",
            },
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    # Related listed without OPF — core "acme" collides across LLP/JSC in ledger.
    names = {"acme"}
    mask, flags = related_party_match_mask(df, names)
    assert not bool(mask.any())
    assert any("RELATED_CORE_COLLISION" in f for f in flags)


def test_merge_auditor_dedup():
    docs = [
        IngestedDocument(doc_id="a1", path="/t/a1.pdf", sha256="1", pages=[_page("AUDITOR notes one")]),
        IngestedDocument(doc_id="a2", path="/t/a2.pdf", sha256="2", pages=[_page("AUDITOR notes two")]),
    ]
    r = Reclassification(
        action="reclassify",
        txn_id="TXN-P-1",
        amount=100.0,
        quote="q",
        from_category="other",
        to_category="payroll",
    )
    f1 = AuditorNotesFacts(
        borrower_name="B",
        reclassifications_present=True,
        reclassification_notes=["n1"],
        reclassifications=[r],
    )
    f2 = AuditorNotesFacts(
        borrower_name="B",
        reclassifications_present=True,
        reclassification_notes=["n2"],
        reclassifications=[r],  # duplicate
    )
    merged, reclass, off, doc_id, flags, note_named = _merge_auditor_facts(docs, [f1, f2])
    assert len(reclass) == 1
    assert any("multiple auditor_notes" in f for f in flags)
    assert doc_id == "a1"
    assert merged is not None
    assert note_named == []


def test_longest_section_prefers_longer_hit():
    toc = "6.1 Overview " + ("x" * 50)
    body = "6.1 " + ("BODY " * 200)
    text = toc + "\n\n" + body
    frag = longest_section_around_anchor(text, "6.1", min_chars=100, window=800)
    assert frag is not None
    assert "BODY" in frag
