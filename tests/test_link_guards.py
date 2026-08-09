"""Fast unit tests for LINK-GUARDS (G1–G3) — no live LLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agent.doc_inventory import build_linking_warnings
from agent.llm import _cache_key
from agent.run import _evaluate_and_flip, missing_linked_type_flags
from agent.s1_vision import image_cache_token, ocr_scan_document
from agent.s2_link import link_documents
from agent.schemas import CovenantSpec, DocumentLinkage, IngestedDocument, PageText


def _page(n: int, text: str, *, is_scan: bool = False) -> PageText:
    body = text
    return PageText(
        page_number=n,
        raw_text=body,
        normalized_text=body,
        char_count=len(body.strip()),
        is_scan=is_scan,
    )


def _doc(doc_id: str, text: str, *, is_scan: bool = False, sha: str | None = None) -> IngestedDocument:
    return IngestedDocument(
        doc_id=doc_id,
        path=f"/tmp/{doc_id}.pdf",
        sha256=sha or f"sha-{doc_id}",
        pages=[_page(1, text, is_scan=is_scan)],
    )


# --- G2: type completeness -------------------------------------------------


def test_missing_linked_type_flags_when_no_kyc_or_auditor():
    flags = missing_linked_type_flags({"loan_agreement": [_doc("loan", "x")]})
    assert "no kyc_dossier linked to scenario" in flags
    assert "no auditor_notes linked to scenario" in flags
    # Answers are not produced by this helper — flags only.
    assert missing_linked_type_flags(
        {
            "loan_agreement": [_doc("l", "x")],
            "kyc_dossier": [_doc("k", "x")],
            "auditor_notes": [_doc("a", "x")],
        }
    ) == []


def test_related_party_stub_flag_when_kyc_missing():
    spec = CovenantSpec(
        clause="6.3",
        covenant_type="sum_related_party_payments",
        comparator="max",
        threshold=1_000_000.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        formula_description="related",
        quote="q",
    )
    import pandas as pd

    cell, flags, trace = _evaluate_and_flip(
        spec,
        pd.DataFrame(columns=["txn_id", "amount_usd", "counterparty", "date"]),
        {},
        None,
        None,
        [],
    )
    assert cell["status"] == "COMPLIANT"
    assert cell["actual"] == 1_000_000.0  # A5: stub uses covenant threshold
    assert cell["evidence_txn_id"] is None
    assert any("related-party covenant missing KYC dossier; stubbed" in f for f in flags)
    assert trace["outcome"] == "stubbed"


# --- G3: unlink_reason ------------------------------------------------------


def test_load_account_collision_majority(tmp_path: Path):
    from agent.s2_link import load_account_to_scenario

    ledger = tmp_path / "l.csv"
    ledger.write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-P1-0001,2025-01-01,ACC-1,A,d,1.0,USD\n"
        "TXN-P1-0002,2025-01-02,ACC-1,A,d,1.0,USD\n"
        "TXN-P2-0001,2025-01-03,ACC-1,A,d,1.0,USD\n",
        encoding="utf-8",
    )
    mapping, warnings = load_account_to_scenario(ledger)
    assert mapping["ACC-1"] == "P1"
    assert warnings


def test_unlink_reason_anchor_ambiguous():
    account_map = {"ACC-1001": "P1", "ACC-1002": "P2"}
    both = _doc(
        "both",
        "TwoBorrower Co JSC\nAccount ACC-1001 and ACC-1002\nДОГОВОР " + ("x" * 40),
    )
    linkages = link_documents([both], account_map)
    assert linkages[0].scenario_id is None
    assert linkages[0].unlink_reason == "anchor_ambiguous"


def test_unlink_reason_four_cases():
    # Anchor known only for ACC-1001 → P1. Bootstrap name "Known Co JSC" from a linked doc.
    account_map = {"ACC-1001": "P1"}

    linked_bootstrap = _doc(
        "boot",
        "Known Co JSC\nAccount ACC-1001\nДОГОВОР БАНКОВСКОГО ЗАЙМА body " + ("x" * 80),
    )
    # 1) no_anchor — empty/noise, no ACC, no company line
    no_anchor = _doc("n1", "noise memo without entity headers " + ("z" * 40))
    # 2) name_not_bootstrapped — company present, never seen with an anchor
    not_boot = _doc("n2", "Orphan Holdings JSC\ninternal note " + ("y" * 40))
    # 3) anchor_unknown_account — ACC not in ledger map
    unknown_acc = _doc("n3", "Ghost Co JSC\nAccount ACC-9999\nmemo " + ("w" * 40))
    # 4) name_ambiguous — same name bootstrapped to two scenarios
    # Need a second account so the same name maps to two scenarios.
    account_map["ACC-1002"] = "P2"
    amb_a = _doc(
        "amb1",
        "Ambiguous Corp JSC\nAccount ACC-1001\nДОГОВОР БАНКОВСКОГО ЗАЙМА " + ("a" * 40),
    )
    amb_b = _doc(
        "amb2",
        "Ambiguous Corp JSC\nAccount ACC-1002\nДОГОВОР БАНКОВСКОГО ЗАЙМА " + ("b" * 40),
    )
    # Unlinked doc that only has the ambiguous name (no usable unique anchor resolution
    # for itself — wait: if it has ACC-1001 it would link. So give it only the name.)
    amb_orphan = _doc("amb3", "Ambiguous Corp JSC\nplanning note " + ("c" * 40))

    linkages = link_documents(
        [linked_bootstrap, no_anchor, not_boot, unknown_acc, amb_a, amb_b, amb_orphan],
        account_map,
    )
    by_id = {l.doc_id: l for l in linkages}

    assert by_id["n1"].scenario_id is None
    assert by_id["n1"].unlink_reason == "no_anchor"

    assert by_id["n2"].scenario_id is None
    assert by_id["n2"].unlink_reason == "name_not_bootstrapped"

    assert by_id["n3"].scenario_id is None
    assert by_id["n3"].unlink_reason == "anchor_unknown_account"

    assert by_id["amb3"].scenario_id is None
    assert by_id["amb3"].unlink_reason == "name_ambiguous"

    # Linked docs must not carry unlink_reason
    assert by_id["boot"].scenario_id == "P1"
    assert by_id["boot"].unlink_reason is None


def test_linking_warnings_includes_scan_and_important_types():
    scan = _doc("scan1", "", is_scan=True)
    # force is_scan property: all pages is_scan
    assert scan.is_scan
    kyc_unlinked = _doc("kyc1", "KYC dossier Знай своего клиента " + ("k" * 40))
    noise = _doc("noise", "brand book colors " + ("n" * 40))
    linkages = [
        DocumentLinkage(doc_id="scan1", unlink_reason="no_anchor"),
        DocumentLinkage(doc_id="kyc1", unlink_reason="no_anchor"),
        DocumentLinkage(doc_id="noise", unlink_reason="no_anchor"),
    ]
    docs = {"scan1": scan, "kyc1": kyc_unlinked, "noise": noise}
    warnings = build_linking_warnings(docs, linkages)
    ids = {w["doc_id"] for w in warnings}
    assert "scan1" in ids
    assert "kyc1" in ids
    assert "noise" not in ids


# --- G1: vision cache key + guard -------------------------------------------


def test_vision_cache_key_stable_and_param_sensitive():
    common = dict(
        model="m",
        effort="high",
        thinking="disabled",
        system="sys",
        user_content="user",
        schema_name="PageTranscript",
        schema_json='{"type":"object"}',
    )
    t1 = image_cache_token("abc", 1, 1500)
    t2 = image_cache_token("abc", 1, 1500)
    assert t1 == t2
    k1 = _cache_key(**common, image_cache_token=t1)
    k2 = _cache_key(**common, image_cache_token=t2)
    assert k1 == k2

    t_other = image_cache_token("abc", 1, 1200)
    assert _cache_key(**common, image_cache_token=t_other) != k1

    # Text-only calls (no token) keep a distinct stable key
    text_key = _cache_key(**common)
    assert text_key != k1
    assert text_key == _cache_key(**common)


def test_vision_guard_short_transcript_flags(tmp_path: Path):
    import asyncio

    import pymupdf as fitz

    pdf_path = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(pdf_path)

    doc = IngestedDocument(
        doc_id="scanpdf",
        path=str(pdf_path),
        sha256="sha-scan",
        pages=[_page(1, "", is_scan=True)],
    )

    class ShortTranscript:
        text = "hi"  # below threshold 50

    llm = MagicMock()
    llm.extract = AsyncMock(return_value=ShortTranscript())

    out, flags = asyncio.run(
        ocr_scan_document(llm, doc, cache_dir=None, use_cache=False, vision_char_threshold=50)
    )
    assert any(f.startswith("vision_failed: doc scanpdf page 1") for f in flags)
    assert any(f.startswith("VISION_PAGE_FAILED:") for f in flags)
    assert out.pages[0].is_scan is True
    # Short transcript: 1500 + 2000 + table@2000 + table@2500 (W5.5).
    assert llm.extract.await_count == 4
    kwargs = llm.extract.await_args.kwargs
    assert kwargs.get("images") and len(kwargs["images"][0]) > 0
    assert kwargs.get("image_cache_token")


def test_vision_guard_exception_flags(tmp_path: Path):
    import asyncio

    import pymupdf as fitz

    pdf_path = tmp_path / "scan2.pdf"
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(pdf_path)

    doc = IngestedDocument(
        doc_id="boom",
        path=str(pdf_path),
        sha256="sha-boom",
        pages=[_page(1, "", is_scan=True)],
    )
    llm = MagicMock()
    llm.extract = AsyncMock(side_effect=RuntimeError("no vision"))

    _out, flags = asyncio.run(
        ocr_scan_document(llm, doc, cache_dir=None, use_cache=False, vision_char_threshold=50)
    )
    assert any("vision_failed: doc boom page 1 transcript ~0 chars" in f for f in flags)
