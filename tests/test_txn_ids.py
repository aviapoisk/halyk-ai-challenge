"""Unit tests for txn_id → scenario resolve (private multi-segment + open-set)."""

from __future__ import annotations

from pathlib import Path

from agent.s2_link import load_account_to_scenario
from agent.s4_engine import extract_txn_ids_from_notes, scrape_missing_txn_amounts_from_notes
from agent.s6_report import sanitize_answers
from agent.txn_ids import parse_txn_scenario
from tools.validate import load_ledger_index


def test_parse_txn_scenario_longest_template_prefix():
    known = {"KC", "K", "B2"}
    assert parse_txn_scenario("TXN-KC-CAP-29", known) == "KC"
    assert parse_txn_scenario("TXN-KC-OPX-51", known) == "KC"
    assert parse_txn_scenario("TXN-B2-0001", known) == "B2"


def test_parse_txn_scenario_open_set_and_noise():
    known = {"P5", "B4"}
    assert parse_txn_scenario("TXN-P5-0001", known) == "P5"
    assert parse_txn_scenario("TXN-9001-0042", known) == "9001"
    assert parse_txn_scenario("TXN-9001-0042", None) == "9001"
    assert parse_txn_scenario("not-a-txn", known) is None


def test_load_account_maps_tele_to_template_key(tmp_path: Path):
    ledger = tmp_path / "l.csv"
    ledger.write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-KC-CAP-29,2025-01-01,TELE-4471,A,d,1.0,USD\n"
        "TXN-KC-OPX-51,2025-01-02,TELE-4471,A,d,1.0,USD\n"
        "TXN-B2-0001,2025-01-03,ACC-7202,A,d,1.0,USD\n",
        encoding="utf-8",
    )
    mapping, warnings = load_account_to_scenario(ledger, {"KC", "B2"})
    assert mapping["TELE-4471"] == "KC"
    assert mapping["ACC-7202"] == "B2"
    assert not warnings


def test_ledger_index_membership_for_multi_segment(tmp_path: Path):
    ledger = tmp_path / "l.csv"
    ledger.write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-KC-CAP-29,2025-01-01,TELE-4471,A,d,1.0,USD\n",
        encoding="utf-8",
    )
    index = load_ledger_index(ledger, {"KC"})
    assert "TXN-KC-CAP-29" in index["KC"]
    assert "KC-CAP" not in index


def test_sanitize_keeps_multi_segment_evidence():
    template = {
        "answers": {"KC": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}}}
    }
    submission = {
        "team": "t",
        "contact_email": "a@b.com",
        "model": "m",
        "answers": {
            "KC": {
                "6.1": {
                    "status": "BREACH",
                    "actual": 1.5,
                    "evidence_txn_id": "TXN-KC-CAP-29",
                }
            }
        },
    }
    ledger_index = {"KC": {"TXN-KC-CAP-29"}}
    fixes = sanitize_answers(submission, template, ledger_index)
    assert submission["answers"]["KC"]["6.1"]["evidence_txn_id"] == "TXN-KC-CAP-29"
    assert not any("не принадлежит" in f for f in fixes)


def test_note_scrape_multi_segment_txn_id():
    note = "See TXN-KC-CAP-29 for $1,234.56 reclass and also TXN-AA-0001."
    assert extract_txn_ids_from_notes([note]) == ["TXN-KC-CAP-29", "TXN-AA-0001"]
    found = scrape_missing_txn_amounts_from_notes([note])
    assert any(r.txn_id == "TXN-KC-CAP-29" and r.amount == 1234.56 for r in found)
