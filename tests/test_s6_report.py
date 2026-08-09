"""S6 sanitize + always-write (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from agent.s6_report import sanitize_answers, write_submission_atomic


def _template() -> dict:
    return {
        "team": "",
        "contact_email": "",
        "model": "",
        "answers": {
            "S1": {
                "6.1": {"status": None, "actual": None, "evidence_txn_id": None},
                "6.2": {"status": None, "actual": None, "evidence_txn_id": None},
            }
        },
    }


def test_sanitize_repairs_bad_cells():
    template = _template()
    submission = {
        "team": "t",
        "contact_email": "a@b.com",
        "model": "m",
        "answers": {
            "S1": {
                "6.1": {"status": "maybe", "actual": 0, "evidence_txn_id": "TXN-OTHER-1"},
                "6.2": {"status": "BREACH", "actual": 1.234, "evidence_txn_id": None},
                "extra": {"status": "COMPLIANT", "actual": 1.0, "evidence_txn_id": None},
            },
            "EXTRA": {},
        },
    }
    ledger_index = {"S1": {"TXN-S1-0001"}}
    fixes = sanitize_answers(submission, template, ledger_index)
    assert submission["answers"]["S1"]["6.1"]["status"] == "COMPLIANT"
    assert submission["answers"]["S1"]["6.1"]["actual"] == 0.01
    assert submission["answers"]["S1"]["6.1"]["evidence_txn_id"] is None
    assert submission["answers"]["S1"]["6.2"]["actual"] == 1.23
    assert "extra" not in submission["answers"]["S1"]
    assert "EXTRA" not in submission["answers"]
    assert fixes


def test_write_always_creates_file(tmp_path: Path):
    template = _template()
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-S1-0001,2025-01-01,ACC-1,A,d,1.0,USD\n",
        encoding="utf-8",
    )
    out = tmp_path / "submission.json"
    submission = {
        "team": "t",
        "contact_email": "a@b.com",
        "model": "m",
        "answers": {
            "S1": {
                "6.1": {"status": "MAYBE", "actual": None, "evidence_txn_id": None},
                "6.2": {"status": "BREACH", "actual": 2.5, "evidence_txn_id": None},
            }
        },
    }
    result, fixes = write_submission_atomic(submission, template, ledger, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["answers"]["S1"]["6.1"]["status"] == "COMPLIANT"
    assert data["answers"]["S1"]["6.1"]["actual"] == 0.01
    assert result.ok
    assert fixes
