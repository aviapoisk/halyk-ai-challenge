"""Unit tests for dataset discovery (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.dataset import discover_dataset


def _write_mini_dataset(root: Path, *, ledger_name: str = "master_ledger_2025.csv", docs_dir: str = "documents") -> None:
    (root / docs_dir).mkdir(parents=True)
    (root / docs_dir / "a.pdf").write_bytes(b"%PDF-1.4")
    (root / docs_dir / "b.txt").write_text("note", encoding="utf-8")
    (root / ledger_name).write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-S1-0001,2025-01-01,ACC-1,A,d,1.0,USD\n",
        encoding="utf-8",
    )
    template = {
        "team": "",
        "contact_email": "",
        "model": "",
        "answers": {"S1": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}}},
    }
    (root / "submission_template.json").write_text(json.dumps(template), encoding="utf-8")
    # Filled GT must not be treated as template.
    gt = {
        "scenarios": {},
        "answers": {"S1": {"6.1": {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None}}},
    }
    (root / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")


def test_discover_renamed_ledger_and_docs(tmp_path: Path):
    _write_mini_dataset(tmp_path, ledger_name="ledger_2027.csv", docs_dir="files")
    paths = discover_dataset(tmp_path)
    assert paths.ledger.name == "ledger_2027.csv"
    assert paths.documents.name == "files"
    assert paths.template.name == "submission_template.json"


def test_discover_rejects_ground_truth_as_template(tmp_path: Path):
    _write_mini_dataset(tmp_path)
    paths = discover_dataset(tmp_path)
    assert paths.template.name == "submission_template.json"


def test_discover_missing_ledger(tmp_path: Path):
    (tmp_path / "documents").mkdir()
    (tmp_path / "submission_template.json").write_text(
        json.dumps(
            {
                "answers": {"S1": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="CSV"):
        discover_dataset(tmp_path)
