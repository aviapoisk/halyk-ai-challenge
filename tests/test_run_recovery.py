"""Unit tests for per-scenario crash recovery + always-write (no LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.run import _CaseExtract, _stub_cell, run


def _mini_dataset(tmp_path: Path) -> Path:
    template = {
        "team": "",
        "contact_email": "",
        "model": "",
        "answers": {
            "S1": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
            "S2": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
        },
    }
    dataset = tmp_path / "ds"
    (dataset / "documents").mkdir(parents=True)
    (dataset / "submission_template.json").write_text(json.dumps(template), encoding="utf-8")
    (dataset / "master_ledger_2025.csv").write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-S1-0001,2025-01-01,ACC-1,A,d,1.0,USD\n"
        "TXN-S2-0001,2025-01-01,ACC-2,B,d,1.0,USD\n",
        encoding="utf-8",
    )
    return dataset


def test_failed_scenario_still_fills_stub_cells(tmp_path: Path):
    dataset = _mini_dataset(tmp_path)
    out = tmp_path / "submission.json"
    template = json.loads((dataset / "submission_template.json").read_text(encoding="utf-8"))

    config = MagicMock()
    config.dataset_dir = dataset
    config.out_path = out
    config.cache_dir = tmp_path / "cache"
    config.vision_char_threshold = 50
    config.use_cache = False
    config.only_scenario = None
    config.seed = 42
    config.profile = "default"
    config.team = "team"
    config.contact_email = "a@b.com"
    config.model_name = "test-model"
    config.self_consistency = False
    config.api_key = "x"
    config.llm_concurrency = 2
    config.llm_effort = "low"
    config.llm_adaptive_thinking = False
    config.dev = False
    config.engine = "enum"
    config.model_provider = "anthropic"

    sink = MagicMock()

    async def boom_or_ok_extract(sid, *args, **kwargs):
        if sid == "S1":
            raise RuntimeError("simulated API death")
        return _CaseExtract(
            scenario_id=sid,
            status="ready",
            flags=[],
            loan_facts=__import__("agent.schemas", fromlist=["LoanAgreementFacts"]).LoanAgreementFacts(
                borrower_name="B",
                account_id="ACC-2",
                contract_date="2025-01-01",
                effective_from="2025-01-01",
                effective_to="2025-12-31",
                is_superseded=False,
                covenants=[],
            ),
            selected_loan_doc=MagicMock(doc_id="d", path="d.pdf"),
            template_clauses=list(template["answers"][sid].keys()),
        )

    async def finish_ok(ext, *args, **kwargs):
        sid = ext.scenario_id
        return (
            {"6.1": {"status": "BREACH", "actual": 2.0, "evidence_txn_id": None}},
            {"scenario_id": sid, "status": "processed", "flags": []},
            {"scenario_id": sid, "status": "processed", "cells": {}, "flags": []},
        )

    async def _run():
        with (
            patch("agent.run.ingest_directory", return_value=([], [])),
            patch("agent.run.load_account_to_scenario", return_value=({}, [])),
            patch("agent.run.link_documents", return_value=[]),
            patch("agent.run.load_ledger", return_value=__import__("pandas").DataFrame()),
            patch("agent.run.LLMClient"),
            patch("agent.run._extract_case", new=AsyncMock(side_effect=boom_or_ok_extract)),
            patch("agent.run._finish_case", new=AsyncMock(side_effect=finish_ok)),
            patch("agent.run.write_confidence_report"),
            patch("agent.run.write_run_config"),
        ):
            return await run(config, sink)

    outcome = asyncio.run(_run())

    assert out.exists()
    assert outcome["submission"]["answers"]["S1"]["6.1"] == _stub_cell()
    assert outcome["submission"]["answers"]["S2"]["6.1"]["status"] == "BREACH"
    failed = [e for e in outcome["report_entries"] if e["scenario_id"] == "S1"][0]
    assert failed["status"] == "failed"


def test_write_survives_bad_status_without_mocking_writer(tmp_path: Path):
    """D5: real write_submission_atomic path — sanitize repairs MAYBE status."""
    dataset = _mini_dataset(tmp_path)
    out = tmp_path / "out.json"

    config = MagicMock()
    config.dataset_dir = dataset
    config.out_path = out
    config.cache_dir = tmp_path / "cache"
    config.vision_char_threshold = 50
    config.use_cache = False
    config.only_scenario = "S1"
    config.seed = 42
    config.profile = "default"
    config.team = "team"
    config.contact_email = "a@b.com"
    config.model_name = "test-model"
    config.self_consistency = False
    config.api_key = "x"
    config.llm_concurrency = 2
    config.llm_effort = "low"
    config.llm_adaptive_thinking = False
    config.dev = False
    config.engine = "enum"
    config.model_provider = "anthropic"

    sink = MagicMock()

    async def _run():
        with (
            patch("agent.run.ingest_directory", return_value=([], [])),
            patch("agent.run.load_account_to_scenario", return_value=({}, [])),
            patch("agent.run.link_documents", return_value=[]),
            patch("agent.run.load_ledger", return_value=__import__("pandas").DataFrame()),
            patch("agent.run.LLMClient"),
            patch("agent.run.write_confidence_report"),
            patch("agent.run.write_run_config"),
        ):
            return await run(config, sink)

    outcome = asyncio.run(_run())
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    for sid, clauses in data["answers"].items():
        for cell in clauses.values():
            assert cell["status"] in {"COMPLIANT", "BREACH"}
            assert cell["actual"] > 0
    assert outcome["validation"].ok
