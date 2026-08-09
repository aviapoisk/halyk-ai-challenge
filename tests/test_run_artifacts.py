"""Unit tests for per-run logs dir resolution and run config archive."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.run import build_run_config, resolve_logs_dir
from agent.s6_report import write_run_config


def test_resolve_logs_dir_cwd_file_uses_root_logs(tmp_path: Path):
    assert resolve_logs_dir(Path("submission.json"), cwd=tmp_path) == tmp_path / "logs"
    assert resolve_logs_dir(Path("./out.json"), cwd=tmp_path) == tmp_path / "logs"


def test_resolve_logs_dir_nested_out_uses_sibling_logs(tmp_path: Path):
    out = tmp_path / "runs" / "g3_x" / "submission.json"
    assert resolve_logs_dir(out, cwd=tmp_path) == tmp_path / "runs" / "g3_x" / "logs"
    # relative path with multiple parts (as passed from CLI)
    assert resolve_logs_dir(Path("runs/g3_x/submission.json"), cwd=tmp_path) == Path(
        "runs/g3_x/logs"
    )


def test_resolve_logs_dir_outside_cwd(tmp_path: Path):
    foreign = tmp_path / "elsewhere" / "submission.json"
    other_cwd = tmp_path / "repo"
    other_cwd.mkdir()
    assert resolve_logs_dir(foreign, cwd=other_cwd) == foreign.parent / "logs"


def test_build_run_config_and_write(tmp_path: Path):
    cfg = SimpleNamespace(
        cache_dir=Path(".cache"),
        dataset_dir=Path("data/x"),
        dev=False,
        engine="tree",
        llm_adaptive_thinking=True,
        llm_concurrency=12,
        llm_effort="high",
        model_name="m",
        model_provider="anthropic",
        only_scenario=None,
        out_path=Path("runs/r1/submission.json"),
        profile="default",
        seed=42,
        self_consistency=True,
        use_cache=True,
        vision_char_threshold=50,
    )
    meta = build_run_config(cfg, sha="abc123")
    assert meta["git_sha"] == "abc123"
    assert meta["engine"] == "tree"
    assert meta["seed"] == 42
    assert "api_key" not in meta

    logs = tmp_path / "logs"
    path = write_run_config(meta, logs)
    assert path == logs / "config.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["git_sha"] == "abc123"
    assert loaded["out_path"] == "runs/r1/submission.json"


def test_nested_out_writes_logs_bundle(tmp_path: Path):
    """`--out runs/<id>/submission.json` archives debug files under runs/<id>/logs/."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from agent.progress import NullSink
    from agent.run import _CaseExtract, run

    dataset = tmp_path / "ds"
    (dataset / "documents").mkdir(parents=True)
    template = {
        "team": "",
        "contact_email": "",
        "model": "",
        "answers": {"S1": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}}},
    }
    (dataset / "submission_template.json").write_text(json.dumps(template), encoding="utf-8")
    (dataset / "master_ledger_2025.csv").write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-S1-0001,2025-01-01,ACC-1,A,d,1.0,USD\n",
        encoding="utf-8",
    )
    out = tmp_path / "runs" / "g3_test" / "submission.json"

    config = SimpleNamespace(
        dataset_dir=dataset,
        out_path=out,
        cache_dir=tmp_path / "cache",
        vision_char_threshold=50,
        use_cache=False,
        only_scenario=None,
        seed=7,
        profile="default",
        team="team",
        contact_email="a@b.com",
        model_name="test-model",
        model_provider="anthropic",
        self_consistency=False,
        api_key="x",
        llm_concurrency=2,
        llm_effort="low",
        llm_adaptive_thinking=False,
        dev=False,
        engine="shadow",
    )
    sink = NullSink()
    sink.tracker.model_name = config.model_name
    sink.tracker.out_path = str(out)

    async def ok_extract(sid, *args, **kwargs):
        return _CaseExtract(
            scenario_id=sid,
            status="ready",
            flags=[],
            loan_facts=__import__("agent.schemas", fromlist=["LoanAgreementFacts"]).LoanAgreementFacts(
                borrower_name="B",
                account_id="ACC-1",
                contract_date="2025-01-01",
                effective_from="2025-01-01",
                effective_to="2025-12-31",
                is_superseded=False,
                covenants=[],
            ),
            selected_loan_doc=MagicMock(doc_id="d", path="d.pdf"),
            template_clauses=["6.1"],
        )

    async def finish_ok(ext, *args, **kwargs):
        return (
            {"6.1": {"status": "COMPLIANT", "actual": 1.0, "evidence_txn_id": None}},
            {"scenario_id": ext.scenario_id, "status": "processed", "flags": []},
            {"scenario_id": ext.scenario_id, "status": "processed", "cells": {}, "flags": []},
        )

    async def _run():
        with (
            patch("agent.run.ingest_directory", return_value=([], [])),
            patch("agent.run.load_account_to_scenario", return_value=({}, [])),
            patch("agent.run.link_documents", return_value=[]),
            patch("agent.run.load_ledger", return_value=__import__("pandas").DataFrame()),
            patch("agent.run.LLMClient"),
            patch("agent.run._extract_case", new=AsyncMock(side_effect=ok_extract)),
            patch("agent.run._finish_case", new=AsyncMock(side_effect=finish_ok)),
        ):
            return await run(config, sink)

    asyncio.run(_run())

    logs = out.parent / "logs"
    assert out.exists()
    assert (logs / "confidence_report.json").exists()
    assert (logs / "config.json").exists()
    assert (logs / "trace.json").exists()
    assert (logs / "agent.log").exists()
    conf = json.loads((logs / "config.json").read_text(encoding="utf-8"))
    assert conf["seed"] == 7
    assert conf["engine"] == "shadow"
    assert "git_sha" in conf
    trace = json.loads((logs / "trace.json").read_text(encoding="utf-8"))
    assert trace["run"]["seed"] == 7
    assert trace["run"]["engine"] == "shadow"
    assert trace["logs_dir"] == str(logs)

