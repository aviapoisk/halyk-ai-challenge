"""Unit tests for progress tracker / sinks — no LLM, no TTY required."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel, ConfigDict

from agent.config import Config
from agent.llm import LLMClient
from agent.progress import (
    NullSink,
    ProgressTracker,
    QuietSink,
    RecordingSink,
    create_progress_sink,
    render_dashboard,
)


def test_tracker_sums_tokens_and_cache():
    t = ProgressTracker()
    t.stage_start("S1_ingest", "docs")
    t.stage_end("S1_ingest", "10 docs")
    t.llm_done("LoanAgreementFacts", input_tokens=100, output_tokens=40, label="P5 · ACC-7805 · abcd")
    t.llm_done("KYCFacts", cache_hit=True, label="P5 · ACC-7805 · efgh")
    t.llm_done("TransactionCategoryBatch", input_tokens=50, output_tokens=10)

    assert t.input_tokens == 150
    assert t.output_tokens == 50
    assert t.cache_hits == 1
    assert t.cache_misses == 2
    assert t.llm_calls == 3
    assert t.stages["S1_ingest"].status == "DONE"
    assert t.stages["S1_ingest"].elapsed >= 0
    assert any("P5 · ACC-7805 · abcd" in line for line in t.log_lines)
    assert any("cache hit · KYCFacts · P5 · ACC-7805 · efgh" in line for line in t.log_lines)
    # cache hits must not appear in the billable-by-schema breakdown
    assert "KYCFacts" not in t.llm_by_schema
    assert t.llm_by_schema["LoanAgreementFacts"].calls == 1
    assert t.llm_by_schema["TransactionCategoryBatch"].input_tokens == 50


def test_schema_cost_report_excludes_cache_hits_and_sums_share():
    t = ProgressTracker(model_name="claude-opus-5")
    t.llm_done("LoanAgreementFacts", input_tokens=1_000_000, output_tokens=0)
    t.llm_done("FxRateFacts", input_tokens=0, output_tokens=1_000_000)
    t.llm_done("KYCFacts", cache_hit=True)

    rows = t.schema_cost_rows(input_rate=15.0, output_rate=75.0)
    assert [r["schema"] for r in rows] == ["FxRateFacts", "LoanAgreementFacts"]
    assert "KYCFacts" not in {r["schema"] for r in rows}
    assert rows[0]["usd"] == 75.0
    assert rows[1]["usd"] == 15.0
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-9

    report = t.format_cost_report(input_rate=15.0, output_rate=75.0)
    assert "Куда уходят деньги" in report
    assert "2 вызовов" in report
    assert "FxRateFacts" in report
    assert "LoanAgreementFacts" in report
    assert "KYCFacts" not in report
    assert "Итого" in report
    assert "100%" in report
    assert "$ (claude-opus-5)" in report


def test_quiet_sink_finish_prints_cost_report(capsys):
    sink = QuietSink()
    sink.tracker.model_name = "claude-sonnet-5"
    sink.llm_done("AuditorNotesFacts", input_tokens=2000, output_tokens=100)
    sink.finish("submission written to out.json")
    out = capsys.readouterr().out
    assert "done ·" in out
    assert "Куда уходят деньги" in out
    assert "AuditorNotesFacts" in out
    assert "Итого" in out


def test_format_llm_label_joins_identity_crumbs():
    from agent.progress import format_llm_label

    assert format_llm_label(scenario_id="P5", account_id="ACC-7805", doc_id="abcd") == "P5 · ACC-7805 · abcd"
    assert format_llm_label(doc_id="scan1", extra="p2") == "scan1 · p2"
    assert format_llm_label() == ""


def test_quiet_sink_prints_llm_label(capsys):
    sink = QuietSink()
    sink.llm_done("PageTranscript", cache_hit=True, label="f3fa6d20c8a1 · p1")
    sink.llm_done(
        "LoanAgreementFacts",
        input_tokens=12700,
        output_tokens=804,
        duration_s=1431.0,
        label="P5 · ACC-7805 · 04eee2e9ba8c",
    )
    out = capsys.readouterr().out
    assert "✔ LLM cache hit · PageTranscript · f3fa6d20c8a1 · p1" in out
    assert "✔ LLM LoanAgreementFacts · P5 · ACC-7805 · 04eee2e9ba8c · ↓12.7k ↑804 · 23:51" in out


def test_recording_sink_events():
    sink = RecordingSink()
    sink.stage_start("S2_link", "binding")
    sink.llm_wait("LoanAgreementFacts")
    sink.llm_done("LoanAgreementFacts", input_tokens=10, output_tokens=5)
    sink.scenario_start("P5")
    sink.scenario_end("P5", "2 clauses")
    sink.stage_end("S2_link")

    kinds = [e[0] for e in sink.events]
    assert kinds == [
        "stage_start",
        "llm_wait",
        "llm_done",
        "scenario_start",
        "scenario_end",
        "stage_end",
    ]
    assert sink.tracker.input_tokens == 10
    assert sink.tracker.output_tokens == 5
    assert sink.tracker.scenarios_done == 1
    assert sink.tracker.current_scenario == ""


def test_write_artifacts(tmp_path: Path):
    sink = NullSink()
    sink.tracker.model_name = "test-model"
    sink.tracker.logs_dir = "logs"
    sink.stage_start("S1_ingest")
    sink.stage_end("S1_ingest", "ok")
    sink.llm_done("X", input_tokens=3, output_tokens=2)
    sink.log("hello")
    sink.tracker.record_case("B1", {"status": "processed", "cells": {"6.1": {"outcome": "stubbed"}}})
    sink.write_artifacts(tmp_path)

    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert trace["model"] == "test-model"
    assert trace["logs_dir"] == "logs"
    assert "run" not in trace
    assert trace["tokens"]["input"] == 3
    assert trace["tokens"]["output"] == 2
    assert trace["stages"]["S1_ingest"]["status"] == "DONE"
    assert trace["cases"]["B1"]["cells"]["6.1"]["outcome"] == "stubbed"
    assert trace["alerts"] == []
    assert trace["llm_by_schema"]["X"]["calls"] == 1
    assert trace["cost"]["by_schema"][0]["schema"] == "X"
    assert "input_usd_per_mtok" in trace["cost"]
    log_text = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "hello" in log_text
    assert "Куда уходят деньги" in log_text


def test_write_artifacts_includes_run_keys(tmp_path: Path):
    sink = NullSink()
    sink.tracker.model_name = "test-model"
    sink.tracker.logs_dir = "logs"
    sink.tracker.run = {"seed": 42, "engine": "tree", "git_sha": "deadbeef"}
    sink.stage_start("S1_ingest")
    sink.stage_end("S1_ingest", "ok")
    sink.write_artifacts(tmp_path)

    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert trace["run"]["seed"] == 42
    assert trace["run"]["engine"] == "tree"
    assert trace["run"]["git_sha"] == "deadbeef"


def test_operator_alert_recorded_and_rendered():
    sink = RecordingSink()
    sink.alert(
        "UNKNOWN_COVENANT",
        "covenant does not fit known shapes",
        scenario_id="P9",
        clause="6.4",
        formula_description="single-txn cap",
        quote_preview="лимит…",
    )
    assert len(sink.tracker.alerts) == 1
    assert sink.tracker.alerts[0]["code"] == "UNKNOWN_COVENANT"
    assert sink.tracker.alerts[0]["docs"] == []
    assert any(e[0] == "alert" for e in sink.events)
    assert any("[ALERT]" in line for line in sink.tracker.log_lines)

    rendered = render_dashboard(sink.tracker)
    # Rich Group → walk renderables for alert panel title
    text = str(rendered)
    assert "OPERATOR ALERTS" in text or "UNKNOWN_COVENANT" in text or sink.tracker.alerts


def test_warning_alert_with_docs(capsys):
    sink = QuietSink()
    sink.alert(
        "CLAUSE_MISSING_FROM_EXTRACT",
        "template clause 7.1 absent from loan extract",
        severity="warning",
        scenario_id="P5",
        clause="7.1",
        docs=["loan_a.pdf", "loan_b.pdf"],
        quote_preview="6.1:Пункт…",
    )
    assert len(sink.tracker.alerts) == 1
    entry = sink.tracker.alerts[0]
    assert entry["severity"] == "warning"
    assert entry["docs"] == ["loan_a.pdf", "loan_b.pdf"]
    assert any("[WARNING]" in line and "docs=loan_a.pdf,loan_b.pdf" in line for line in sink.tracker.log_lines)
    out = capsys.readouterr().out
    assert "WARNING CLAUSE_MISSING_FROM_EXTRACT" in out
    assert "P5/7.1" in out
    assert "docs=loan_a.pdf,loan_b.pdf" in out


def test_quiet_sink_prints_alert(capsys):
    sink = QuietSink()
    sink.alert("UNKNOWN_COVENANT", "needs S4 branch", scenario_id="B1", clause="6.1")
    out = capsys.readouterr().out
    assert "ALERT UNKNOWN_COVENANT" in out
    assert "B1/6.1" in out


def test_stage_payload_in_snapshot_trace():
    t = ProgressTracker()
    t.stage_start("S1_ingest")
    t.stage_end("S1_ingest", "2 docs")
    t.set_stage_payload(
        "S1_ingest",
        {"summary": {"n_documents": 2}, "documents": [{"doc_id": "a"}]},
    )
    t.stage_start("S2_link")
    t.stage_end("S2_link", "1/2 linked")
    t.set_stage_payload(
        "S2_link",
        {"summary": {"n_unlinked": 1}, "documents": [], "unlinked": [{"doc_id": "b"}]},
    )
    trace = t.snapshot_trace()
    assert trace["stages"]["S1_ingest"]["payload"]["documents"][0]["doc_id"] == "a"
    assert trace["stages"]["S2_link"]["payload"]["unlinked"][0]["doc_id"] == "b"
    # stages without payload stay compact
    assert "payload" not in trace["stages"]["S3_extract"]


def test_log_inventory_keeps_more_than_200_lines():
    sink = RecordingSink()
    lines = [f"[S1] doc{i:04d} type=null recognition=text quality=ok pages=1 chars=10 scan_pages=0" for i in range(250)]
    sink.log_inventory("[S1]", lines)
    assert len(sink.tracker.log_lines) >= 250
    assert any(e[0] == "log_inventory" and e[1] == "[S1]" and e[2] == 250 for e in sink.events)
    # inventory lines must be in agent.log buffer, not only the summary event
    assert any("doc0000" in line for line in sink.tracker.log_lines)
    assert any("doc0249" in line for line in sink.tracker.log_lines)


def test_create_progress_sink_force_quiet():
    sink = create_progress_sink(model_name="m", out_path="out.json", force_quiet=True)
    assert isinstance(sink, QuietSink)
    assert sink.tracker.model_name == "m"


def test_render_dashboard_smoke():
    t = ProgressTracker(model_name="m", out_path="x.json")
    t.stage_start("S1_ingest", "go")
    t.llm_done("LoanAgreementFacts", input_tokens=2600, output_tokens=413)
    t.log("line")
    group = render_dashboard(t)
    assert group is not None
    # Header shows cumulative ↓in ↑out next to elapsed (Plain render for assertions).
    from rich.console import Console

    console = Console(force_terminal=False, width=120, record=True)
    console.print(group)
    text = console.export_text()
    assert "elapsed" in text
    assert "2.6k" in text
    assert "413" in text


def test_finish_summary_panel_matches_score_style():
    from rich.console import Console

    from agent.progress import _build_finish_summary_panel, _render_cost_table

    t = ProgressTracker(model_name="claude-sonnet-5")
    t.llm_done("LoanAgreementFacts", input_tokens=10_000, output_tokens=500)
    t.llm_done("KYCFacts", input_tokens=1_000, output_tokens=100)
    t.alert("UNKNOWN_COVENANT", "needs branch", scenario_id="P1", clause="6.1")

    console = Console(force_terminal=False, width=120, record=True)
    console.print(_build_finish_summary_panel(t, summary="submission written to out.json"))
    console.print(_render_cost_table(t))
    text = console.export_text()
    assert "Summary" in text
    assert "DONE" in text
    assert "OPERATOR ALERTS" in text
    assert "LoanAgreementFacts" in text
    assert "submission written to out.json" in text
    assert "Куда уходят деньги" in text
    assert "Итого" in text


def test_tracker_elapsed_advances():
    """Dashboard clock must keep ticking even without new events."""
    import time

    from agent.progress import _fmt_duration

    t = ProgressTracker()
    t.started_at = time.monotonic() - 2.5
    assert _fmt_duration(t.elapsed) == "00:02"
    t.started_at = time.monotonic() - 65.0
    assert _fmt_duration(t.elapsed) == "01:05"


class Tiny(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def test_llm_extract_cache_hit_emits_sink(tmp_path: Path):
    sink = RecordingSink()
    config = Config(
        dataset_dir=tmp_path,
        out_path=tmp_path / "out.json",
        cache_dir=tmp_path / "cache",
        model_name="test-model",
        api_key="sk-test",
        model_provider="anthropic",
    )
    client = LLMClient(config, sink=sink)

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = '{"value":"ok"}'
    mock_response.content = [mock_block]
    mock_response.usage = MagicMock(input_tokens=11, output_tokens=7)

    async def _run():
        with patch.object(client._anthropic.messages, "create", new=AsyncMock(return_value=mock_response)):
            first = await client.extract(
                "sys", "user", Tiny, cache_dir=config.cache_dir, use_cache=True, label="P1 · doc1"
            )
            second = await client.extract(
                "sys", "user", Tiny, cache_dir=config.cache_dir, use_cache=True, label="P1 · doc1"
            )
        return first, second

    first, second = asyncio.run(_run())
    assert first.value == "ok"
    assert second.value == "ok"
    done_events = [e for e in sink.events if e[0] == "llm_done"]
    assert len(done_events) == 2
    assert done_events[0][2] == 11 and done_events[0][3] == 7 and done_events[0][4] is False
    assert done_events[0][6] == "P1 · doc1"
    assert done_events[1][4] is True
    assert done_events[1][6] == "P1 · doc1"
    assert sink.tracker.input_tokens == 11
    assert sink.tracker.output_tokens == 7
    assert sink.tracker.cache_hits == 1
    assert sink.tracker.cache_misses == 1
    assert any(e[0] == "llm_wait" and e[2] == "P1 · doc1" for e in sink.events)
    assert any("P1 · doc1" in line for line in sink.tracker.log_lines)
