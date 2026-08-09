"""Progress-only observability for agent.run — no HITL / no input prompts.

TTY: Rich Live dashboard (stages, logs, tokens). Pipe/CI: line-oriented QuietSink.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STAGE_ORDER = (
    "S1_ingest",
    "S2_link",
    "S3_extract",
    "S4_engine",
    "S5_evidence",
    "S6_write",
)

STAGE_LABELS = {
    "S1_ingest": "1. PDF/CSV Ingestion",
    "S2_link": "2. Entity Linking",
    "S3_extract": "3. Case Extraction",
    "S4_engine": "4. Deterministic Engine",
    "S5_evidence": "5. Evidence Flip-Test",
    "S6_write": "6. Submission Write",
}

_ORANGE = "#D96B27"
_GREEN = "#4EBF71"
_YELLOW = "#E5C07B"
_RED = "#E06C75"
_LOG_KEEP = 12
_LOG_LINES_MAX = 10_000


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


# Report-only USD rates ($ per 1M tokens). Override via env; defaults match Opus-scale pricing.
_DEFAULT_INPUT_USD_PER_MTOK = 15.0
_DEFAULT_OUTPUT_USD_PER_MTOK = 75.0


def _pricing_rates() -> tuple[float, float]:
    return (
        _env_float("AGENT_INPUT_USD_PER_MTOK", _DEFAULT_INPUT_USD_PER_MTOK),
        _env_float("AGENT_OUTPUT_USD_PER_MTOK", _DEFAULT_OUTPUT_USD_PER_MTOK),
    )


def _usd_cost(input_tokens: int, output_tokens: int, *, input_rate: float, output_rate: float) -> float:
    return input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate


def _model_short_label(model_name: str) -> str:
    """Short label for the $ column header (not a pricing source)."""
    name = (model_name or "").strip()
    if not name:
        return "model"
    # claude-opus-5-2025… → claude-opus-5; keep last path segment for provider/id forms
    leaf = name.rsplit("/", 1)[-1]
    parts = leaf.split("-")
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) >= 6:
        leaf = "-".join(parts[:-1])
    return leaf


@dataclass
class SchemaUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def format_llm_label(
    *,
    scenario_id: str = "",
    account_id: str = "",
    doc_id: str = "",
    extra: str = "",
) -> str:
    """Join identity crumbs for LLM wait/done lines (scenario · ACC · doc · …)."""
    return " · ".join(p for p in (scenario_id, account_id, doc_id, extra) if p)


def _llm_ctx(label: str) -> str:
    return f" · {label}" if label else ""


@dataclass
class StageState:
    status: str = "IDLE"  # IDLE | RUNNING | DONE
    detail: str = ""
    started_at: float | None = None
    elapsed: float = 0.0
    payload: dict = field(default_factory=dict)


@dataclass
class ProgressTracker:
    model_name: str = ""
    out_path: str = ""
    started_at: float = field(default_factory=time.monotonic)
    stages: dict[str, StageState] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    status_line: str = "starting…"
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_by_schema: dict[str, SchemaUsage] = field(default_factory=dict)
    scenarios_done: int = 0
    scenarios_total: int = 0
    current_scenario: str = ""
    cases: dict[str, dict] = field(default_factory=dict)
    logs_dir: str = "logs"
    # CLI / env keys for this run (seed, engine, git_sha, …) — written into trace.json.
    run: dict = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key in STAGE_ORDER:
            self.stages.setdefault(key, StageState())

    def record_case(self, scenario_id: str, case: dict) -> None:
        self.cases[scenario_id] = case

    def alert(
        self,
        code: str,
        message: str,
        *,
        severity: str = "critical",
        scenario_id: str = "",
        clause: str = "",
        formula_description: str = "",
        quote_preview: str = "",
        docs: list[str] | None = None,
    ) -> None:
        doc_list = list(docs or [])
        entry = {
            "severity": severity,
            "code": code,
            "scenario_id": scenario_id,
            "clause": clause,
            "message": message,
            "formula_description": formula_description,
            "quote_preview": quote_preview,
            "docs": doc_list,
        }
        self.alerts.append(entry)
        label = "WARNING" if severity == "warning" else "ALERT"
        parts = [f"[{label}] {code}"]
        if scenario_id:
            parts.append(scenario_id)
        if clause:
            parts.append(clause)
        parts.append(message)
        if doc_list:
            parts.append(f"docs={','.join(doc_list)}")
        self.log(" · ".join(parts))
        self.status_line = f"{label} {code}" + (f" {scenario_id}/{clause}" if scenario_id else "")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def stage_start(self, stage: str, detail: str = "") -> None:
        st = self.stages.setdefault(stage, StageState())
        st.status = "RUNNING"
        st.detail = detail
        st.started_at = time.monotonic()
        self.status_line = detail or STAGE_LABELS.get(stage, stage)

    def stage_end(self, stage: str, detail: str = "") -> None:
        st = self.stages.setdefault(stage, StageState())
        if st.started_at is not None:
            st.elapsed = time.monotonic() - st.started_at
        st.status = "DONE"
        if detail:
            st.detail = detail
        self.status_line = f"{STAGE_LABELS.get(stage, stage)} done"

    def set_stage_payload(self, stage: str, payload: dict) -> None:
        st = self.stages.setdefault(stage, StageState())
        st.payload = payload

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {message}")
        if len(self.log_lines) > _LOG_LINES_MAX:
            self.log_lines = self.log_lines[-_LOG_LINES_MAX:]

    def log_inventory(self, tag: str, lines: list[str]) -> str:
        """Append many inventory lines to agent.log; return a one-line TTY summary."""
        for message in lines:
            self.log(message)
        summary = f"{tag} document inventory · {len(lines)} lines (see agent.log)"
        self.status_line = summary
        return summary

    def llm_wait(self, schema_name: str, *, label: str = "") -> None:
        ctx = _llm_ctx(label)
        self.status_line = f"waiting LLM ({schema_name}){ctx}…"
        self.log(f"[LLM] waiting · {schema_name}{ctx}")

    def llm_done(
        self,
        schema_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        duration_s: float = 0.0,
        label: str = "",
    ) -> None:
        self.llm_calls += 1
        ctx = _llm_ctx(label)
        if cache_hit:
            self.cache_hits += 1
            self.log(f"[LLM] cache hit · {schema_name}{ctx}")
            self.status_line = f"cache hit · {schema_name}{ctx}"
            return
        self.cache_misses += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        usage = self.llm_by_schema.setdefault(schema_name, SchemaUsage())
        usage.calls += 1
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        self.log(
            f"[LLM] {schema_name}{ctx} · ↓{_fmt_tokens(input_tokens)} ↑{_fmt_tokens(output_tokens)} · {_fmt_duration(duration_s)}"
        )
        self.status_line = (
            f"LLM {schema_name}{ctx} · ↓{_fmt_tokens(input_tokens)} ↑{_fmt_tokens(output_tokens)}"
        )

    def scenario_start(self, scenario_id: str) -> None:
        self.current_scenario = scenario_id
        self.status_line = f"scenario {scenario_id}"
        self.log(f"[CASE] start {scenario_id}")

    def scenario_end(self, scenario_id: str, detail: str = "") -> None:
        self.scenarios_done += 1
        suffix = f" — {detail}" if detail else ""
        self.log(f"[CASE] done {scenario_id}{suffix}")
        self.current_scenario = ""

    def schema_cost_rows(
        self,
        *,
        input_rate: float | None = None,
        output_rate: float | None = None,
    ) -> list[dict]:
        """Billable (cache-miss) usage by schema, sorted by USD cost descending."""
        in_rate, out_rate = _pricing_rates()
        if input_rate is not None:
            in_rate = input_rate
        if output_rate is not None:
            out_rate = output_rate
        rows: list[dict] = []
        for name, usage in self.llm_by_schema.items():
            cost = _usd_cost(
                usage.input_tokens,
                usage.output_tokens,
                input_rate=in_rate,
                output_rate=out_rate,
            )
            rows.append(
                {
                    "schema": name,
                    "calls": usage.calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "usd": cost,
                }
            )
        rows.sort(key=lambda r: (-r["usd"], r["schema"]))
        total_usd = sum(r["usd"] for r in rows)
        for row in rows:
            row["share"] = (row["usd"] / total_usd) if total_usd > 0 else 0.0
        return rows

    def format_cost_report(
        self,
        *,
        input_rate: float | None = None,
        output_rate: float | None = None,
    ) -> str:
        """Plaintext cost table for agent.log / QuietSink (Russian header like battle triage)."""
        in_rate, out_rate = _pricing_rates()
        if input_rate is not None:
            in_rate = input_rate
        if output_rate is not None:
            out_rate = output_rate
        rows = self.schema_cost_rows(input_rate=in_rate, output_rate=out_rate)
        billed_calls = sum(r["calls"] for r in rows)
        in_tok = sum(r["input_tokens"] for r in rows)
        out_tok = sum(r["output_tokens"] for r in rows)
        total_usd = sum(r["usd"] for r in rows)
        model_label = _model_short_label(self.model_name)
        dollar_hdr = f"$ ({model_label})"
        title = (
            f"Куда уходят деньги (по логу, {billed_calls} вызовов, "
            f"{_fmt_tokens(in_tok)} in / {_fmt_tokens(out_tok)} out)"
        )
        # Column widths: schema left, numeric right-aligned
        schema_w = max([len("Стадия"), len("Итого"), *(len(r["schema"]) for r in rows)] or [6])
        schema_w = max(schema_w, 20)
        calls_w = max(len("Вызовов"), 7)
        in_w = max(len("Input"), 7)
        out_w = max(len("Output"), 7)
        usd_w = max(len(dollar_hdr), 8)
        share_w = max(len("Доля"), 5)

        def _line(schema: str, calls: str, inp: str, out: str, usd: str, share: str) -> str:
            return (
                f"{schema:<{schema_w}}  {calls:>{calls_w}}  {inp:>{in_w}}  "
                f"{out:>{out_w}}  {usd:>{usd_w}}  {share:>{share_w}}"
            )

        lines = [
            title,
            _line("Стадия", "Вызовов", "Input", "Output", dollar_hdr, "Доля"),
        ]
        for r in rows:
            lines.append(
                _line(
                    r["schema"],
                    str(r["calls"]),
                    _fmt_tokens(r["input_tokens"]),
                    _fmt_tokens(r["output_tokens"]),
                    f"{r['usd']:.2f}",
                    f"{round(r['share'] * 100):.0f}%",
                )
            )
        lines.append(
            _line(
                "Итого",
                str(billed_calls),
                _fmt_tokens(in_tok),
                _fmt_tokens(out_tok),
                f"{total_usd:.2f}",
                "100%" if rows else "0%",
            )
        )
        return "\n".join(lines)

    def log_cost_report(self) -> None:
        """Append cost table lines to the agent.log buffer."""
        report = self.format_cost_report()
        for line in report.splitlines():
            self.log(line)

    def snapshot_trace(self) -> dict:
        in_rate, out_rate = _pricing_rates()
        rows = self.schema_cost_rows(input_rate=in_rate, output_rate=out_rate)
        total_usd = sum(r["usd"] for r in rows)
        trace = {
            "model": self.model_name,
            "out_path": self.out_path,
            "logs_dir": self.logs_dir,
            "elapsed_s": round(self.elapsed, 3),
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.input_tokens + self.output_tokens,
            },
            "llm": {
                "calls": self.llm_calls,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
            "llm_by_schema": {
                name: {
                    "calls": usage.calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
                for name, usage in sorted(self.llm_by_schema.items())
            },
            "cost": {
                "input_usd_per_mtok": in_rate,
                "output_usd_per_mtok": out_rate,
                "total_usd": round(total_usd, 4),
                "by_schema": [
                    {
                        "schema": r["schema"],
                        "calls": r["calls"],
                        "input_tokens": r["input_tokens"],
                        "output_tokens": r["output_tokens"],
                        "usd": round(r["usd"], 4),
                        "share": round(r["share"], 4),
                    }
                    for r in rows
                ],
            },
            "scenarios": {"done": self.scenarios_done, "total": self.scenarios_total},
            "alerts": list(self.alerts),
            "stages": {
                key: {
                    "status": st.status,
                    "detail": st.detail,
                    "elapsed_s": round(st.elapsed, 3),
                    **({"payload": st.payload} if st.payload else {}),
                }
                for key, st in self.stages.items()
            },
            "cases": {sid: self.cases[sid] for sid in sorted(self.cases)},
        }
        if self.run:
            trace["run"] = dict(self.run)
        return trace


class ProgressSink(Protocol):
    tracker: ProgressTracker

    def stage_start(self, stage: str, detail: str = "") -> None: ...
    def stage_end(self, stage: str, detail: str = "") -> None: ...
    def set_stage_payload(self, stage: str, payload: dict) -> None: ...
    def log(self, message: str) -> None: ...
    def log_inventory(self, tag: str, lines: list[str]) -> None: ...
    def llm_wait(self, schema_name: str, *, label: str = "") -> None: ...
    def llm_done(
        self,
        schema_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        duration_s: float = 0.0,
        label: str = "",
    ) -> None: ...
    def scenario_start(self, scenario_id: str) -> None: ...
    def scenario_end(self, scenario_id: str, detail: str = "") -> None: ...
    def set_scenarios_total(self, total: int) -> None: ...
    def alert(
        self,
        code: str,
        message: str,
        *,
        severity: str = "critical",
        scenario_id: str = "",
        clause: str = "",
        formula_description: str = "",
        quote_preview: str = "",
        docs: list[str] | None = None,
    ) -> None: ...
    def finish(self, summary: str = "") -> None: ...
    def write_artifacts(self, out_dir: Path) -> None: ...


class NullSink:
    """No-op sink for unit tests."""

    def __init__(self) -> None:
        self.tracker = ProgressTracker()

    def stage_start(self, stage: str, detail: str = "") -> None:
        self.tracker.stage_start(stage, detail)

    def stage_end(self, stage: str, detail: str = "") -> None:
        self.tracker.stage_end(stage, detail)

    def set_stage_payload(self, stage: str, payload: dict) -> None:
        self.tracker.set_stage_payload(stage, payload)

    def log(self, message: str) -> None:
        self.tracker.log(message)

    def log_inventory(self, tag: str, lines: list[str]) -> None:
        self.tracker.log_inventory(tag, lines)

    def llm_wait(self, schema_name: str, *, label: str = "") -> None:
        self.tracker.llm_wait(schema_name, label=label)

    def llm_done(
        self,
        schema_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        duration_s: float = 0.0,
        label: str = "",
    ) -> None:
        self.tracker.llm_done(
            schema_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            duration_s=duration_s,
            label=label,
        )

    def scenario_start(self, scenario_id: str) -> None:
        self.tracker.scenario_start(scenario_id)

    def scenario_end(self, scenario_id: str, detail: str = "") -> None:
        self.tracker.scenario_end(scenario_id, detail)

    def set_scenarios_total(self, total: int) -> None:
        self.tracker.scenarios_total = total

    def alert(
        self,
        code: str,
        message: str,
        *,
        severity: str = "critical",
        scenario_id: str = "",
        clause: str = "",
        formula_description: str = "",
        quote_preview: str = "",
        docs: list[str] | None = None,
    ) -> None:
        self.tracker.alert(
            code,
            message,
            severity=severity,
            scenario_id=scenario_id,
            clause=clause,
            formula_description=formula_description,
            quote_preview=quote_preview,
            docs=docs,
        )

    def finish(self, summary: str = "") -> None:
        return

    def write_artifacts(self, out_dir: Path) -> None:
        _write_artifacts(self.tracker, out_dir)


class QuietSink(NullSink):
    """Line-oriented stdout for pipes / CI / --no-tui."""

    def stage_start(self, stage: str, detail: str = "") -> None:
        super().stage_start(stage, detail)
        label = STAGE_LABELS.get(stage, stage)
        extra = f" — {detail}" if detail else ""
        print(f"→ {label}{extra}", flush=True)

    def stage_end(self, stage: str, detail: str = "") -> None:
        super().stage_end(stage, detail)
        st = self.tracker.stages[stage]
        label = STAGE_LABELS.get(stage, stage)
        extra = f" — {detail}" if detail else ""
        print(f"✔ {label} ({_fmt_duration(st.elapsed)}){extra}", flush=True)

    def log(self, message: str) -> None:
        super().log(message)
        print(message, flush=True)

    def log_inventory(self, tag: str, lines: list[str]) -> None:
        summary = self.tracker.log_inventory(tag, lines)
        print(summary, flush=True)

    def llm_wait(self, schema_name: str, *, label: str = "") -> None:
        super().llm_wait(schema_name, label=label)
        print(f"… waiting LLM ({schema_name}){_llm_ctx(label)}", flush=True)

    def llm_done(
        self,
        schema_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        duration_s: float = 0.0,
        label: str = "",
    ) -> None:
        super().llm_done(
            schema_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            duration_s=duration_s,
            label=label,
        )
        # tracker.llm_done appends to the log buffer only; QuietSink prints here.
        ctx = _llm_ctx(label)
        if cache_hit:
            print(f"✔ LLM cache hit · {schema_name}{ctx}", flush=True)
        else:
            print(
                f"✔ LLM {schema_name}{ctx} · ↓{_fmt_tokens(input_tokens)} ↑{_fmt_tokens(output_tokens)} · {_fmt_duration(duration_s)}",
                flush=True,
            )

    def scenario_start(self, scenario_id: str) -> None:
        super().scenario_start(scenario_id)
        print(f"→ scenario {scenario_id}", flush=True)

    def scenario_end(self, scenario_id: str, detail: str = "") -> None:
        super().scenario_end(scenario_id, detail)
        extra = f" — {detail}" if detail else ""
        print(f"✔ scenario {scenario_id}{extra}", flush=True)

    def alert(
        self,
        code: str,
        message: str,
        *,
        severity: str = "critical",
        scenario_id: str = "",
        clause: str = "",
        formula_description: str = "",
        quote_preview: str = "",
        docs: list[str] | None = None,
    ) -> None:
        super().alert(
            code,
            message,
            severity=severity,
            scenario_id=scenario_id,
            clause=clause,
            formula_description=formula_description,
            quote_preview=quote_preview,
            docs=docs,
        )
        loc = f"{scenario_id}/{clause}" if scenario_id else ""
        label = "WARNING" if severity == "warning" else "ALERT"
        line = f"{label} {code} {loc} — {message}".strip()
        if docs:
            line = f"{line} · docs={','.join(docs)}"
        print(line, flush=True)

    def finish(self, summary: str = "") -> None:
        t = self.tracker
        alert_n = len(t.alerts)
        alert_bit = f" · {alert_n} OPERATOR ALERTS" if alert_n else ""
        print(
            f"done · {_fmt_duration(t.elapsed)} · "
            f"tokens ↓{_fmt_tokens(t.input_tokens)} ↑{_fmt_tokens(t.output_tokens)} · "
            f"cache {t.cache_hits}/{t.cache_hits + t.cache_misses}"
            + alert_bit
            + (f" · {summary}" if summary else ""),
            flush=True,
        )
        print(t.format_cost_report(), flush=True)


class RecordingSink(NullSink):
    """Captures events for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple] = []

    def stage_start(self, stage: str, detail: str = "") -> None:
        self.events.append(("stage_start", stage, detail))
        super().stage_start(stage, detail)

    def stage_end(self, stage: str, detail: str = "") -> None:
        self.events.append(("stage_end", stage, detail))
        super().stage_end(stage, detail)

    def log(self, message: str) -> None:
        self.events.append(("log", message))
        super().log(message)

    def alert(
        self,
        code: str,
        message: str,
        *,
        severity: str = "critical",
        scenario_id: str = "",
        clause: str = "",
        formula_description: str = "",
        quote_preview: str = "",
        docs: list[str] | None = None,
    ) -> None:
        self.events.append(("alert", code, scenario_id, clause, message))
        super().alert(
            code,
            message,
            severity=severity,
            scenario_id=scenario_id,
            clause=clause,
            formula_description=formula_description,
            quote_preview=quote_preview,
            docs=docs,
        )

    def set_stage_payload(self, stage: str, payload: dict) -> None:
        self.events.append(("set_stage_payload", stage, payload))
        super().set_stage_payload(stage, payload)

    def log_inventory(self, tag: str, lines: list[str]) -> None:
        self.events.append(("log_inventory", tag, len(lines)))
        super().log_inventory(tag, lines)

    def llm_wait(self, schema_name: str, *, label: str = "") -> None:
        self.events.append(("llm_wait", schema_name, label))
        super().llm_wait(schema_name, label=label)

    def llm_done(
        self,
        schema_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        duration_s: float = 0.0,
        label: str = "",
    ) -> None:
        self.events.append(
            ("llm_done", schema_name, input_tokens, output_tokens, cache_hit, duration_s, label)
        )
        super().llm_done(
            schema_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            duration_s=duration_s,
            label=label,
        )

    def scenario_start(self, scenario_id: str) -> None:
        self.events.append(("scenario_start", scenario_id))
        super().scenario_start(scenario_id)

    def scenario_end(self, scenario_id: str, detail: str = "") -> None:
        self.events.append(("scenario_end", scenario_id, detail))
        super().scenario_end(scenario_id, detail)


def _status_cell(st: StageState) -> str:
    if st.status == "DONE":
        return f"[{_GREEN}]✔ DONE[/{_GREEN}]"
    if st.status == "RUNNING":
        return f"[{_YELLOW}]◒ RUNNING[/{_YELLOW}]"
    return "[dim]· IDLE[/dim]"


def _latency_cell(st: StageState) -> str:
    if st.status == "RUNNING" and st.started_at is not None:
        return _fmt_duration(time.monotonic() - st.started_at)
    if st.status == "DONE":
        return _fmt_duration(st.elapsed)
    return "—"


def render_dashboard(tracker: ProgressTracker) -> Group:
    header = Table.grid(expand=True)
    header.add_column(justify="left", ratio=1)
    header.add_column(justify="right", ratio=1)
    header.add_row(
        f"[bold {_ORANGE}]AI Covenant Auditor[/bold {_ORANGE}] [dim]· {tracker.model_name or 'model?'}[/dim]",
        f"[dim]elapsed[/dim] [bold {_YELLOW}]{_fmt_duration(tracker.elapsed)}[/bold {_YELLOW}]  "
        f"[dim]↓[/dim][bold {_ORANGE}]{_fmt_tokens(tracker.input_tokens)}[/bold {_ORANGE}] "
        f"[dim]↑[/dim][bold {_ORANGE}]{_fmt_tokens(tracker.output_tokens)}[/bold {_ORANGE}]",
    )
    header.add_row(
        f"[dim]out[/dim] {tracker.out_path or '—'}",
        f"[dim]status[/dim] [{_ORANGE}]{tracker.status_line}[/{_ORANGE}]",
    )

    stages = Table(expand=True, box=None, show_header=True, header_style=f"bold {_ORANGE}")
    stages.add_column("Stage", style="bold white")
    stages.add_column("Status", justify="center")
    stages.add_column("Details", ratio=2)
    stages.add_column("Latency", justify="right", style="dim")
    for key in STAGE_ORDER:
        st = tracker.stages[key]
        stages.add_row(STAGE_LABELS[key], _status_cell(st), st.detail or "—", _latency_cell(st))

    total_llm = tracker.cache_hits + tracker.cache_misses
    hit_rate = f"{100 * tracker.cache_hits / total_llm:.0f}%" if total_llm else "—"
    scen = (
        f"{tracker.scenarios_done}/{tracker.scenarios_total}"
        if tracker.scenarios_total
        else str(tracker.scenarios_done)
    )
    metrics = Text.from_markup(
        f"[bold {_ORANGE}]TELEMETRY[/bold {_ORANGE}]\n\n"
        f"• Tokens in:  [bold white]{_fmt_tokens(tracker.input_tokens)}[/bold white]\n"
        f"• Tokens out: [bold white]{_fmt_tokens(tracker.output_tokens)}[/bold white]\n"
        f"• LLM calls:  [bold white]{tracker.llm_calls}[/bold white]\n"
        f"• Cache hit:  [bold white]{hit_rate}[/bold white] "
        f"[dim]({tracker.cache_hits}/{total_llm})[/dim]\n"
        f"• Scenarios:  [bold white]{scen}[/bold white]\n"
        f"• Current:    [bold white]{tracker.current_scenario or '—'}[/bold white]"
    )

    main = Table.grid(expand=True)
    main.add_column(ratio=2)
    main.add_column(ratio=1)
    main.add_row(
        Panel(stages, title="[bold white]Pipeline Stages[/bold white]", border_style="#444444"),
        Panel(metrics, title="[bold white]Metrics[/bold white]", border_style="#444444"),
    )

    log_text = Text()
    for line in tracker.log_lines[-_LOG_KEEP:]:
        style = "dim"
        if "[ALERT]" in line:
            style = f"bold {_ORANGE}"
        elif "[LLM]" in line:
            style = "dim cyan"
        elif "[CASE]" in line:
            style = "dim green"
        elif "FAIL" in line or "error" in line.lower():
            style = f"bold {_ORANGE}"
        log_text.append(line + "\n", style=style)

    logs = Panel(
        log_text if tracker.log_lines else Text("[dim]waiting for events…[/dim]"),
        title="[bold white]Real-time Execution Stream[/bold white]",
        border_style=_ORANGE,
    )

    panels: list = [
        Panel(header, style="on #1E1E1E", border_style="#444444"),
        main,
    ]
    if tracker.alerts:
        alert_text = Text()
        for a in tracker.alerts[-8:]:
            loc = f"{a.get('scenario_id', '')}/{a.get('clause', '')}".strip("/")
            sev = a.get("severity", "critical")
            sev_bit = f"[{sev}] " if sev else ""
            line = f"{sev_bit}{a.get('code', '?')} {loc} — {a.get('message', '')}"
            docs = a.get("docs") or []
            if docs:
                line += f" · docs={','.join(docs)}"
            if a.get("formula_description"):
                line += f" | {a['formula_description'][:80]}"
            alert_text.append(line + "\n", style=f"bold {_ORANGE}")
        panels.append(
            Panel(
                alert_text,
                title=f"[bold {_ORANGE}]OPERATOR ALERTS ({len(tracker.alerts)})[/bold {_ORANGE}]",
                border_style=_ORANGE,
            )
        )
    panels.append(logs)
    return Group(*panels)


class RichProgressUI(NullSink):
    """Live Rich dashboard — display only, never reads stdin."""

    def __init__(self, *, model_name: str = "", out_path: str = "", console: Console | None = None) -> None:
        super().__init__()
        self.tracker.model_name = model_name
        self.tracker.out_path = out_path
        self._console = console or Console(stderr=False)
        # get_renderable re-reads tracker each tick so elapsed/latency keep counting
        # during long LLM waits (static update() snapshots would freeze the clock).
        self._live: Live | None = Live(
            console=self._console,
            auto_refresh=True,
            refresh_per_second=1,
            transient=False,
            get_renderable=lambda: render_dashboard(self.tracker),
        )
        self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def stage_start(self, stage: str, detail: str = "") -> None:
        super().stage_start(stage, detail)
        self._refresh()

    def stage_end(self, stage: str, detail: str = "") -> None:
        super().stage_end(stage, detail)
        self._refresh()

    def log(self, message: str) -> None:
        super().log(message)
        self._refresh()

    def set_stage_payload(self, stage: str, payload: dict) -> None:
        super().set_stage_payload(stage, payload)
        self._refresh()

    def log_inventory(self, tag: str, lines: list[str]) -> None:
        super().log_inventory(tag, lines)
        self._refresh()

    def llm_wait(self, schema_name: str, *, label: str = "") -> None:
        super().llm_wait(schema_name, label=label)
        self._refresh()

    def llm_done(
        self,
        schema_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        duration_s: float = 0.0,
        label: str = "",
    ) -> None:
        super().llm_done(
            schema_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            duration_s=duration_s,
            label=label,
        )
        self._refresh()

    def scenario_start(self, scenario_id: str) -> None:
        super().scenario_start(scenario_id)
        self._refresh()

    def scenario_end(self, scenario_id: str, detail: str = "") -> None:
        super().scenario_end(scenario_id, detail)
        self._refresh()

    def alert(
        self,
        code: str,
        message: str,
        *,
        severity: str = "critical",
        scenario_id: str = "",
        clause: str = "",
        formula_description: str = "",
        quote_preview: str = "",
        docs: list[str] | None = None,
    ) -> None:
        super().alert(
            code,
            message,
            severity=severity,
            scenario_id=scenario_id,
            clause=clause,
            formula_description=formula_description,
            quote_preview=quote_preview,
            docs=docs,
        )
        self._refresh()

    def set_scenarios_total(self, total: int) -> None:
        super().set_scenarios_total(total)
        self._refresh()

    def finish(self, summary: str = "") -> None:
        n = len(self.tracker.alerts)
        status = summary or "complete"
        if n:
            status = f"{status} · {n} OPERATOR ALERTS" if summary else f"{n} OPERATOR ALERTS"
        self.tracker.status_line = status
        self._refresh()
        if self._live is not None:
            self._live.stop()
            self._live = None
        t = self.tracker
        self._console.print(_build_finish_summary_panel(t, summary=summary))
        self._console.print(_render_cost_table(t))


def _build_finish_summary_panel(tracker: ProgressTracker, *, summary: str = "") -> Panel:
    """TTY finish Summary — same visual language as tools.score._build_summary_panel."""
    rows = tracker.schema_cost_rows()
    billed_calls = sum(r["calls"] for r in rows)
    in_tok = sum(r["input_tokens"] for r in rows)
    out_tok = sum(r["output_tokens"] for r in rows)
    total_usd = sum(r["usd"] for r in rows)
    model_label = _model_short_label(tracker.model_name)
    cache_total = tracker.cache_hits + tracker.cache_misses
    alert_n = len(tracker.alerts)

    alert_note = ""
    if alert_n:
        alert_note = f"  [{_YELLOW}]← {alert_n} OPERATOR ALERTS[/{_YELLOW}]"

    top = rows[0] if rows else None
    top_line = "—"
    if top is not None:
        top_line = (
            f"[bold white]{top['schema']}[/bold white]  "
            f"[dim]${top['usd']:.2f} · {round(top['share'] * 100):.0f}%[/dim]"
        )

    hit_style = _GREEN
    miss_style = _RED if tracker.cache_misses else _GREEN
    out_line = summary.strip() if summary.strip() else "—"
    nl = chr(10)
    body = Text.from_markup(
        f"[bold {_ORANGE}]DONE[/bold {_ORANGE}]  "
        f"[bold white]{_fmt_duration(tracker.elapsed)}[/bold white]  "
        f"[dim]tokens ↓{_fmt_tokens(tracker.input_tokens)} ↑{_fmt_tokens(tracker.output_tokens)}[/dim]  "
        f"[dim]cache {tracker.cache_hits}/{cache_total}[/dim]  "
        f"[dim]$ {total_usd:.2f}[/dim]{alert_note}"
        + nl
        + nl
        + f"• Calls:    [bold white]{billed_calls}[/bold white] billed  "
        f"[dim]·[/dim]  model [bold white]{model_label or '—'}[/bold white]"
        + nl
        + f"• Tokens:   in [bold white]{_fmt_tokens(in_tok)}[/bold white]  "
        f"out [bold white]{_fmt_tokens(out_tok)}[/bold white]"
        + nl
        + f"• Cache:    hit [{hit_style}]{tracker.cache_hits}[/{hit_style}]  "
        f"miss [{miss_style}]{tracker.cache_misses}[/{miss_style}]  "
        f"[dim]({cache_total} LLM calls)[/dim]"
        + nl
        + f"• Top:      {top_line}"
        + nl
        + f"• Output:   {out_line}"
    )
    return Panel(body, title="[bold white]Summary[/bold white]", border_style="#444444")


def _render_cost_table(tracker: ProgressTracker) -> Table:
    """Rich cost table for TTY finish — score-style: no box, orange headers."""
    rows = tracker.schema_cost_rows()
    billed_calls = sum(r["calls"] for r in rows)
    in_tok = sum(r["input_tokens"] for r in rows)
    out_tok = sum(r["output_tokens"] for r in rows)
    total_usd = sum(r["usd"] for r in rows)
    model_label = _model_short_label(tracker.model_name)
    table = Table(
        title=(
            f"[bold white]Куда уходят деньги · {_fmt_tokens(in_tok)} in / "
            f"{_fmt_tokens(out_tok)} out · {billed_calls} вызовов[/bold white]"
        ),
        title_style=f"bold {_ORANGE}",
        expand=False,
        box=None,
        show_header=True,
        header_style=f"bold {_ORANGE}",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Стадия", style="bold white", no_wrap=True)
    table.add_column("Вызовов", justify="right", no_wrap=True)
    table.add_column("Input", justify="right", no_wrap=True)
    table.add_column("Output", justify="right", no_wrap=True)
    table.add_column(f"$ ({model_label})", justify="right", no_wrap=True)
    table.add_column("Доля", justify="right", no_wrap=True)
    for r in rows:
        share_pct = round(r["share"] * 100)
        share_style = _ORANGE if share_pct >= 40 else ("white" if share_pct >= 15 else "dim")
        table.add_row(
            r["schema"],
            str(r["calls"]),
            _fmt_tokens(r["input_tokens"]),
            _fmt_tokens(r["output_tokens"]),
            f"{r['usd']:.2f}",
            Text(f"{share_pct:.0f}%", style=share_style),
        )
    table.add_row(
        Text("Итого", style="bold white"),
        Text(str(billed_calls), style="bold white"),
        Text(_fmt_tokens(in_tok), style="bold white"),
        Text(_fmt_tokens(out_tok), style="bold white"),
        Text(f"{total_usd:.2f}", style=f"bold {_ORANGE}"),
        Text("100%" if rows else "0%", style="bold white"),
    )
    return table


def _write_artifacts(tracker: ProgressTracker, out_dir: Path) -> None:
    tracker.log_cost_report()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace.json").write_text(
        json.dumps(tracker.snapshot_trace(), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    log_path = out_dir / "agent.log"
    log_path.write_text("\n".join(tracker.log_lines) + ("\n" if tracker.log_lines else ""), encoding="utf-8")


def create_progress_sink(
    *,
    model_name: str = "",
    out_path: str = "",
    force_quiet: bool = False,
) -> ProgressSink:
    """TUI when stdout is a TTY; QuietSink for pipes/CI/--no-tui/AGENT_NO_TUI."""
    env_quiet = os.environ.get("AGENT_NO_TUI", "").strip() in {"1", "true", "yes"}
    use_quiet = force_quiet or env_quiet or not sys.stdout.isatty()
    if use_quiet:
        sink: ProgressSink = QuietSink()
        sink.tracker.model_name = model_name
        sink.tracker.out_path = out_path
        return sink
    return RichProgressUI(model_name=model_name, out_path=out_path)
