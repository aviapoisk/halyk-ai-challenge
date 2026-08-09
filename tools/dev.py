"""Dev-only TUI: run the agent with --dev and walk through scenarios like an operator.

Never imported from agent.run (CLAUDE.md: dev triage tools live outside the battle
path; read-only — no answer editing of any kind).

Usage:
    uv run python -m tools.dev --dataset data/agentic-bank-public
    uv run python -m tools.dev --dataset data/agentic-bank-public --browse   # skip the run, open last dump
    uv run python -m tools.dev --dataset data/agentic-bank-public --only P3 --no-cache

Keys: Enter open · Esc back · q quit · r re-run agent · u unlinked docs
Transactions tab: c cycle category filter · e evidence/contributing only · type in the filter box
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static, TabbedContent, TabPane

from tools.score import cell_score, normalize_gt

_CATEGORY_CYCLE = (None, "revenue", "operating_expense", "capex", "financing", "other", "uncategorized")

# Match agent/progress.py Rich dashboard palette.
_ORANGE = "#D96B27"
_GREEN = "#4EBF71"
_YELLOW = "#E5C07B"
_RED = "#E06C75"
_BG = "#1E1E1E"
_BORDER = "#444444"

# Same stamp as agent/s2_classify._SUPERSEDED_PATTERN (+ soft fallbacks for quote snippets).
_SUPERSEDED_STAMP = re.compile(r"НЕДЕЙСТВУЮЩАЯ\s+РЕДАКЦИЯ", re.IGNORECASE)
_SUPERSEDED_FALLBACK = re.compile(r"заменена|superseded", re.IGNORECASE)
_CLAUSE_NORM_RE = re.compile(r"clause key normalized:.*->\s*'([^']+)'")
_CLAUSE_BRACKET_RE = re.compile(r"^\[([^\]]+)\]\s*")
_QUOTE_IN_FLAG_RE = re.compile(r"quote='([^']*)'")

AUDITOR_THEME = Theme(
    name="auditor",
    primary=_ORANGE,
    secondary="#A04E1C",
    warning=_YELLOW,
    error=_ORANGE,
    success=_GREEN,
    accent=_ORANGE,
    foreground="#E8E8E8",
    background=_BG,
    surface="#252525",
    panel="#2A2A2A",
    dark=True,
)


# --------------------------------------------------------------------------- data


class DevData:
    """Read-only view over logs/dev_dump.json + logs/trace.json + confidence + dataset GT."""

    def __init__(
        self,
        dump: dict,
        trace: dict,
        gt: dict | None,
        submission: dict | None,
        confidence: dict | None = None,
    ):
        self.dump = dump
        self.trace = trace
        self.gt_by_scenario = normalize_gt(gt) if gt else {}
        self.submission = submission or {}
        self.confidence = confidence or {}

    @classmethod
    def load(
        cls,
        dataset: Path,
        logs_dir: Path,
        out_path: Path,
        confidence_path: Path | None = None,
    ) -> "DevData":
        dump = json.loads((logs_dir / "dev_dump.json").read_text(encoding="utf-8"))
        trace_path = logs_dir / "trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        conf_path = confidence_path or (logs_dir / "confidence_report.json")
        confidence = None
        if conf_path.exists():
            confidence = json.loads(conf_path.read_text(encoding="utf-8"))
        gt_path = dataset / "ground_truth.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
        submission = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
        return cls(dump, trace, gt, submission, confidence)

    @property
    def has_gt(self) -> bool:
        return bool(self.gt_by_scenario)

    @property
    def scenario_ids(self) -> list[str]:
        return list(self.dump.get("scenario_ids", []))

    def case(self, sid: str) -> dict:
        return (self.trace.get("cases") or {}).get(sid, {})

    def dev_case(self, sid: str) -> dict:
        return (self.dump.get("scenarios") or {}).get(sid, {})

    def submitted_cells(self, sid: str) -> dict:
        return (self.submission.get("answers") or {}).get(sid, {})

    def gt_cells(self, sid: str) -> dict:
        return self.gt_by_scenario.get(sid, {})

    def scenario_score(self, sid: str) -> tuple[float, float] | None:
        """(score, max) over GT clauses, official lenient formula. None without GT."""
        gt_cells = self.gt_cells(sid)
        if not gt_cells:
            return None
        submitted = self.submitted_cells(sid)
        total = 0.0
        for clause, key in gt_cells.items():
            score, _err = cell_score(submitted.get(clause), key)
            total += score
        return total, float(len(gt_cells))

    def scenario_risk(self, sid: str) -> int:
        """Higher = worse. Used when GT is absent (private set triage)."""
        case = self.case(sid)
        cells = case.get("cells") or {}
        stubbed = sum(1 for c in cells.values() if c.get("outcome") == "stubbed")
        flags = len(case.get("flags") or [])
        status = case.get("status") or ""
        risk = stubbed * 100 + flags * 10
        if status == "failed":
            risk += 1000
        alert_hit = any(a.get("scenario_id") == sid for a in self.alerts)
        if alert_hit:
            risk += 200
        binding = case.get("binding") or {}
        if not binding.get("account_id"):
            risk += 50
        cat = case.get("categorization") or {}
        n_rows = int(cat.get("n_rows") or 0)
        n_labeled = int(cat.get("n_labeled") or 0)
        if n_rows and n_labeled < n_rows:
            risk += (n_rows - n_labeled) * 2
        return risk

    def run_health(self) -> dict:
        """Aggregate operator health from confidence + trace (works without GT)."""
        conf = self.confidence
        cases = self.trace.get("cases") or {}
        processed = sum(1 for c in cases.values() if c.get("status") == "processed")
        failed = sum(1 for c in cases.values() if c.get("status") == "failed")
        stubs = 0
        for c in cases.values():
            stubs += sum(1 for cell in (c.get("cells") or {}).values() if cell.get("outcome") == "stubbed")
        unlinked = self.dump.get("unlinked_documents") or []
        return {
            "validation_ok": conf.get("validation_ok"),
            "sanitize_fixes": list(conf.get("sanitize_fixes") or self.dump.get("sanitize_fixes") or []),
            "ingest_failures": list(conf.get("ingest_failures") or self.dump.get("ingest_failures") or []),
            "linking_warnings": conf.get("linking_warnings") or [],
            "account_map_warnings": conf.get("account_map_warnings") or [],
            "n_unlinked": len(unlinked),
            "n_processed": processed,
            "n_failed": failed,
            "n_stubs": stubs,
            "n_alerts": len(self.alerts),
            "has_gt": self.has_gt,
        }

    def txn_clause_map(self, sid: str) -> dict[str, set[str]]:
        """txn_id -> clause keys where it contributed / was chosen as evidence."""
        out: dict[str, set[str]] = {}
        for clause, cell in (self.case(sid).get("cells") or {}).items():
            ids = set(cell.get("contributing_txn_ids") or [])
            evidence = (cell.get("evidence") or {}).get("chosen")
            if evidence:
                ids.add(evidence)
            for txn_id in ids:
                out.setdefault(str(txn_id), set()).add(clause)
        return out

    @property
    def alerts(self) -> list[dict]:
        """Operator alerts from the last run (trace.json → confidence_report)."""
        raw = self.trace.get("alerts")
        if isinstance(raw, list) and raw:
            return list(raw)
        raw = self.confidence.get("alerts")
        return list(raw) if isinstance(raw, list) else []


def _format_run_health(health: dict) -> Text:
    text = Text()
    if not health.get("has_gt"):
        text.append("режим без GT · триаж по флагам/риску\n", style=f"bold {_YELLOW}")
    vok = health.get("validation_ok")
    vok_s = "ok" if vok else ("fail" if vok is False else "?")
    vok_style = _GREEN if vok else _ORANGE
    text.append(f"validation={vok_s}", style=f"bold {vok_style}")
    text.append(
        f" · processed={health.get('n_processed', 0)} failed={health.get('n_failed', 0)} "
        f"stubs={health.get('n_stubs', 0)} alerts={health.get('n_alerts', 0)} "
        f"unlinked={health.get('n_unlinked', 0)}\n",
        style="dim",
    )
    fixes = health.get("sanitize_fixes") or []
    if fixes:
        text.append(f"sanitize_fixes ({len(fixes)}): {fixes[0][:100]}\n", style=f"bold {_ORANGE}")
    fails = health.get("ingest_failures") or []
    if fails:
        text.append(f"ingest_failures ({len(fails)}): {fails[0][:100]}\n", style=f"bold {_ORANGE}")
    return text


def _fmt_num(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value}"


def _status_markup(status: str | None) -> str:
    if status == "BREACH":
        return f"[bold {_ORANGE}]{status}[/]"
    if status == "COMPLIANT":
        return f"[bold {_GREEN}]{status}[/]"
    return escape(str(status))


def _format_operator_alerts(alerts: list[dict]) -> Text:
    """Same line shape as agent/progress.py OPERATOR ALERTS panel."""
    text = Text()
    if not alerts:
        text.append("нет алертов", style="dim")
        return text
    for a in alerts:
        loc = f"{a.get('scenario_id', '')}/{a.get('clause', '')}".strip("/")
        sev = a.get("severity", "critical")
        sev_bit = f"[{sev}] " if sev else ""
        line = f"{sev_bit}{a.get('code', '?')} {loc} — {a.get('message', '')}"
        docs = a.get("docs") or []
        if docs:
            line += f" · docs={','.join(docs)}"
        if a.get("formula_description"):
            line += f" | {a['formula_description'][:80]}"
        text.append(line + "\n", style=f"bold {_ORANGE}")
    return text


def _doc_by_id(documents: list[dict], doc_id: str | None) -> dict | None:
    if not doc_id:
        return None
    for doc in documents:
        if doc.get("doc_id") == doc_id:
            return doc
    return None


def _doc_filename(doc_or_facts: dict | None) -> str:
    if not doc_or_facts:
        return "—"
    path = doc_or_facts.get("path")
    if path:
        return Path(str(path)).name
    doc_id = doc_or_facts.get("doc_id")
    return f"{doc_id}.pdf" if doc_id else "—"


def _snippet_around(match: re.Match[str], text: str, radius: int = 40) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    snippet = " ".join(text[start:end].split())
    return snippet[:120]


def _superseded_stamp_quote(text: str) -> str | None:
    for pattern in (_SUPERSEDED_STAMP, _SUPERSEDED_FALLBACK):
        match = pattern.search(text or "")
        if match:
            return _snippet_around(match, text or "")
    return None


def _edition_status(facts: dict, loan_all: list[dict] | None = None) -> str:
    parts: list[str] = []
    selected = bool(facts.get("selected"))
    if not selected and loan_all is not None and not facts.get("is_superseded"):
        # Older dumps omit `selected`; treat the sole non-superseded edition as selected.
        if facts.get("selected") is None and sum(1 for f in loan_all if not f.get("is_superseded")) == 1:
            selected = True
    if selected:
        parts.append("selected")
    if facts.get("is_superseded"):
        parts.append("superseded")
    return ", ".join(parts) if parts else "—"


def _edition_reason(facts: dict, documents: list[dict], loan_all: list[dict]) -> str:
    """Why this edition is superseded / not selected — from stamp text + selection context."""
    doc_id = facts.get("doc_id")
    doc = _doc_by_id(documents, doc_id)
    text = (doc or {}).get("text") or ""
    selected = next((f for f in loan_all if f.get("selected")), None)
    if selected is None:
        selected = next((f for f in loan_all if not f.get("is_superseded")), None)

    def _same(a: dict, b: dict) -> bool:
        if a is b:
            return True
        if a.get("doc_id") and a.get("doc_id") == b.get("doc_id"):
            return True
        return (
            a.get("contract_date") == b.get("contract_date")
            and a.get("effective_from") == b.get("effective_from")
            and a.get("effective_to") == b.get("effective_to")
        )

    if facts.get("is_superseded"):
        quote = _superseded_stamp_quote(text)
        if quote:
            return f"штамп: «{quote}»"
        return "LLM: is_superseded=true (штамп в тексте не найден)"

    if selected and _same(facts, selected):
        return "выбрана как действующая (не superseded)"
    if selected:
        sel_file = _doc_filename(selected)
        return (
            f"не выбрана; рабочая редакция = {sel_file} "
            f"({selected.get('effective_from')}—{selected.get('effective_to')})"
        )
    return "—"


def _loan_selected_doc_id(extracts: dict, documents: list[dict] | None = None) -> str | None:
    selected = extracts.get("loan_selected") or {}
    if selected.get("doc_id"):
        return str(selected["doc_id"])
    for facts in extracts.get("loan_agreements_all") or []:
        if facts.get("selected") or (not facts.get("is_superseded") and facts.get("doc_id")):
            if facts.get("doc_id") and (facts.get("selected") or not facts.get("is_superseded")):
                return str(facts["doc_id"])
    docs = documents or []
    loan_docs = [d for d in docs if d.get("type") == "loan_agreement"]
    for doc in loan_docs:
        if doc.get("doc_id") and not _superseded_stamp_quote(str(doc.get("text") or "")):
            return str(doc["doc_id"])
    if loan_docs and loan_docs[0].get("doc_id"):
        return str(loan_docs[0]["doc_id"])
    return None


def _covenant_quote(extracts: dict, clause: str) -> str:
    for source in (extracts.get("loan_selected"), *(extracts.get("loan_agreements_all") or [])):
        if not source:
            continue
        for cov in source.get("covenants") or []:
            if str(cov.get("clause")) == clause and cov.get("quote"):
                return str(cov["quote"])
    return ""


def _auditor_doc_id(documents: list[dict]) -> str | None:
    for doc in documents:
        if doc.get("type") == "auditor_notes":
            return str(doc.get("doc_id")) if doc.get("doc_id") else None
    return None


def _enrich_flag(flag: str, extracts: dict, documents: list[dict]) -> tuple[list[str], str]:
    """Attach doc_ids + quote to a flat pipeline flag string for Overview display."""
    text = str(flag)
    loan_doc = _loan_selected_doc_id(extracts, documents)

    m = _CLAUSE_NORM_RE.search(text)
    if m:
        clause = m.group(1)
        quote = _covenant_quote(extracts, clause)
        return ([loan_doc] if loan_doc else []), quote[:160]

    lower = text.lower()
    if "FX" in text or "EUR present" in text or "eur_amount" in lower or "converted" in lower:
        attempts = extracts.get("fx_attempts") or []
        doc_ids = [str(a["doc_id"]) for a in attempts if a.get("doc_id")]
        quotes = [str(a.get("quote") or "") for a in attempts if a.get("quote")]
        if quotes:
            return doc_ids, quotes[0][:160]
        if attempts:
            return doc_ids, "rate null / quote empty"
        return doc_ids, ""

    if "reclass" in lower:
        auditor = extracts.get("auditor") or {}
        recs = auditor.get("reclassifications") or []
        quotes = [str(r.get("quote") or "") for r in recs if r.get("quote")]
        aud_id = _auditor_doc_id(documents)
        doc_ids = [aud_id] if aud_id else []
        return doc_ids, (quotes[0][:160] if quotes else "")

    m = _CLAUSE_BRACKET_RE.match(text)
    if m:
        clause = m.group(1)
        inline = _QUOTE_IN_FLAG_RE.search(text)
        quote = inline.group(1) if inline else _covenant_quote(extracts, clause)
        return ([loan_doc] if loan_doc else []), quote[:160]

    return [], ""


# --------------------------------------------------------------------------- screens


class RunScreen(Screen):
    """Runs `python -m agent.run --dev` as a subprocess, streaming its output."""

    BINDINGS = [
        Binding("enter", "proceed", "К сценариям", show=True),
        Binding("q", "app.quit", "Выход", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._done_ok = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(highlight=False, wrap=True, id="runlog", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "AI Covenant Auditor · agent.run --dev"
        self.sub_title = "Real-time Execution Stream"
        self.run_worker(self._run_agent(), exclusive=True)

    async def _run_agent(self) -> None:
        app: DevTuiApp = self.app  # type: ignore[assignment]
        log = self.query_one("#runlog", RichLog)
        cmd = [
            sys.executable,
            "-m",
            "agent.run",
            "--dataset",
            str(app.args.dataset),
            "--out",
            str(app.args.out),
            "--dev",
            "--no-tui",
        ]
        if app.args.only:
            cmd += ["--only", app.args.only]
        if app.args.no_cache:
            cmd += ["--no-cache"]
        log.write(Text(" ".join(cmd), style=f"bold {_ORANGE}"))
        env = dict(os.environ, AGENT_NO_TUI="1")
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            style = "dim"
            if "[ALERT]" in decoded or "ALERT " in decoded or "FAIL" in decoded or "error" in decoded.lower():
                style = f"bold {_ORANGE}"
            elif "[LLM]" in decoded:
                style = "dim cyan"
            elif "[CASE]" in decoded:
                style = "dim green"
            log.write(Text(decoded, style=style))
        code = await proc.wait()
        if code == 0:
            self._done_ok = True
            log.write(Text("✔ done — Enter, чтобы открыть браузер сценариев.", style=f"bold {_GREEN}"))
            self.action_proceed()
        else:
            log.write(Text(f"agent.run завершился с кодом {code} — смотри лог выше.", style=f"bold {_ORANGE}"))
            # A dump may still exist from an earlier run; allow browsing it.
            self._done_ok = True

    def action_proceed(self) -> None:
        if not self._done_ok:
            return
        app: DevTuiApp = self.app  # type: ignore[assignment]
        if app.reload_data():
            app.switch_screen(ScenarioListScreen())


class ScenarioListScreen(Screen):
    BINDINGS = [
        Binding("enter", "open", "Открыть", show=True),
        Binding("u", "unlinked", "Непривязанные документы", show=True),
        Binding("r", "rerun", "Перезапустить агента", show=True),
        Binding("s", "sort_risk", "Сорт. по риску", show=True),
        Binding("q", "app.quit", "Выход", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sort_by_risk = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="scenario-list-body"):
            yield Static(id="run-health", classes="alerts-panel")
            yield Static(id="operator-alerts", classes="alerts-panel")
            yield DataTable(id="scenarios", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        data = self._data()
        model = data.dump.get("model") or data.trace.get("model") or ""
        self.title = "AI Covenant Auditor"
        mode = "без GT" if not data.has_gt else "vs GT"
        self.sub_title = f"сценарии · {mode} · {model}" if model else f"сценарии · {mode}"

        health = data.run_health()
        health_w = self.query_one("#run-health", Static)
        health_w.update(_format_run_health(health))
        health_w.border_title = "RUN HEALTH"
        health_w.remove_class("-empty")

        alerts = data.alerts
        alerts_widget = self.query_one("#operator-alerts", Static)
        if alerts:
            alerts_widget.update(_format_operator_alerts(alerts))
            alerts_widget.border_title = f"OPERATOR ALERTS ({len(alerts)})"
            alerts_widget.remove_class("-empty")
        else:
            alerts_widget.update(Text("нет алертов", style="dim"))
            alerts_widget.border_title = "OPERATOR ALERTS (0)"
            alerts_widget.add_class("-empty")

        table = self.query_one("#scenarios", DataTable)
        table.clear(columns=True)
        score_hdr = "Скор" if data.has_gt else "Риск"
        table.add_columns("SID", "Статус", "Заёмщик", "Счёт", "Док-ты", "Транз.", "Вычислено", "Стабы", "Флаги", score_hdr)
        alert_sids = {str(a.get("scenario_id")) for a in alerts if a.get("scenario_id")}

        sids = list(data.scenario_ids)
        if self._sort_by_risk and not data.has_gt:
            sids = sorted(sids, key=lambda s: -data.scenario_risk(s))
        elif self._sort_by_risk and data.has_gt:
            def _key(s: str) -> float:
                sc = data.scenario_score(s)
                return -(sc[0] if sc else 0.0)
            sids = sorted(sids, key=_key)

        for sid in sids:
            case = data.case(sid)
            dev = data.dev_case(sid)
            cells = case.get("cells") or {}
            computed = sum(1 for c in cells.values() if c.get("outcome") == "computed")
            stubbed = sum(1 for c in cells.values() if c.get("outcome") == "stubbed")
            binding = case.get("binding") or {}
            status = case.get("status") or "?"
            status_markup = {
                "processed": f"[{_GREEN}]✔ processed[/{_GREEN}]",
                "failed": f"[bold {_ORANGE}]failed[/]",
            }.get(status, f"[{_YELLOW}]{escape(status)}[/{_YELLOW}]")
            has_alert = sid in alert_sids
            sid_cell = f"[bold {_RED}]{escape(sid)}[/]" if has_alert else escape(sid)
            if data.has_gt:
                score = data.scenario_score(sid)
                score_cell = "—" if score is None else f"{score[0]:.2f}/{score[1]:.0f}"
                if has_alert and score is not None:
                    score_cell = f"[bold {_RED}]{score_cell}[/]"
            else:
                risk = data.scenario_risk(sid)
                score_cell = f"[bold {_ORANGE}]{risk}[/]" if risk >= 100 else str(risk)
            table.add_row(
                sid_cell,
                status_markup,
                escape(str(binding.get("borrower_name") or "—")),
                escape(str(binding.get("account_id") or "—")),
                str(len(dev.get("documents") or [])),
                str(len(dev.get("transactions") or [])),
                str(computed),
                str(stubbed) if not stubbed else f"[bold {_ORANGE}]{stubbed}[/]",
                str(len(case.get("flags") or [])),
                score_cell,
                key=sid,
            )
        table.focus()

    def _data(self) -> DevData:
        app: DevTuiApp = self.app  # type: ignore[assignment]
        assert app.data is not None
        return app.data

    def action_open(self) -> None:
        table = self.query_one("#scenarios", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        self.app.push_screen(ScenarioDetailScreen(str(row_key.value)))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(ScenarioDetailScreen(str(event.row_key.value)))

    def action_unlinked(self) -> None:
        self.app.push_screen(UnlinkedDocsScreen())

    def action_rerun(self) -> None:
        self.app.switch_screen(RunScreen())

    def action_sort_risk(self) -> None:
        self._sort_by_risk = not self._sort_by_risk
        self._rebuild()


class ScenarioDetailScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Назад", show=True),
        Binding("c", "cycle_category", "Категория-фильтр", show=True),
        Binding("e", "toggle_evidence", "Только участвующие", show=True),
        Binding("q", "app.quit", "Выход", show=True),
    ]

    def __init__(self, sid: str) -> None:
        super().__init__()
        self.sid = sid
        self._category_idx = 0
        self._contributing_only = False
        self._search = ""

    # ---- layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(initial="tab-overview"):
            with TabPane("Обзор", id="tab-overview"):
                with VerticalScroll(id="overview-scroll"):
                    yield Static(id="overview-header")
                    yield Static(id="overview-editions-title")
                    yield DataTable(id="overview-editions", cursor_type="row")
                    yield Static(id="overview-mid")
                    yield Static(id="overview-flags-title")
                    yield DataTable(id="overview-flags", cursor_type="row")
            with TabPane("Документы", id="tab-docs"):
                yield DataTable(id="docs", cursor_type="row")
            with TabPane("Ковенанты", id="tab-covenants"):
                yield VerticalScroll(Static(id="covenants"))
            with TabPane("Транзакции", id="tab-txns"):
                with Vertical():
                    yield Input(placeholder="фильтр: подстрока по txn/контрагенту/описанию", id="txn-search")
                    yield Static("", id="txn-filter-line")
                    yield DataTable(id="txns", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        data = self._data()
        binding = data.case(self.sid).get("binding") or {}
        self.title = f"{self.sid} · {binding.get('borrower_name') or '—'}"
        self._render_overview()
        self._render_docs()
        self._render_covenants()
        self._render_txns()

    def _data(self) -> DevData:
        app: DevTuiApp = self.app  # type: ignore[assignment]
        assert app.data is not None
        return app.data

    # ---- overview

    def _open_doc_by_id(self, doc_id: str | None) -> None:
        if not doc_id:
            self.app.notify("нет doc_id — перезапусти с --dev", severity="warning")
            return
        docs = self._data().dev_case(self.sid).get("documents") or []
        doc = _doc_by_id(docs, doc_id)
        if doc is None:
            self.app.notify(f"документ {doc_id} не найден в dump", severity="warning")
            return
        self.app.push_screen(DocViewScreen(doc))

    def _resolve_edition_doc_id(self, facts: dict, documents: list[dict], loan_all: list[dict]) -> str | None:
        """Prefer dump doc_id; for older dumps, match loan_agreement text heuristically."""
        if facts.get("doc_id"):
            return str(facts["doc_id"])
        loan_docs = [d for d in documents if d.get("type") == "loan_agreement"]
        if not loan_docs:
            return None
        # Prefer stamp match for superseded editions.
        if facts.get("is_superseded"):
            for doc in loan_docs:
                if _superseded_stamp_quote(str(doc.get("text") or "")):
                    return str(doc.get("doc_id")) if doc.get("doc_id") else None
        # Prefer non-stamp docs for current edition.
        for doc in loan_docs:
            if not _superseded_stamp_quote(str(doc.get("text") or "")):
                return str(doc.get("doc_id")) if doc.get("doc_id") else None
        if len(loan_all) == len(loan_docs):
            try:
                idx = loan_all.index(facts)
            except ValueError:
                idx = -1
            if 0 <= idx < len(loan_docs) and loan_docs[idx].get("doc_id"):
                return str(loan_docs[idx]["doc_id"])
        return str(loan_docs[0].get("doc_id")) if loan_docs[0].get("doc_id") else None

    def _render_overview(self) -> None:
        data = self._data()
        case = data.case(self.sid)
        dev = data.dev_case(self.sid)
        extracts = dev.get("extracts") or {}
        documents = list(dev.get("documents") or [])
        binding = case.get("binding") or {}
        cat = case.get("categorization") or {}

        header: list[str] = []
        header.append(f"[bold {_ORANGE}]Статус:[/] {escape(str(case.get('status')))}")
        header.append(
            f"[bold {_ORANGE}]Заёмщик:[/] {escape(str(binding.get('borrower_name') or '—'))}"
            f"   [bold {_ORANGE}]Счёт:[/] {escape(str(binding.get('account_id') or '—'))}"
        )
        score = data.scenario_score(self.sid)
        if score is not None:
            header.append(f"[bold {_ORANGE}]Скор vs GT (lenient):[/] {score[0]:.2f} / {score[1]:.0f}")
        else:
            header.append(
                f"[bold {_ORANGE}]Риск (без GT):[/] {data.scenario_risk(self.sid)}  "
                f"[dim]· выше = хуже[/]"
            )
        self.query_one("#overview-header", Static).update("\n".join(header))

        loan_all = list(extracts.get("loan_agreements_all") or [])
        editions_title = (
            f"[bold {_ORANGE}]Редакции договора:[/] {len(loan_all)}  [dim](Enter — открыть файл)[/]"
            if loan_all
            else f"[bold {_ORANGE}]Редакции договора:[/] нет"
        )
        self.query_one("#overview-editions-title", Static).update(editions_title)
        editions = self.query_one("#overview-editions", DataTable)
        editions.clear(columns=True)
        editions.add_columns(
            "Файл",
            "contract_date",
            "effective_from",
            "effective_to",
            "account_id",
            "ковенантов",
            "статус",
            "причина",
        )
        for i, facts in enumerate(loan_all):
            doc_id = self._resolve_edition_doc_id(facts, documents, loan_all)
            # Attach resolved id for open + stamp reason without mutating dump semantics.
            view = dict(facts)
            if doc_id and not view.get("doc_id"):
                view["doc_id"] = doc_id
            doc = _doc_by_id(documents, doc_id)
            filename = _doc_filename({"path": (doc or {}).get("path") or facts.get("path"), "doc_id": doc_id})
            editions.add_row(
                escape(filename),
                escape(str(facts.get("contract_date") or "—")),
                escape(str(facts.get("effective_from") or "—")),
                escape(str(facts.get("effective_to") or "—")),
                escape(str(facts.get("account_id") or "—")),
                str(len(facts.get("covenants") or [])),
                escape(_edition_status(facts, loan_all)),
                escape(_edition_reason(view, documents, loan_all)),
                key=f"ed-{i}:{doc_id or ''}",
            )

        mid: list[str] = []
        fx_attempts = extracts.get("fx_attempts") or []
        if fx_attempts:
            mid.append(f"[bold {_ORANGE}]FX-экстракция:[/]")
            for attempt in fx_attempts:
                rate = attempt.get("rate")
                rate_str = f"[{_GREEN}]{rate}[/{_GREEN}]" if rate is not None else f"[bold {_ORANGE}]null[/]"
                mid.append(
                    f"  · {escape(str(attempt.get('source')))} ({escape(str(attempt.get('doc_id')))}) → {rate_str}"
                    + (f" — «{escape(str(attempt.get('quote'))[:120])}»" if attempt.get("quote") else "")
                )
        else:
            mid.append(f"[bold {_ORANGE}]FX-экстракция:[/] не запускалась")
        auditor = extracts.get("auditor")
        mid.append("")
        if auditor:
            recs = auditor.get("reclassifications") or []
            mid.append(f"[bold {_ORANGE}]Аудитор:[/] переклассификаций {len(recs)}")
            for rec in recs:
                status = rec.get("status") or "applied"
                mid.append(
                    f"  · [{escape(str(status))}] {escape(str(rec.get('action')))} "
                    f"txn={escape(str(rec.get('txn_id')))} "
                    f"{escape(str(rec.get('from_category')))}→{escape(str(rec.get('to_category')))} "
                    f"— «{escape(str(rec.get('quote'))[:120])}»"
                )
        else:
            mid.append(f"[bold {_ORANGE}]Аудитор:[/] документ не найден / не извлекался")
        kyc = extracts.get("kyc_selected")
        mid.append("")
        if kyc:
            parties = kyc.get("counterparties") or []
            threshold = kyc.get("related_party_threshold_pct")
            mid.append(f"[bold {_ORANGE}]KYC:[/] порог {threshold}% · контрагентов {len(parties)}")
            for party in parties:
                over = party.get("voting_rights_pct", 0) >= (threshold or 0)
                mark = f"[bold {_ORANGE}]≥[/]" if over else " "
                mid.append(f"  {mark} {escape(str(party.get('counterparty_name')))} — {party.get('voting_rights_pct')}%")
        else:
            mid.append(f"[bold {_ORANGE}]KYC:[/] нет")
        if extracts.get("group_capex"):
            capex = extracts["group_capex"]
            mid.append("")
            mid.append(
                f"[bold {_ORANGE}]Group capex:[/] {_fmt_num(capex.get('value'))} из {escape(str(capex.get('doc_id')))} — "
                f"{escape(str(capex.get('derivation') or ''))}"
            )
        if cat:
            mid.append("")
            mid.append(
                f"[bold {_ORANGE}]Категоризация:[/] {cat.get('n_labeled')}/{cat.get('n_rows')} строк "
                f"· по категориям: {escape(json.dumps(cat.get('by_category_count') or {}, ensure_ascii=False))}"
            )
        self.query_one("#overview-mid", Static).update("\n".join(mid))

        flags = list(case.get("flags") or [])
        flags_title = (
            f"[bold {_ORANGE}]Флаги ({len(flags)}):[/]  [dim](Enter — открыть связанный файл)[/]"
            if flags
            else f"[bold {_ORANGE}]Флаги:[/] нет"
        )
        self.query_one("#overview-flags-title", Static).update(flags_title)
        flags_table = self.query_one("#overview-flags", DataTable)
        flags_table.clear(columns=True)
        flags_table.add_columns("Флаг", "Файл", "Цитата")
        for i, flag in enumerate(flags):
            doc_ids, quote = _enrich_flag(str(flag), extracts, documents)
            filenames = []
            for did in doc_ids:
                doc = _doc_by_id(documents, did)
                filenames.append(_doc_filename(doc or {"doc_id": did}))
            file_col = ", ".join(filenames) if filenames else "—"
            flags_table.add_row(
                escape(str(flag)),
                escape(file_col),
                escape(quote or "—"),
                key=f"fl-{i}:{doc_ids[0] if doc_ids else ''}",
            )

    # ---- docs

    def _render_docs(self) -> None:
        table = self.query_one("#docs", DataTable)
        table.add_columns("doc_id", "Тип", "Канал", "Стр.", "Симв.", "Скан", "Файл")
        docs = self._data().dev_case(self.sid).get("documents") or []
        for i, doc in enumerate(docs):
            table.add_row(
                escape(str(doc.get("doc_id"))),
                escape(str(doc.get("type") or "—")),
                escape(str(doc.get("link_channel") or "—")),
                str(doc.get("n_pages")),
                str(doc.get("chars")),
                "да" if doc.get("is_scan") else "",
                escape(Path(str(doc.get("path"))).name),
                key=str(i),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table_id = event.data_table.id
        if table_id == "docs":
            docs = self._data().dev_case(self.sid).get("documents") or []
            idx = int(str(event.row_key.value))
            if 0 <= idx < len(docs):
                self.app.push_screen(DocViewScreen(docs[idx]))
            return
        if table_id in ("overview-editions", "overview-flags"):
            key = str(event.row_key.value)
            # keys: "ed-0:docid" / "fl-3:docid"
            doc_id = key.split(":", 1)[1] if ":" in key else ""
            self._open_doc_by_id(doc_id or None)
            return

    # ---- covenants

    def _render_covenants(self) -> None:
        data = self._data()
        case = data.case(self.sid)
        cells = case.get("cells") or {}
        gt_cells = data.gt_cells(self.sid)
        submitted = data.submitted_cells(self.sid)
        loan = case.get("loan") or {}
        lines: list[str] = []
        missing = loan.get("clauses_missing_from_extract") or []
        if missing:
            lines.append(f"[bold {_ORANGE}]Не найдены в договоре:[/] {', '.join(map(escape, missing))}")
            lines.append("")
        for clause in sorted(cells):
            cell = cells[clause]
            spec = cell.get("spec") or {}
            pred = submitted.get(clause) or cell.get("pred") or {}
            key = gt_cells.get(clause)
            lines.append(f"[bold underline {_ORANGE}]Пункт {escape(clause)}[/] — {escape(str(cell.get('outcome')))}")
            if cell.get("stub_reason"):
                lines.append(f"  [bold {_ORANGE}]stub:[/] {escape(str(cell.get('stub_reason')))}")
            if spec:
                lines.append(
                    f"  [bold {_ORANGE}]Спека:[/] {escape(str(spec.get('covenant_type')))} "
                    f"{escape(str(spec.get('comparator')))} {_fmt_num(spec.get('threshold'))} "
                    f"· период {escape(str(spec.get('period_start')))}—{escape(str(spec.get('period_end')))}"
                )
                extra = {
                    k: spec.get(k)
                    for k in ("filter_category", "numerator_category", "denominator_category")
                    if spec.get(k)
                }
                if extra:
                    lines.append(f"  [bold {_ORANGE}]Категории:[/] {escape(json.dumps(extra, ensure_ascii=False))}")
            inputs = cell.get("inputs") or {}
            if inputs:
                lines.append(f"  [bold {_ORANGE}]Расчёт:[/] {escape(json.dumps(inputs, ensure_ascii=False))}")
            pred_line = (
                f"  [bold {_ORANGE}]Ответ:[/] {_status_markup(pred.get('status'))} · actual {_fmt_num(pred.get('actual'))} "
                f"· evidence {escape(str(pred.get('evidence_txn_id')))}"
            )
            lines.append(pred_line)
            if key is not None:
                score, err = cell_score(pred, key)
                status_ok = pred.get("status") == key.get("status")
                mark = f"[{_GREEN}]✓[/{_GREEN}]" if status_ok else f"[bold {_ORANGE}]✗[/]"
                err_str = "—" if err is None else f"{err * 100:.1f}%"
                lines.append(
                    f"  [bold {_ORANGE}]GT:[/] {mark} {_status_markup(key.get('status'))} · actual {_fmt_num(key.get('actual'))} "
                    f"(err {err_str}) · evidence {escape(str(key.get('evidence_txn_id')))} · [bold]score {score:.2f}[/]"
                )
            evidence = cell.get("evidence") or {}
            if evidence.get("flippers") or evidence.get("doc_named_candidates") or evidence.get("note"):
                lines.append(
                    f"  [bold {_ORANGE}]Флип-тест:[/] flippers={escape(str(evidence.get('flippers')))} "
                    f"doc-named={escape(str(evidence.get('doc_named_candidates')))} — {escape(str(evidence.get('note') or ''))}"
                )
            cell_flags = cell.get("flags") or []
            lines.extend(f"  [{_YELLOW}]![/{_YELLOW}] {escape(str(flag))}" for flag in cell_flags)
            if spec.get("quote"):
                lines.append(f"  [dim]«{escape(str(spec.get('quote'))[:400])}»[/]")
            lines.append("")
        self.query_one("#covenants", Static).update("\n".join(lines) or "нет данных")

    # ---- transactions

    def _render_txns(self) -> None:
        data = self._data()
        table = self.query_one("#txns", DataTable)
        table.clear(columns=True)
        table.add_columns("txn_id", "Дата", "Контрагент", "Описание", "Сумма", "Вал.", "USD", "Категория", "Пункты", "Реклас")
        txns = data.dev_case(self.sid).get("transactions") or []
        clause_map = data.txn_clause_map(self.sid)
        category_filter = _CATEGORY_CYCLE[self._category_idx]
        search = self._search.lower()
        shown = 0
        for txn in txns:
            category = txn.get("category")
            if category_filter == "uncategorized":
                if category:
                    continue
            elif category_filter is not None and category != category_filter:
                continue
            txn_id = str(txn.get("txn_id"))
            clauses = ",".join(sorted(clause_map.get(txn_id, ())))
            if self._contributing_only and not clauses:
                continue
            if search and search not in " ".join(
                str(txn.get(field, "")) for field in ("txn_id", "counterparty", "description")
            ).lower():
                continue
            amount = txn.get("amount") or 0.0
            table.add_row(
                escape(txn_id),
                escape(str(txn.get("date"))),
                escape(str(txn.get("counterparty"))[:32]),
                escape(str(txn.get("description"))[:48]),
                f"[{_ORANGE if amount < 0 else _GREEN}]{amount:+,.2f}[/]",
                escape(str(txn.get("currency"))),
                f"{(txn.get('amount_usd') or 0.0):+,.2f}",
                escape(str(category or "[dim]—[/]")),
                escape(clauses),
                f"[bold {_ORANGE}]excl[/]" if txn.get("excluded_by_reclass") else "",
                key=txn_id,
            )
            shown += 1
        filter_line = (
            f"показано {shown}/{len(txns)} · категория: [bold]{category_filter or 'все'}[/] (c) "
            f"· участвующие: [bold]{'да' if self._contributing_only else 'нет'}[/] (e)"
        )
        self.query_one("#txn-filter-line", Static).update(filter_line)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "txn-search":
            self._search = event.value
            self._render_txns()

    def action_cycle_category(self) -> None:
        self._category_idx = (self._category_idx + 1) % len(_CATEGORY_CYCLE)
        self._render_txns()

    def action_toggle_evidence(self) -> None:
        self._contributing_only = not self._contributing_only
        self._render_txns()


class DocViewScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Назад", show=True),
        Binding("q", "app.quit", "Выход", show=True),
    ]

    def __init__(self, doc: dict) -> None:
        super().__init__()
        self.doc = doc

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        header = (
            f"[bold {_ORANGE}]{escape(str(self.doc.get('doc_id')))}[/] · {escape(str(self.doc.get('type') or '—'))} · "
            f"{self.doc.get('n_pages')} стр. · {self.doc.get('chars')} симв. · "
            f"канал {escape(str(self.doc.get('link_channel') or '—'))} · {escape(str(self.doc.get('path')))}"
        )
        yield Static(header, id="doc-header")
        yield VerticalScroll(Static(escape(str(self.doc.get("text") or "(пусто)")), id="doc-text"))
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Документ {self.doc.get('doc_id')}"


class UnlinkedDocsScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Назад", show=True),
        Binding("q", "app.quit", "Выход", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield DataTable(id="unlinked", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        app: DevTuiApp = self.app  # type: ignore[assignment]
        assert app.data is not None
        docs = app.data.dump.get("unlinked_documents") or []
        self.title = f"Непривязанные документы ({len(docs)})"
        table = self.query_one("#unlinked", DataTable)
        table.add_columns("doc_id", "Тип", "Якорь", "Имя компании", "Причина", "Превью")
        for doc in docs:
            table.add_row(
                escape(str(doc.get("doc_id"))),
                escape(str(doc.get("type") or "—")),
                escape(str(doc.get("account_anchor") or "—")),
                escape(str(doc.get("company_name") or "—")),
                escape(str(doc.get("unlink_reason") or "—")),
                escape(" ".join(str(doc.get("preview") or "").split())[:80]),
            )
        table.focus()


# --------------------------------------------------------------------------- app


class DevTuiApp(App):
    TITLE = "AI Covenant Auditor"
    CSS = f"""
    Screen {{
        background: {_BG};
    }}
    Header {{
        background: {_BG};
        color: {_ORANGE};
        text-style: bold;
        dock: top;
        height: 1;
    }}
    Header .header--title {{
        color: {_ORANGE};
        text-style: bold;
    }}
    Footer {{
        background: #252525;
        color: #E8E8E8;
    }}
    Footer > .footer--key {{
        background: {_ORANGE};
        color: {_BG};
    }}
    DataTable {{
        background: {_BG};
        height: 1fr;
    }}
    DataTable > .datatable--header {{
        color: {_ORANGE};
        text-style: bold;
        background: #252525;
    }}
    DataTable > .datatable--cursor {{
        background: #3A2A1A;
        color: #FFFFFF;
    }}
    #overview-scroll {{
        height: 1fr;
        padding: 0 1;
    }}
    #overview-header,
    #overview-mid,
    #overview-editions-title,
    #overview-flags-title {{
        height: auto;
        margin: 0 0 1 0;
    }}
    #overview-editions,
    #overview-flags {{
        height: auto;
        max-height: 14;
        margin: 0 0 1 0;
    }}
    #docs,
    #txns,
    #unlinked {{
        height: 1fr;
    }}
    #runlog {{
        border: solid {_ORANGE};
        background: {_BG};
        height: 1fr;
        margin: 0 1 1 1;
        padding: 0 1;
    }}
    #runlog:focus {{
        border: solid {_ORANGE};
    }}
    #scenario-list-body {{
        height: 1fr;
        padding: 0 1;
    }}
    .alerts-panel {{
        height: auto;
        max-height: 10;
        border: solid {_ORANGE};
        border-title-color: {_ORANGE};
        border-title-style: bold;
        padding: 0 1;
        margin: 0 0 1 0;
        background: #252525;
        overflow-y: auto;
    }}
    .alerts-panel.-empty {{
        display: none;
    }}
    #doc-header {{
        padding: 0 1;
        background: #252525;
        color: {_ORANGE};
        border-bottom: solid {_BORDER};
    }}
    #doc-text {{
        padding: 0 1;
    }}
    #txn-search {{
        margin: 0 1;
        border: solid {_BORDER};
    }}
    #txn-search:focus {{
        border: solid {_ORANGE};
    }}
    #txn-filter-line {{
        padding: 0 1;
        color: #888888;
    }}
    TabbedContent {{
        height: 1fr;
    }}
    Tabs {{
        background: {_BG};
    }}
    Tab {{
        color: #AAAAAA;
    }}
    Tab.--highlight {{
        color: {_ORANGE};
        text-style: bold;
    }}
    Underline > .underline--bar {{
        background: {_ORANGE};
        color: {_ORANGE};
    }}
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.data: DevData | None = None

    def on_mount(self) -> None:
        self.register_theme(AUDITOR_THEME)
        self.theme = "auditor"
        dump_path = self.args.logs / "dev_dump.json"
        if self.args.browse:
            if self.reload_data():
                self.push_screen(ScenarioListScreen())
            else:
                self.exit(message=f"{dump_path} не найден — запусти без --browse, чтобы прогнать агента.")
        else:
            self.push_screen(RunScreen())

    def reload_data(self) -> bool:
        dump_path = self.args.logs / "dev_dump.json"
        if not dump_path.exists():
            self.notify(f"{dump_path} отсутствует", severity="error")
            return False
        self.data = DevData.load(
            self.args.dataset,
            self.args.logs,
            self.args.out,
            confidence_path=self.args.confidence,
        )
        return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dev TUI: run agent --dev and browse per-scenario debug data")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("submission.json"))
    p.add_argument("--logs", type=Path, default=Path("logs"))
    p.add_argument(
        "--confidence",
        type=Path,
        default=None,
        help="path to confidence_report.json (default: <logs>/confidence_report.json)",
    )
    p.add_argument("--only", default=None, help="process a single scenario_id")
    p.add_argument("--no-cache", action="store_true", help="pass --no-cache to agent.run")
    p.add_argument("--browse", action="store_true", help="skip the agent run; open the existing logs/dev_dump.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    DevTuiApp(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
