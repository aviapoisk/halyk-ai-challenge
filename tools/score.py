"""Scorer for submission.json against ground_truth.json (TESTING-PLAN.md §3).

ground_truth.json shape:  scenarios.<scenario_id>.covenants.<clause>.{status,actual,evidence_txn_id}
submission.json shape:    answers.<scenario_id>.<clause>.{status,actual,evidence_txn_id}

Evidence scoring when the GT evidence_txn_id is null — confirmed verbatim by
CASE.ru.md §3/§4: «присланное значение не учитывается — в том числе null», а 0.20
убывают вместе с actual по той же шкале. Это режим "lenient" — он официальный.

  lenient (официальный) — присланный evidence игнорируется: 0.50 * acc
  strict (пессимистичная диагностика, НЕ правила) — лишний evidence стоит 0.20-долю:
    0.30*acc + (0.20*acc if pred is None else 0)

Usage:
    python -m tools.score --submission runs/xxx/submission.json --ground-truth <DIR>/ground_truth.json
    python -m tools.score ... --no-tui   # plain text (also AGENT_NO_TUI=1 / non-TTY)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

EvidenceMode = Literal["lenient", "strict"]

# Same palette as agent/progress.py
_ORANGE = "#D96B27"
_GREEN = "#4EBF71"
_YELLOW = "#E5C07B"
_RED = "#E06C75"


@dataclass
class CellResult:
    scenario: str
    clause: str
    key_status: str | None
    pred_status: str | None
    key_actual: float | None
    pred_actual: float | None
    key_evidence: str | None
    pred_evidence: str | None
    error: float | None  # relative error on actual, None if status wrong or actual missing
    score_lenient: float
    score_strict: float


def normalize_gt(gt: dict) -> dict[str, dict[str, dict]]:
    """scenarios.<id>.covenants.<clause> -> {scenario: {clause: cell}}"""
    out: dict[str, dict[str, dict]] = {}
    for sid, body in gt.get("scenarios", {}).items():
        out[sid] = dict(body.get("covenants", {}))
    return out


def normalize_submission(sub: dict) -> dict[str, dict[str, dict]]:
    """answers.<id>.<clause> -> {scenario: {clause: cell}}"""
    out: dict[str, dict[str, dict]] = {}
    for sid, body in sub.get("answers", {}).items():
        out[sid] = dict(body)
    return out


def cell_score(pred: dict | None, key: dict, evidence_mode: EvidenceMode = "lenient") -> tuple[float, float | None]:
    """Returns (score, relative_error_on_actual_or_None)."""
    if pred is None:
        return 0.0, None

    pred_status = pred.get("status")
    key_status = key["status"]
    if pred_status != key_status:
        return 0.0, None

    s = 0.50

    k = key["actual"]
    pred_actual = pred.get("actual")
    if pred_actual is None or not isinstance(pred_actual, (int, float)) or isinstance(pred_actual, bool):
        # CASE.ru.md §4: отсутствующий/нечисловой actual теряет 0.30 (и 0.20 только при
        # null-ключе); при не-null ключе верный evidence_txn_id всё равно даёт 0.20.
        e = None
        acc = 0.0
    else:
        e = abs(float(pred_actual) - float(k)) / abs(float(k))
        acc = max(0.0, 1.0 - e / 0.05)

    key_evidence = key.get("evidence_txn_id")
    pred_evidence = pred.get("evidence_txn_id")

    if key_evidence is not None:
        s += 0.30 * acc
        s += 0.20 if pred_evidence == key_evidence else 0.0
    else:
        if evidence_mode == "lenient":
            s += 0.50 * acc
        else:
            ok = pred_evidence is None
            s += 0.30 * acc + (0.20 * acc if ok else 0.0)

    return s, e


def _scenario_order(
    gt_by_scenario: dict[str, dict[str, dict]],
    sub_by_scenario: dict[str, dict[str, dict]],
) -> list[str]:
    """Prefer submission/template key order; append any GT-only scenarios after."""
    ordered = list(sub_by_scenario.keys())
    seen = set(ordered)
    for sid in gt_by_scenario:
        if sid not in seen:
            ordered.append(sid)
            seen.add(sid)
    return ordered


def _clause_order(
    gt_clauses: dict[str, dict],
    sub_clauses: dict[str, dict],
) -> list[str]:
    """Prefer submission clause order; append any GT-only clauses after."""
    ordered = [c for c in sub_clauses if c in gt_clauses]
    seen = set(ordered)
    for clause in gt_clauses:
        if clause not in seen:
            ordered.append(clause)
            seen.add(clause)
    return ordered


def score_all(
    gt_by_scenario: dict[str, dict[str, dict]],
    sub_by_scenario: dict[str, dict[str, dict]],
) -> list[CellResult]:
    results: list[CellResult] = []
    for sid in _scenario_order(gt_by_scenario, sub_by_scenario):
        clauses = gt_by_scenario.get(sid, {})
        if not clauses:
            continue
        sub_clauses = sub_by_scenario.get(sid, {})
        for clause in _clause_order(clauses, sub_clauses):
            key = clauses[clause]
            pred = sub_clauses.get(clause)
            score_l, e_l = cell_score(pred, key, "lenient")
            score_s, e_s = cell_score(pred, key, "strict")
            results.append(
                CellResult(
                    scenario=sid,
                    clause=clause,
                    key_status=key.get("status"),
                    pred_status=pred.get("status") if pred else None,
                    key_actual=key.get("actual"),
                    pred_actual=pred.get("actual") if pred else None,
                    key_evidence=key.get("evidence_txn_id"),
                    pred_evidence=pred.get("evidence_txn_id") if pred else None,
                    error=e_l,
                    score_lenient=score_l,
                    score_strict=score_s,
                )
            )
    return results


def loss_attribution(results: list[CellResult]) -> dict[str, float]:
    """How many points leaked on status / actual / evidence separately (lenient)."""
    status_loss = 0.0
    actual_loss = 0.0
    evidence_loss = 0.0
    for r in results:
        if r.key_status != r.pred_status:
            status_loss += 0.50
            actual_loss += 0.30
            evidence_loss += 0.20
            continue
        # Status correct — decompose remaining shortfall from score_lenient.
        got = r.score_lenient
        # Ideal with correct status is 1.0; shortfall split by formula weights.
        shortfall = max(0.0, 1.0 - got)
        # Prefer reporting field-level gaps when computable.
        if r.key_actual is not None and isinstance(r.pred_actual, (int, float)) and not isinstance(r.pred_actual, bool):
            e = abs(float(r.pred_actual) - float(r.key_actual)) / max(abs(float(r.key_actual)), 1e-12)
            acc = max(0.0, 1.0 - e / 0.05)
            if r.key_evidence is not None:
                actual_loss += 0.30 * (1.0 - acc)
                evidence_loss += 0.0 if r.pred_evidence == r.key_evidence else 0.20
            else:
                # lenient: actual+evidence share 0.50 * acc
                actual_loss += 0.50 * (1.0 - acc)
        else:
            actual_loss += shortfall
    return {"status": status_loss, "actual": actual_loss, "evidence": evidence_loss}


def worst_cells(results: list[CellResult], n: int = 10) -> list[CellResult]:
    return sorted(results, key=lambda r: r.score_lenient)[:n]


def summarize(results: list[CellResult]) -> dict:
    n = len(results)
    total_lenient = sum(r.score_lenient for r in results)
    total_strict = sum(r.score_strict for r in results)

    status_correct = sum(1 for r in results if r.key_status == r.pred_status)

    confusion = {"COMPLIANT->COMPLIANT": 0, "COMPLIANT->BREACH": 0, "BREACH->COMPLIANT": 0, "BREACH->BREACH": 0}
    for r in results:
        if r.key_status and r.pred_status:
            confusion[f"{r.key_status}->{r.pred_status}"] = confusion.get(f"{r.key_status}->{r.pred_status}", 0) + 1
        elif r.key_status:
            confusion[f"{r.key_status}->MISSING"] = confusion.get(f"{r.key_status}->MISSING", 0) + 1

    errs = [r.error for r in results if r.error is not None]
    within_2_5 = sum(1 for e in errs if e <= 0.025)
    within_5 = sum(1 for e in errs if e <= 0.05)

    evidence_cells = [r for r in results if r.key_evidence is not None]
    ev_hit = sum(1 for r in evidence_cells if r.pred_evidence == r.key_evidence)
    ev_false = sum(1 for r in evidence_cells if r.pred_evidence is not None and r.pred_evidence != r.key_evidence)
    ev_missed = sum(1 for r in evidence_cells if r.pred_evidence is None)

    loss = loss_attribution(results)
    worst = [
        {"scenario": r.scenario, "clause": r.clause, "score_lenient": r.score_lenient}
        for r in worst_cells(results, 10)
    ]

    return {
        "n_cells": n,
        "score_lenient": total_lenient,
        "score_strict": total_strict,
        "score_lenient_avg": total_lenient / n if n else 0.0,
        "score_strict_avg": total_strict / n if n else 0.0,
        "status_accuracy": status_correct / n if n else 0.0,
        "confusion": confusion,
        "n_with_error": len(errs),
        "within_2_5pct": within_2_5,
        "within_5pct": within_5,
        "evidence_total": len(evidence_cells),
        "evidence_hit": ev_hit,
        "evidence_false": ev_false,
        "evidence_missed": ev_missed,
        "loss_attribution": loss,
        "worst_cells": worst,
    }


def _fmt_num(v: float | None) -> str:
    if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    x = float(v)
    if abs(x) >= 1_000_000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    return f"{x:.2f}"


def _fmt_err(error: float | None) -> str:
    if error is None:
        return "—"
    pct = error * 100
    if pct > 999:
        return ">999%"
    return f"{pct:.1f}%"


def _fmt_evid(v: str | None) -> str:
    return v or "—"


def _fmt_status(v: str | None) -> str:
    if not v:
        return "—"
    return v


def _status_style(ok: bool) -> str:
    return _GREEN if ok else _RED


def _score_style(score: float) -> str:
    if score >= 0.999:
        return _GREEN
    if score <= 0.001:
        return _RED
    return _YELLOW


def _use_tui(*, force_quiet: bool) -> bool:
    env_quiet = os.environ.get("AGENT_NO_TUI", "").strip() in {"1", "true", "yes"}
    return not (force_quiet or env_quiet or not sys.stdout.isatty())


def print_report_plain(results: list[CellResult], summary: dict) -> None:
    print("=== Agent vs Public (per-cell) ===")
    header = (
        f"{'scenario':10}{'clause':8}"
        f"{'status Public':14}{'status Agent':14}"
        f"{'actual Public':>14}{'actual Agent':>14}{'err%':>8}"
        f"{'evid Public':>16}{'evid Agent':>16}"
        f"{'L':>6}{'S':>6}"
    )
    print(header)
    # Preserve submission key order from score_all (not lexicographic P1/P10/P2…).
    for r in results:
        print(
            f"{r.scenario:10}{r.clause:8}"
            f"{_fmt_status(r.key_status):14}{_fmt_status(r.pred_status):14}"
            f"{_fmt_num(r.key_actual):>14}{_fmt_num(r.pred_actual):>14}{_fmt_err(r.error):>8}"
            f"{_fmt_evid(r.key_evidence):>16}{_fmt_evid(r.pred_evidence):>16}"
            f"{r.score_lenient:>6.2f}{r.score_strict:>6.2f}"
        )

    print("\n=== Summary ===")
    print(f"cells: {summary['n_cells']}")
    print(f"score (lenient): {summary['score_lenient']:.3f}  (avg {summary['score_lenient_avg']:.3f})")
    print(f"score (strict):  {summary['score_strict']:.3f}  (avg {summary['score_strict_avg']:.3f})")
    gap = summary["score_lenient_avg"] - summary["score_strict_avg"]
    print(
        f"lenient-strict gap (avg): {gap:.4f}"
        + ("  <-- review evidence_txn_id-on-null-key cells" if abs(gap) > 0.01 else "")
    )
    print(f"status accuracy: {summary['status_accuracy']:.3f}")
    print(f"confusion: {summary['confusion']}")
    loss = summary.get("loss_attribution") or {}
    if loss:
        print(
            f"loss attribution: status={loss.get('status', 0):.2f} "
            f"actual={loss.get('actual', 0):.2f} evidence={loss.get('evidence', 0):.2f}"
        )
    worst = summary.get("worst_cells") or []
    if worst:
        print("worst cells:")
        for w in worst:
            print(f"  {w['scenario']}/{w['clause']}: {w['score_lenient']:.2f}")
    if summary["n_with_error"]:
        print(f"actual within 2.5%: {summary['within_2_5pct']}/{summary['n_with_error']}")
        print(f"actual within 5%:   {summary['within_5pct']}/{summary['n_with_error']}")
    print(
        f"evidence (non-null key cells): total={summary['evidence_total']} "
        f"hit={summary['evidence_hit']} false={summary['evidence_false']} missed={summary['evidence_missed']}"
    )


def _build_diff_table(results: list[CellResult]) -> Table:
    """Compact Agent-vs-Public table; fits ~100 cols without clipping status."""
    table = Table(
        title="[bold white]Per-cell · Public vs Agent[/bold white]",
        title_style="bold " + _ORANGE,
        expand=False,
        box=None,
        show_header=True,
        header_style="bold " + _ORANGE,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Scen", style="bold white", no_wrap=True)
    table.add_column("Cl", style="dim", no_wrap=True)
    table.add_column("status Public", justify="left", no_wrap=True)
    table.add_column("status Agent", justify="left", no_wrap=True)
    table.add_column("actual Public", justify="right", no_wrap=True)
    table.add_column("actual Agent", justify="right", no_wrap=True)
    table.add_column("err%", justify="right", no_wrap=True)
    table.add_column("evid Public", justify="left", no_wrap=True)
    table.add_column("evid Agent", justify="left", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)

    # Preserve submission key order from score_all (not lexicographic P1/P10/P2…).
    for r in results:
        status_ok = r.key_status == r.pred_status
        evid_ok = (r.key_evidence is None) or (r.pred_evidence == r.key_evidence)
        actual_ok = r.error is not None and r.error <= 0.05

        pub_status = Text(_fmt_status(r.key_status), style="bold white")
        agent_status = Text(
            _fmt_status(r.pred_status),
            style="bold " + _status_style(status_ok),
        )

        pub_actual = Text(_fmt_num(r.key_actual), style="white")
        agent_actual_style = _GREEN if actual_ok else (_YELLOW if status_ok else "dim")
        agent_actual = Text(_fmt_num(r.pred_actual), style=agent_actual_style)

        if r.error is None:
            err_text = Text("—", style="dim")
        elif r.error <= 0.025:
            err_text = Text(_fmt_err(r.error), style=_GREEN)
        elif r.error <= 0.05:
            err_text = Text(_fmt_err(r.error), style=_YELLOW)
        else:
            err_text = Text(_fmt_err(r.error), style=_RED)

        pub_evid = Text(_fmt_evid(r.key_evidence), style="dim" if not r.key_evidence else "white")
        if r.key_evidence is None:
            agent_evid_style = "dim"
        elif evid_ok:
            agent_evid_style = _GREEN
        elif r.pred_evidence is None:
            agent_evid_style = _RED
        else:
            agent_evid_style = _YELLOW
        agent_evid = Text(_fmt_evid(r.pred_evidence), style=agent_evid_style)

        score_text = Text(f"{r.score_lenient:.2f}", style="bold " + _score_style(r.score_lenient))

        table.add_row(
            r.scenario,
            r.clause,
            pub_status,
            agent_status,
            pub_actual,
            agent_actual,
            err_text,
            pub_evid,
            agent_evid,
            score_text,
        )
    return table



def _build_summary_panel(summary: dict) -> Panel:
    n = summary["n_cells"]
    avg = summary["score_lenient_avg"]
    conf = summary["confusion"]
    gap = summary["score_lenient_avg"] - summary["score_strict_avg"]

    cc = conf.get("COMPLIANT->COMPLIANT", 0)
    bc = conf.get("BREACH->COMPLIANT", 0)
    bb = conf.get("BREACH->BREACH", 0)
    cb = conf.get("COMPLIANT->BREACH", 0)
    status_acc = summary["status_accuracy"] * 100
    status_line = (
        f"[bold white]{status_acc:.1f}%[/bold white] status accuracy  "
        f"[dim]·[/dim]  "
        f"[{_GREEN}]C→C {cc}[/{_GREEN}]  "
        f"[{_RED}]B→C {bc}[/{_RED}]  "
        f"[{_GREEN}]B→B {bb}[/{_GREEN}]  "
        f"[{_YELLOW}]C→B {cb}[/{_YELLOW}]"
    )

    actual_line = "—"
    if summary["n_with_error"]:
        w25 = summary["within_2_5pct"]
        w5 = summary["within_5pct"]
        nerr = summary["n_with_error"]
        actual_line = (
            f"within 2.5%: [bold]{w25}/{nerr}[/bold]  "
            f"within 5%: [bold]{w5}/{nerr}[/bold]"
        )

    ev_hit = summary["evidence_hit"]
    ev_missed = summary["evidence_missed"]
    ev_false = summary["evidence_false"]
    ev_total = summary["evidence_total"]
    evid_line = (
        f"hit [{_GREEN}]{ev_hit}[/{_GREEN}]  "
        f"missed [{_RED}]{ev_missed}[/{_RED}]  "
        f"false [{_YELLOW}]{ev_false}[/{_YELLOW}]  "
        "[dim](" + str(ev_total) + " non-null GT cells)[/dim]"
    )

    gap_note = ""
    if abs(gap) > 0.01:
        gap_note = f"  [{_YELLOW}]← review evidence on null-key cells[/{_YELLOW}]"

    score_l = summary["score_lenient"]
    score_s = summary["score_strict"]
    nl = chr(10)
    body = Text.from_markup(
        f"[bold {_ORANGE}]SCORE[/bold {_ORANGE}]  "
        f"[bold white]{score_l:.3f}[/bold white] / {n}  "
        f"[dim](avg {avg:.3f})[/dim]  "
        f"[dim]strict {score_s:.3f}  gap {gap:.4f}[/dim]{gap_note}"
        + nl
        + nl
        + f"• Status:   {status_line}"
        + nl
        + f"• Actual:   {actual_line}"
        + nl
        + f"• Evidence: {evid_line}"
    )
    return Panel(body, title="[bold white]Summary[/bold white]", border_style="#444444")


def print_report_tui(results: list[CellResult], summary: dict, *, console: Console | None = None) -> None:
    if console is None:
        # Rich 15 only honors Console(width=…) when height is also set.
        try:
            term = shutil.get_terminal_size(fallback=(120, 40))
            cols = max(term.columns, 100)
            rows = max(term.lines, 40)
        except Exception:
            cols, rows = 120, 40
        console = Console(width=cols, height=rows)
    header = Table.grid(expand=True)
    header.add_column(justify="left", ratio=1)
    header.add_column(justify="right", ratio=1)
    header.add_row(
        f"[bold {_ORANGE}]AI Covenant Auditor[/bold {_ORANGE}] [dim]· score[/dim]",
        "[dim]Public = ground truth · Agent = submission[/dim]",
    )
    header.add_row(
        "[bold white]Agent vs Public[/bold white]",
        f"[{_GREEN}]✔ match[/{_GREEN}]  [{_RED}]✘ mismatch[/{_RED}]  "
        f"[{_YELLOW}]~ partial[/{_YELLOW}]",
    )

    console.print(Panel(header, style="on #1E1E1E", border_style="#444444"))
    console.print(_build_diff_table(results))
    console.print(_build_summary_panel(summary))


def print_report(results: list[CellResult], summary: dict, *, force_quiet: bool = False) -> None:
    if _use_tui(force_quiet=force_quiet):
        print_report_tui(results, summary)
    else:
        print_report_plain(results, summary)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score submission.json against ground_truth.json")
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to dump the summary as JSON")
    p.add_argument(
        "--no-tui",
        action="store_true",
        help="Plain text output (also AGENT_NO_TUI=1 or non-TTY)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    with open(args.submission, encoding="utf-8") as f:
        submission = json.load(f)
    with open(args.ground_truth, encoding="utf-8") as f:
        gt = json.load(f)

    gt_by_scenario = normalize_gt(gt)
    sub_by_scenario = normalize_submission(submission)

    results = score_all(gt_by_scenario, sub_by_scenario)
    summary = summarize(results)
    print_report(results, summary, force_quiet=args.no_tui)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
