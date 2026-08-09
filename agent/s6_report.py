"""S6 — Confidence report + submission writer.

Sanitize cells to the contract, validate, then always write atomically.
Absence of a file is worse than a flawed cell; sanitize_fixes go to the
confidence report for triage.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from tools.validate import ValidationResult, load_ledger_index, validate

_VALID_STATUS = {"COMPLIANT", "BREACH"}


def sanitize_answers(submission: dict, template: dict, ledger_index: dict[str, set[str]]) -> list[str]:
    """Bring every cell to the contract. Returns fix messages for the confidence report."""
    fixes: list[str] = []
    answers = submission.setdefault("answers", {})
    for sid, clauses in template.get("answers", {}).items():
        cells = answers.setdefault(sid, {})
        for clause in clauses:
            cell = cells.get(clause)
            if not isinstance(cell, dict):
                cell = {}
                fixes.append(f"{sid}.{clause}: ячейка отсутствовала")
            status = str(cell.get("status", "")).upper()
            if status not in _VALID_STATUS:
                fixes.append(f"{sid}.{clause}: status={cell.get('status')!r} → COMPLIANT")
                status = "COMPLIANT"
            actual = cell.get("actual")
            try:
                actual_f = abs(float(actual))
                if not math.isfinite(actual_f) or actual_f == 0.0:
                    raise ValueError
            except (TypeError, ValueError):
                fixes.append(f"{sid}.{clause}: actual={cell.get('actual')!r} → 0.01")
                actual_f = 0.01
            ev = cell.get("evidence_txn_id")
            if ev is not None:
                if not isinstance(ev, str) or ev not in ledger_index.get(sid, set()):
                    fixes.append(f"{sid}.{clause}: evidence {ev!r} не принадлежит сценарию → null")
                    ev = None
            cells[clause] = {
                "status": status,
                "actual": round(actual_f, 2),
                "evidence_txn_id": ev,
            }
        for extra in set(cells) - set(clauses):
            del cells[extra]
            fixes.append(f"{sid}.{extra}: лишний пункт удалён")
    for extra in set(answers) - set(template.get("answers", {})):
        del answers[extra]
        fixes.append(f"{extra}: лишний сценарий удалён")
    return fixes


def write_submission_atomic(
    submission: dict, template: dict, ledger_path: Path, out_path: Path
) -> tuple[ValidationResult, list[str]]:
    """Sanitize, validate, always write. Returns (validation, sanitize_fixes)."""
    known = set((template.get("answers") or {}).keys())
    ledger_index = load_ledger_index(ledger_path, known)
    fixes = sanitize_answers(submission, template, ledger_index)
    result = validate(submission, template, ledger_index)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return result, fixes


def triage_duplicate_actuals(submission: dict, template: dict) -> list[str]:
    """Flag identical actual values shared by ≥2 clauses within one scenario."""
    flags: list[str] = []
    answers = submission.get("answers") or {}
    for sid, clauses in (template.get("answers") or {}).items():
        cells = answers.get(sid) or {}
        by_actual: dict[float, list[str]] = {}
        for clause in clauses:
            cell = cells.get(clause) if isinstance(cells.get(clause), dict) else None
            if not isinstance(cell, dict):
                continue
            try:
                actual = round(abs(float(cell.get("actual"))), 2)
            except (TypeError, ValueError):
                continue
            by_actual.setdefault(actual, []).append(str(clause))
        for actual, clause_list in sorted(by_actual.items(), key=lambda kv: kv[0]):
            if len(clause_list) >= 2:
                joined = ",".join(sorted(clause_list))
                flags.append(f"{sid}: duplicate_actual {actual} on {joined}")
    return flags


def triage_actual_vs_threshold(submission: dict, template: dict) -> list[str]:
    """C2 fallback: flag incomplete cells when threshold traces are unavailable."""
    flags: list[str] = []
    answers = submission.get("answers") or {}
    for sid, clauses in (template.get("answers") or {}).items():
        cells = answers.get(sid) or {}
        for clause in clauses:
            cell = cells.get(clause) if isinstance(cells.get(clause), dict) else None
            if not isinstance(cell, dict):
                flags.append(f"{sid}/{clause}: triage incomplete_cell")
                continue
            try:
                actual = abs(float(cell.get("actual")))
            except (TypeError, ValueError):
                flags.append(f"{sid}/{clause}: triage incomplete_actual")
                continue
            if actual <= 0:
                flags.append(f"{sid}/{clause}: triage actual_nonpositive")
    return flags


def write_confidence_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = report.get("submission")
    tmpl = report.get("template")
    case_traces = report.get("case_traces") or {}
    thr_flags = _triage_from_case_traces(case_traces)
    fallback: list[str] = []
    dup_flags: list[str] = []
    if isinstance(sub, dict) and isinstance(tmpl, dict):
        fallback = triage_actual_vs_threshold(sub, tmpl)
        dup_flags = triage_duplicate_actuals(sub, tmpl)
    combined = list(dict.fromkeys([*thr_flags, *fallback, *dup_flags]))
    if len(combined) > 20:
        preferred = [
            f
            for f in combined
            if any(k in f for k in ("extreme", "near_threshold", "incomplete", "duplicate_actual"))
        ]
        combined = preferred[:20]
        if len(preferred) > 20:
            combined.append("triage_truncated")
    report = {**report, "triage_flags": combined}
    # Drop bulky passthrough keys from the on-disk report.
    slim = {k: v for k, v in report.items() if k not in {"submission", "template", "case_traces"}}
    (out_dir / "confidence_report.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_run_config(config: dict, out_dir: Path) -> Path:
    """Write run CLI/env keys + git sha next to other log artifacts (no secrets)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _triage_from_case_traces(case_traces: dict) -> list[str]:
    """actual/threshold >5 or <0.2; within 10% of threshold."""
    out: list[str] = []
    if not isinstance(case_traces, dict):
        return out
    for sid, case in case_traces.items():
        if not isinstance(case, dict):
            continue
        cells = case.get("cells") or {}
        if not isinstance(cells, dict):
            continue
        for clause, trace in cells.items():
            if not isinstance(trace, dict):
                continue
            thr = trace.get("threshold")
            if thr is None:
                spec = trace.get("spec")
                if isinstance(spec, dict):
                    thr = spec.get("threshold")
            pred = trace.get("pred") if isinstance(trace.get("pred"), dict) else {}
            actual = pred.get("actual", trace.get("actual"))
            if thr is None or actual is None:
                continue
            try:
                t = abs(float(thr))
                a = abs(float(actual))
            except (TypeError, ValueError):
                continue
            if t <= 0:
                continue
            ratio = a / t
            key = f"{sid}/{clause}"
            if ratio > 5:
                out.append(f"{key}: triage extreme_high actual/threshold={ratio:.2f}")
            elif ratio < 0.2:
                out.append(f"{key}: triage extreme_low actual/threshold={ratio:.2f}")
            elif abs(a - t) / t <= 0.10:
                out.append(f"{key}: triage near_threshold actual={a} thr={t}")
    return out
