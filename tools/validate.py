"""Contract validator for submission.json (TESTING-PLAN.md §2).

Runs with no ground truth. This is the only check that can run on battle day
right before submitting. Zero violations is a hard requirement.

Usage:
    python -m tools.validate --submission runs/xxx/submission.json --dataset <DIR>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOP_LEVEL_KEYS = {"team", "contact_email", "model", "answers"}
CELL_KEYS = {"status", "actual", "evidence_txn_id"}
VALID_STATUSES = {"COMPLIANT", "BREACH"}

_TXN_RE = re.compile(r"^TXN-(?P<scenario>.+)-(?P<seq>\d+)$")


@dataclass
class Violation:
    address: str
    message: str

    def __str__(self) -> str:
        return f"{self.address}: {self.message}"


@dataclass
class Warning_:
    address: str
    message: str

    def __str__(self) -> str:
        return f"{self.address}: {self.message}"


@dataclass
class ValidationResult:
    violations: list[Violation] = field(default_factory=list)
    warnings: list[Warning_] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ledger_index(ledger_path: Path) -> dict[str, set[str]]:
    """scenario_id -> set of txn_id belonging to it, derived from the txn_id prefix.

    Never hardcode scenario ids: they are read from the ledger at runtime.
    """
    index: dict[str, set[str]] = {}
    with open(ledger_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txn_id = row.get("txn_id", "")
            m = _TXN_RE.match(txn_id)
            if not m:
                continue
            scenario = m.group("scenario")
            index.setdefault(scenario, set()).add(txn_id)
    return index


def validate(submission: dict, template: dict, ledger_index: dict[str, set[str]]) -> ValidationResult:
    result = ValidationResult()

    if not isinstance(submission, dict):
        result.violations.append(Violation("$", "submission root is not a JSON object"))
        return result

    top_keys = set(submission.keys())
    missing = TOP_LEVEL_KEYS - top_keys
    extra = top_keys - TOP_LEVEL_KEYS
    for k in sorted(missing):
        result.violations.append(Violation("$", f"missing required top-level key '{k}'"))
    for k in sorted(extra):
        result.violations.append(Violation("$", f"unexpected top-level key '{k}'"))

    team = submission.get("team")
    if "team" in submission and (not isinstance(team, str) or not team.strip()):
        result.violations.append(Violation("$.team", "team must be a non-empty string"))
    contact = submission.get("contact_email")
    if "contact_email" in submission and (not isinstance(contact, str) or not contact.strip()):
        result.violations.append(Violation("$.contact_email", "contact_email must be a non-empty string"))

    answers = submission.get("answers")
    template_answers = template.get("answers", {})

    if not isinstance(answers, dict):
        result.violations.append(Violation("$.answers", "must be an object"))
        return result

    expected_scenarios = set(template_answers.keys())
    got_scenarios = set(answers.keys())
    for sid in sorted(expected_scenarios - got_scenarios):
        result.violations.append(Violation(f"$.answers.{sid}", "scenario missing from submission"))
    for sid in sorted(got_scenarios - expected_scenarios):
        result.violations.append(Violation(f"$.answers.{sid}", "scenario not present in submission_template.json"))

    breach_count = 0
    compliant_count = 0

    for sid in sorted(expected_scenarios & got_scenarios):
        expected_clauses = set(template_answers[sid].keys())
        got_clauses = set(answers[sid].keys())
        for clause in sorted(expected_clauses - got_clauses):
            result.violations.append(Violation(f"$.answers.{sid}.{clause}", "clause missing from submission"))
        for clause in sorted(got_clauses - expected_clauses):
            result.violations.append(
                Violation(f"$.answers.{sid}.{clause}", "clause not present in submission_template.json")
            )

        for clause in sorted(expected_clauses & got_clauses):
            addr = f"$.answers.{sid}.{clause}"
            cell = answers[sid][clause]
            if not isinstance(cell, dict):
                result.violations.append(Violation(addr, "cell is not an object"))
                continue

            cell_keys = set(cell.keys())
            for k in sorted(CELL_KEYS - cell_keys):
                result.violations.append(Violation(addr, f"missing key '{k}'"))
            for k in sorted(cell_keys - CELL_KEYS):
                result.violations.append(Violation(addr, f"unexpected key '{k}'"))

            status = cell.get("status")
            if status is None:
                result.violations.append(Violation(addr, "status is empty (null)"))
            elif not isinstance(status, str) or status not in VALID_STATUSES:
                result.violations.append(Violation(addr, f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}"))
            else:
                if status == "BREACH":
                    breach_count += 1
                else:
                    compliant_count += 1

            actual = cell.get("actual")
            if actual is None:
                result.violations.append(Violation(addr, "actual is empty (null)"))
            elif isinstance(actual, bool) or not isinstance(actual, (int, float)):
                result.violations.append(Violation(addr, f"actual must be a number, got {type(actual).__name__}"))
            elif not math.isfinite(float(actual)):
                result.violations.append(Violation(addr, f"actual must be finite, got {actual}"))
            elif actual <= 0:
                result.violations.append(Violation(addr, f"actual must be strictly > 0, got {actual}"))
            elif round(float(actual), 2) != float(actual):
                result.violations.append(Violation(addr, f"actual must be rounded to 2 decimals, got {actual}"))

            evidence = cell.get("evidence_txn_id")
            if evidence is not None:
                if not isinstance(evidence, str):
                    result.violations.append(Violation(addr, f"evidence_txn_id must be null or a string, got {type(evidence).__name__}"))
                else:
                    scenario_txns = ledger_index.get(sid, set())
                    if evidence not in scenario_txns:
                        result.violations.append(
                            Violation(addr, f"evidence_txn_id '{evidence}' does not exist in the ledger for scenario '{sid}'")
                        )
                    if isinstance(status, str) and status == "COMPLIANT":
                        result.warnings.append(Warning_(addr, "COMPLIANT with non-null evidence_txn_id (unusual, not forbidden)"))

    total_status_cells = breach_count + compliant_count
    if total_status_cells > 0:
        breach_share = breach_count / total_status_cells
        if not (0.30 <= breach_share <= 0.70):
            result.warnings.append(
                Warning_("$", f"BREACH share {breach_share:.2%} is outside the 30-70% sanity range ({breach_count}/{total_status_cells})")
            )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contract validator for submission.json")
    p.add_argument("--submission", required=True, type=Path, help="Path to submission.json to validate")
    p.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Dataset dir (ledger/template/documents discovered structurally)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from agent.dataset import discover_dataset

    paths = discover_dataset(args.dataset)
    submission = load_json(args.submission)
    template = load_json(paths.template)
    ledger_index = load_ledger_index(paths.ledger)

    result = validate(submission, template, ledger_index)

    for w in result.warnings:
        print(f"WARN  {w}")
    for v in result.violations:
        print(f"FAIL  {v}")

    print(f"\n{len(result.violations)} violation(s), {len(result.warnings)} warning(s)")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
