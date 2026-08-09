#!/usr/bin/env python3
"""Dev-only reconstruction helper for open-set GT formulas (PLAN-36-V4).

NOT imported by agent.run. Literals of the open set live here / in tests only —
never copy numeric constants into agent/.

Usage:
  uv run python -m tools.recon
  uv run python -m tools.recon --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Confirmed reconstructions from docs/PLAN-36-V4-2026-08-09.md appendix.
# P4/6.3 intentionally omitted (open-set threshold typo).
_RECON_CELLS: list[dict] = [
    {
        "scenario": "P1",
        "clause": "6.1",
        "note": "capex / (berth + lease); P1-0045 auditor-excluded (2026 services)",
        "expected_actual": 0.46,
        "expected_status": "BREACH",
        "components": {
            "num": 1_842_006.44,
            "den": 3_104_882.61 + 918_443.27,
        },
    },
    {
        "scenario": "P2",
        "clause": "6.1",
        "note": "(rev+reclass opex)/(opex incl Tien Shan); retainer not in opex",
        "expected_actual": 1.18,
        "expected_status": "BREACH",
        "components": {
            "num": 7_214_663.82 + 4_882_117.46,
            "den": 5_918_004.37 + 3_204_881.55 + 1_104_663.28,
        },
    },
    {
        "scenario": "P3",
        "clause": "6.1",
        "note": "FP / (rev − plant − EUR USD)",
        "expected_actual": 1.71,
        "expected_status": "BREACH",
        "components": {
            "num": 5_442_118.93,
            "den": 8_104_772.36 - 4_218_006.51 - 708_761.82,
        },
    },
    {
        "scenario": "P4",
        "clause": "6.1",
        "note": "Adj EBITDA/Rev with Ilek+Aral add-backs ≥ materiality",
        "expected_actual": 0.33,
        "expected_status": "COMPLIANT",
        "components": {
            "rev": 7_004_318.47,
            "opex": 5_507_154.04,
            "addback": 481_247.63 + 342_905.28,
        },
    },
    {
        "scenario": "P5",
        "clause": "6.1",
        "note": "group_capex NBV roll-forward / (rev − turbine opex)",
        "expected_actual": 9.45,
        "expected_status": "BREACH",
        "components": {
            "group_capex": 21_847_362.55,
            "rev": 8_214_663.28,
            "opex": 5_902_447.13,
        },
    },
    {
        "scenario": "P6",
        "clause": "6.1",
        "note": "RP retainer / kiln servicing opex",
        "expected_actual": 0.10,
        "expected_status": "BREACH",
        "components": {
            "num": 418_662.44,
            "den": 4_204_663.19,
        },
    },
    {
        "scenario": "P6",
        "clause": "6.2",
        "note": "W5.1: revenue / (payroll + utilities); draft utilities→taxes scrape denied",
        "expected_actual": 3.64,
        "expected_status": "COMPLIANT",
        "components": {
            "num": 6_918_204.37,
            "den": 1_482_663.19 + 418_204.37,
        },
    },
    {
        "scenario": "P7",
        "clause": "6.1",
        "note": "capex sum / (rev − opex)",
        "expected_actual": 0.36,
        "expected_status": "BREACH",
        "components": {
            "num": 402_118.64 + 486_204.19 + 91_447.35,
            "den": 9_146_882.53 - 6_418_003.77,
        },
    },
    {
        "scenario": "P8",
        "clause": "6.1",
        "note": "payroll ledger + off-ledger severance",
        "expected_actual": 4_221_314.95,
        "expected_status": "BREACH",
        "components": {
            "ledger": 2_418_663.27 + 884_204.16,
            "off_ledger": 918_447.52,
        },
    },
    {
        "scenario": "B1",
        "clause": "6.1",
        "note": "(rev−opex)/(interest+reclass interest); evidence B1-0020",
        "expected_actual": 1.68,
        "expected_status": "BREACH",
        "components": {
            "num": 9_741_934.78 - 6_166_592.66,
            "den": 1_540_833.29 + 592_296.10,
        },
    },
    {
        "scenario": "B4",
        "clause": "6.1",
        "note": "Q4 revenue only (exclude advance)",
        "expected_actual": 3_084_375.68,
        "expected_status": "BREACH",
        "components": {"q4": 3_084_375.68},
    },
    {
        "scenario": "P9",
        "clause": "6.1",
        "note": "unrestricted transfer / (capex ∪ all transfers)",
        "expected_actual": 0.22,
        "expected_status": "BREACH",
        "components": {
            "num": 418_204.37,
            "den": 1_204_663.28 + 302_118.64 + 418_204.37,
        },
    },
]


def compute_cell(cell: dict) -> float | None:
    """Recompute actual from documented components; round to 2 decimals at output."""
    c = cell.get("components") or {}
    if "num" in c and "den" in c and c["den"]:
        return round(abs(float(c["num"]) / float(c["den"])), 2)
    if "group_capex" in c:
        ebitda = float(c["rev"]) - float(c["opex"])
        return round(abs(float(c["group_capex"]) / ebitda), 2) if ebitda else None
    if "rev" in c and "opex" in c and "addback" in c:
        adj = float(c["rev"]) - float(c["opex"]) + float(c["addback"])
        return round(abs(adj / float(c["rev"])), 2) if c["rev"] else None
    if "ledger" in c and "off_ledger" in c:
        return round(abs(float(c["ledger"]) + float(c["off_ledger"])), 2)
    if "q4" in c:
        return round(abs(float(c["q4"])), 2)
    return None


def recon_rows() -> list[dict]:
    rows = []
    for cell in _RECON_CELLS:
        got = compute_cell(cell)
        ok = got is not None and abs(got - float(cell["expected_actual"])) < 0.015
        rows.append(
            {
                **{
                    k: cell[k]
                    for k in ("scenario", "clause", "note", "expected_actual", "expected_status")
                },
                "computed": got,
                "match_expected": ok,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/agentic-bank-public/ground_truth.json"),
    )
    args = ap.parse_args()

    gt: dict[tuple[str, str], dict] = {}
    if args.ground_truth.exists():
        raw = json.loads(args.ground_truth.read_text())
        for sid, sc in raw.get("scenarios", {}).items():
            for cl, cell in sc.get("covenants", {}).items():
                gt[(sid, cl)] = cell

    rows = []
    for base in recon_rows():
        g = gt.get((base["scenario"], base["clause"]), {})
        rows.append({**base, "gt_actual": g.get("actual"), "gt_status": g.get("status")})

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for r in rows:
            mark = "OK" if r["match_expected"] else "??"
            print(
                f"{mark} {r['scenario']}/{r['clause']}: computed={r['computed']} "
                f"expected={r['expected_actual']} gt={r['gt_actual']} — {r['note'][:60]}"
            )
    return 0 if all(r["match_expected"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
