"""Offline golden gate: PLAN-36-V4 recon formulas match documented actuals."""

from __future__ import annotations

from tools.recon import _RECON_CELLS, compute_cell, recon_rows


def test_recon_all_cells_match_expected():
    rows = recon_rows()
    assert rows, "recon cell list must not be empty"
    failed = [r for r in rows if not r["match_expected"]]
    assert not failed, "recon mismatches: " + ", ".join(
        f"{r['scenario']}/{r['clause']} got={r['computed']} want={r['expected_actual']}"
        for r in failed
    )


def test_recon_covers_v4_defect_cells():
    keys = {(c["scenario"], c["clause"]) for c in _RECON_CELLS}
    for sid, cl in (
        ("P2", "6.1"),
        ("P3", "6.1"),
        ("P4", "6.1"),
        ("P5", "6.1"),
        ("P7", "6.1"),
        ("P8", "6.1"),
        ("P9", "6.1"),
        ("B1", "6.1"),
    ):
        assert (sid, cl) in keys
    assert ("P4", "6.3") not in keys  # open-set typo — do not chase


def test_compute_p5_group_capex():
    cell = next(c for c in _RECON_CELLS if c["scenario"] == "P5")
    assert compute_cell(cell) == 9.45
