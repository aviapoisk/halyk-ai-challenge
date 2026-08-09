"""S5 — Evidence. Flip-test: for a BREACH cell, try excluding each candidate
transaction and reclassifying it into every other category; see whether the
verdict flips to COMPLIANT.

R7 (mechanism-flip): prefer candidates that are auditor-touched / document-named
/ carve-out / related-party retainer mechanisms. Contribution to a sum alone is
not evidence (open GT: single-capex BREACH cells have evidence=null).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from agent.s4_engine import CATEGORIES

RecomputeFn = Callable[[pd.DataFrame, dict[str, str]], str]


def _collect_flippers(
    period_df: pd.DataFrame,
    candidate_txn_ids: list[str],
    categories: dict[str, str],
    recompute_status: RecomputeFn,
    category_universe: tuple[str, ...],
) -> list[str]:
    flippers: list[str] = []
    for txn_id in candidate_txn_ids:
        subset = period_df[period_df["txn_id"] != txn_id]
        if recompute_status(subset, categories) == "COMPLIANT":
            flippers.append(txn_id)
            continue
        current_cat = categories.get(txn_id)
        for new_cat in category_universe:
            if new_cat == current_cat:
                continue
            override = dict(categories)
            override[txn_id] = new_cat
            if recompute_status(period_df, override) == "COMPLIANT":
                flippers.append(txn_id)
                break
    seen: set[str] = set()
    uniq: list[str] = []
    for t in flippers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def flip_test(
    period_df: pd.DataFrame,
    candidate_txn_ids: list[str],
    categories: dict[str, str],
    recompute_status: RecomputeFn,
    current_status: str,
    doc_named_txn_ids: list[str] | None = None,
    category_universe: tuple[str, ...] = CATEGORIES,
    mechanism_txn_ids: list[str] | None = None,
) -> tuple[str | None, list[str], dict]:
    """Returns (evidence_txn_id_or_None, flip_diagnostics, detail).

    When several mechanical flippers exist, only a unique mechanism-pool flipper
    (auditor/doc-named ∪ explicit mechanism_txn_ids) is accepted; otherwise null.
    """
    doc_named = sorted(set(doc_named_txn_ids or []))
    mechanism = sorted(set(mechanism_txn_ids or []) | set(doc_named))
    empty_detail = {
        "flippers": [],
        "doc_named_candidates": doc_named,
        "mechanism_candidates": mechanism,
        "note": "",
    }

    if current_status != "BREACH":
        return None, [], empty_detail

    mech_set = set(mechanism)
    # Prefer mechanism pool first when non-empty.
    mech_first = [t for t in candidate_txn_ids if t in mech_set]
    if mech_first:
        flippers = _collect_flippers(
            period_df, mech_first, categories, recompute_status, category_universe
        )
        if len(flippers) == 1:
            return (
                flippers[0],
                [],
                {
                    "flippers": flippers,
                    "doc_named_candidates": doc_named,
                    "mechanism_candidates": mechanism,
                    "note": "единственный флиппер, mechanism-пул",
                },
            )

    doc_set = set(doc_named)
    doc_first = [t for t in candidate_txn_ids if t in doc_set]
    if doc_first and doc_first != mech_first:
        flippers = _collect_flippers(
            period_df, doc_first, categories, recompute_status, category_universe
        )
        if len(flippers) == 1:
            return (
                flippers[0],
                [],
                {
                    "flippers": flippers,
                    "doc_named_candidates": doc_named,
                    "mechanism_candidates": mechanism,
                    "note": "единственный флиппер, документный пул",
                },
            )

    flippers = _collect_flippers(
        period_df, candidate_txn_ids, categories, recompute_status, category_universe
    )

    if len(flippers) == 1:
        # W2.9: contribution-only singleton is never evidence. Require mechanism pool
        # membership; empty mechanism pool → null (no legacy accept).
        if not mechanism or flippers[0] not in mech_set:
            note = (
                f"flip-test: single contributor flipper {flippers[0]} "
                + (
                    "outside mechanism pool — evidence null"
                    if mechanism
                    else "with empty mechanism pool — evidence null"
                )
            )
            return (
                None,
                [note],
                {
                    "flippers": flippers,
                    "doc_named_candidates": doc_named,
                    "mechanism_candidates": mechanism,
                    "note": note,
                },
            )
        return (
            flippers[0],
            [],
            {
                "flippers": flippers,
                "doc_named_candidates": doc_named,
                "mechanism_candidates": mechanism,
                "note": "single flipper",
            },
        )
    if len(flippers) == 0:
        note = "flip-test: no single transaction flips the verdict"
        return (
            None,
            [note],
            {
                "flippers": [],
                "doc_named_candidates": doc_named,
                "mechanism_candidates": mechanism,
                "note": note,
            },
        )

    agreed = [t for t in flippers if t in mech_set]
    if len(agreed) == 1:
        note = (
            f"flip-test: {len(flippers)} mechanical flippers; "
            f"mechanism channel selected {agreed[0]}"
        )
        return (
            agreed[0],
            [note],
            {
                "flippers": flippers,
                "doc_named_candidates": doc_named,
                "mechanism_candidates": mechanism,
                "note": note,
            },
        )

    note = f"flip-test: {len(flippers)} transactions each flip the verdict independently — {flippers}"
    return (
        None,
        [note],
        {
            "flippers": flippers,
            "doc_named_candidates": doc_named,
            "mechanism_candidates": mechanism,
            "note": note,
        },
    )
