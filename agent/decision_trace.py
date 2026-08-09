"""Builders for per-scenario / per-cell decision audit (trace.json → cases).

Pure functions, no I/O. LLM payloads stay in .cache/; this only keeps the
numbers and choices needed to triage a tools.score mismatch.
"""

from __future__ import annotations

from typing import Any

from agent.s4_engine import CovenantResult, make_stub_cell
from agent.schemas import (
    CovenantSpec,
    DocumentLinkage,
    IngestedDocument,
    LoanAgreementFacts,
    Reclassification,
)


def build_binding(
    scenario_id: str,
    linkages: list[DocumentLinkage],
    by_type: dict[str, list[IngestedDocument]],
    loan_facts: LoanAgreementFacts | None,
) -> dict:
    link_by_id = {l.doc_id: l for l in linkages if l.scenario_id == scenario_id}
    docs: list[dict] = []
    for doc_type, typed_docs in sorted(by_type.items()):
        for doc in typed_docs:
            link = link_by_id.get(doc.doc_id)
            docs.append(
                {
                    "doc_id": doc.doc_id,
                    "type": doc_type,
                    "channel": link.link_channel if link else None,
                }
            )
    return {
        "account_id": loan_facts.account_id if loan_facts else None,
        "borrower_name": loan_facts.borrower_name if loan_facts else None,
        "docs": docs,
    }


def build_loan_section(loan_facts: LoanAgreementFacts | None, template_clauses: list[str]) -> dict:
    if loan_facts is None:
        return {
            "covenants_extracted": [],
            "clauses_missing_from_extract": sorted(template_clauses),
        }
    extracted = []
    seen: set[str] = set()
    for spec in loan_facts.covenants:
        seen.add(spec.clause)
        extracted.append(
            {
                "clause": spec.clause,
                "covenant_type": spec.covenant_type,
                "comparator": spec.comparator,
                "threshold": spec.threshold,
                "period_start": spec.period_start,
                "period_end": spec.period_end,
                "formula_description": spec.formula_description,
                "filter_category": spec.filter_category,
                "numerator_category": spec.numerator_category,
                "denominator_category": spec.denominator_category,
                "quote": spec.quote,
            }
        )
    missing = sorted(c for c in template_clauses if c not in seen)
    return {"covenants_extracted": extracted, "clauses_missing_from_extract": missing}


def build_auditor_section(
    doc_id: str | None,
    reclassifications: list[Reclassification],
    applied_keys: list[str],
    unmatched_indices: list[int],
) -> dict:
    applied_set = set(applied_keys)
    unmatched_set = set(unmatched_indices)
    items = []
    for i, rec in enumerate(reclassifications):
        key = rec.txn_id or f"idx:{i}"
        applied = key in applied_set or (rec.txn_id is not None and rec.txn_id in applied_set)
        if i in unmatched_set:
            applied = False
        items.append(
            {
                "txn_id": rec.txn_id,
                "action": rec.action,
                "from_category": rec.from_category,
                "to_category": rec.to_category,
                "applied": applied,
            }
        )
    return {"doc_id": doc_id, "reclassifications": items}


def _spec_slim(spec: CovenantSpec | None) -> dict | None:
    if spec is None:
        return None
    return {
        "clause": spec.clause,
        "covenant_type": spec.covenant_type,
        "comparator": spec.comparator,
        "threshold": spec.threshold,
        "period_start": spec.period_start,
        "period_end": spec.period_end,
        "filter_category": spec.filter_category,
        "numerator_category": spec.numerator_category,
        "denominator_category": spec.denominator_category,
    }


def diagnose_cell(cell_trace: dict) -> str:
    if cell_trace.get("outcome") == "stubbed":
        reason = cell_trace.get("stub_reason") or "stubbed"
        return f"clause absent from extract → stubbed" if "not computed" in reason or "absent" in reason else reason
    evidence = cell_trace.get("evidence") or {}
    note = evidence.get("note") or ""
    if note and "no single" in note:
        return "computed; flip-test found no unique evidence"
    if note and "mechanical flippers" in note:
        return f"computed; evidence via document channel ({evidence.get('chosen')})"
    if note and "independently" in note:
        return "computed; ambiguous flip-test → evidence null"
    return "computed"


def build_cell_trace(
    pred: dict,
    *,
    spec: CovenantSpec | None = None,
    result: CovenantResult | None = None,
    evidence_detail: dict | None = None,
    stub_reason: str | None = None,
    clause_flags: list[str] | None = None,
) -> dict:
    detail = evidence_detail or {"flippers": [], "doc_named_candidates": [], "note": ""}
    outcome = "stubbed" if stub_reason else "computed"
    cell: dict[str, Any] = {
        "pred": {
            "status": pred.get("status"),
            "actual": pred.get("actual"),
            "evidence_txn_id": pred.get("evidence_txn_id"),
        },
        "outcome": outcome,
        "stub_reason": stub_reason,
        "spec": _spec_slim(spec),
        "inputs": dict(result.inputs) if result is not None else {},
        "contributing_txn_ids": list(result.contributing_txn_ids) if result is not None else [],
        "evidence": {
            "chosen": pred.get("evidence_txn_id"),
            "flippers": list(detail.get("flippers") or []),
            "doc_named_candidates": list(detail.get("doc_named_candidates") or []),
            "note": detail.get("note") or "",
        },
        "flags": list(clause_flags or []),
    }
    if stub_reason and "not computed" in stub_reason:
        cell["diagnosis"] = "clause absent from extract → stubbed"
    else:
        cell["diagnosis"] = diagnose_cell(cell)
    return cell


def empty_case_trace(scenario_id: str, status: str, flags: list[str], template_clauses: list[str]) -> dict:
    cells = {
        clause: build_cell_trace(
            make_stub_cell(),
            stub_reason="scenario unprocessed; stubbed",
        )
        for clause in template_clauses
    }
    return {
        "scenario_id": scenario_id,
        "status": status,
        "binding": {"account_id": None, "borrower_name": None, "docs": []},
        "loan": {"covenants_extracted": [], "clauses_missing_from_extract": sorted(template_clauses)},
        "auditor": {"doc_id": None, "reclassifications": []},
        "categorization": {
            "n_rows": 0,
            "n_labeled": 0,
            "missing_ids": [],
            "by_category_count": {},
            "by_category_sum": {},
        },
        "fx_rate": None,
        "group_capex": None,
        "cells": cells,
        "flags": list(flags),
    }
