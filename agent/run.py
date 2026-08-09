"""CLI entry point: python -m agent.run --dataset <DIR> --out <FILE.json> [...]

S1-S2 run deterministically across the whole corpus; S3-S6 run end to end for every
scenario (G2). `--only <scenario_id>` remains a dev-only scope limiter. Every cell is
always filled — stubs + confidence-report flags when a scenario fails mid-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agent.carveouts import (
    apply_carve_outs,
    carve_out_status_flip_flag,
    clause_has_carve_out_language,
    verify_carve_out_quotes,
)
from agent.dataset import discover_dataset
from agent.decision_trace import (
    build_auditor_section,
    build_binding,
    build_cell_trace,
    build_loan_section,
    empty_case_trace,
)
from agent.doc_inventory import (
    build_linking_warnings,
    build_s2_entries,
    format_s2_unlinked_log_line,
    s2_summary,
)
from agent.expr_tree import (
    ExprContext,
    evaluate_expr,
    sanitize_expr_on_spec,
    synthesize_expr_from_enum,
)
from agent.ledger_signals import build_ledger_signal_index
from agent.llm import LLMClient
from agent.normalize import clause_key
from agent.progress import ProgressSink, create_progress_sink, format_llm_label
from agent.s1_ingest import ingest_directory
from agent.s1_vision import ocr_scan_document
from agent.s2_classify import classify_doc_type, find_group_capex_document
from agent.s2_link import link_documents, load_account_to_scenario
from agent.s3_extract import (
    confirm_related_party_overlap,
    extract_auditor_notes,
    extract_group_capex,
    extract_kyc,
    extract_loan_agreement,
)
from agent.s4_engine import (
    CATEGORIES,
    apply_fx,
    apply_pledge_based_transfer_categories,
    apply_reclassifications,
    apply_relevance_and_retainer_filters,
    apply_sanity_gates,
    apply_springing,
    categorize_transactions,
    category_summary,
    borrower_own_capex,
    compute_group_capex,
    ensure_formula_categories_present,
    evaluate_max_category,
    evaluate_ratio_generic,
    evaluate_ratio_group_capex_to_ebitda,
    evaluate_revenue_minus_max_category,
    evaluate_sum_category,
    evaluate_sum_group_capex,
    evaluate_sum_related_party_payments,
    evaluate_sum_revenue,
    evaluate_description_bucket,
    evaluate_sum_categories_vs_group_pct,
    evaluate_ebitda_plus_categories,
    fill_missing_amounts_from_reclassifications,
    is_absolute_group_capex_ceiling,
    reduce_over_calendar_quarters,
    sum_description_bucket,
    filter_df_by_relevance,
    filter_period,
    infer_to_category_from_quote,
    fx_facts_from_embedded,
    load_ledger,
    make_stub_cell,
    normalize_off_ledger_additions,
    prepare_aggregation_frame,
    recategorize_filled_amount_rows,
    looks_like_group_capex_ebitda_ratio,
    refine_covenant_spec,
    related_party_token_overlap_candidates,
    repair_degenerate_categories,
    resolve_fx_rate,
    scrape_materiality_from_notes,
    scrape_missing_txn_amounts_from_notes,
    scrape_off_ledger_from_notes,
    scenario_ledger,
    spec_incomplete,
    stub_cell_after_sanity,
    validate_spec_against_taxonomy,
)
from agent.s5_evidence import flip_test
from agent.s6_report import write_confidence_report, write_run_config, write_submission_atomic
from agent.schemas import (
    AuditorNotesFacts,
    CovenantSpec,
    DocumentLinkage,
    IngestedDocument,
    KYCFacts,
    LoanAgreementFacts,
    Reclassification,
)
from agent.sections import longest_section_around_anchor

LOGS_DIR = Path("logs")


def resolve_logs_dir(out_path: Path, *, cwd: Path | None = None) -> Path:
    """Where to write trace / confidence / config for this run.

    `--out submission.json` (file in CWD) → `logs/` at CWD (classic).
    `--out runs/<id>/submission.json` (or any non-CWD path) → `<out.parent>/logs/`
    so each archived run keeps its own debug bundle next to the submission.
    """
    out = Path(out_path)
    base = Path.cwd() if cwd is None else Path(cwd)
    try:
        rel = out if not out.is_absolute() else out.resolve().relative_to(base.resolve())
    except ValueError:
        return out.parent / "logs"
    if len(Path(rel).parts) <= 1:
        return LOGS_DIR if cwd is None else Path(cwd) / "logs"
    return out.parent / "logs"


def git_sha(*, repo: Path | None = None) -> str:
    """Best-effort HEAD sha for run archives; empty string when git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def build_run_config(config, *, sha: str | None = None) -> dict:
    """CLI/env keys for this run (no secrets) — mirrored into logs/config.json + trace.json."""
    return {
        "cache_dir": str(config.cache_dir),
        "dataset_dir": str(config.dataset_dir),
        "dev": bool(config.dev),
        "engine": config.engine,
        "git_sha": sha if sha is not None else git_sha(),
        "llm_adaptive_thinking": bool(config.llm_adaptive_thinking),
        "llm_concurrency": int(config.llm_concurrency),
        "llm_effort": config.llm_effort,
        "model_name": config.model_name,
        "model_provider": config.model_provider,
        "only_scenario": config.only_scenario,
        "out_path": str(config.out_path),
        "profile": config.profile,
        "seed": int(config.seed),
        "self_consistency": bool(config.self_consistency),
        "use_cache": bool(config.use_cache),
        "vision_char_threshold": int(config.vision_char_threshold),
    }


def _merge_auditor_facts(
    auditor_docs: list[IngestedDocument],
    auditor_facts_list: list[AuditorNotesFacts],
) -> tuple[
    AuditorNotesFacts | None,
    list[Reclassification],
    list,
    str | None,
    list[str],
    list[str],
]:
    """Merge all auditor extracts: reclass/off-ledger deduped; flag when >1 doc.

    R8 / W5.1: interim/draft notes do not contribute applied reclassifications or
    scrape corpus when an authoritative sibling exists. Returns note-named txn_ids
    (from all docs, including drafts) for evidence-only (W5.2).
    """
    from agent.s2_classify import (
        auditor_doc_is_authoritative,
        auditor_doc_is_interim_or_draft,
        auditor_notes_deny_reclassifications,
    )
    from agent.s4_engine import extract_txn_ids_from_notes

    flags: list[str] = []
    if not auditor_facts_list:
        return None, [], [], None, flags, []
    if len(auditor_facts_list) > 1:
        flags.append(f"multiple auditor_notes docs={len(auditor_facts_list)}; merged with dedup")

    paired_all = list(zip(auditor_facts_list, auditor_docs))
    # W5.2: TXN ids from every auditor doc (draft included) — evidence-only, never category.
    note_named_txn_ids = extract_txn_ids_from_notes(
        [d.normalized_text for _, d in paired_all]
        + [n for f, _ in paired_all for n in (f.reclassification_notes or [])]
    )

    authoritative = [
        (f, d) for f, d in paired_all if auditor_doc_is_authoritative(d.normalized_text)
    ]
    interims = [
        (f, d) for f, d in paired_all if auditor_doc_is_interim_or_draft(d.normalized_text)
    ]
    draft_for_materiality: list[tuple[AuditorNotesFacts, IngestedDocument]] = []
    if authoritative and interims:
        flags.append(
            f"auditor interim/draft skipped n={len(interims)} in favour of "
            f"authoritative n={len(authoritative)}"
        )
        draft_for_materiality = list(interims)
        paired = list(authoritative)
    elif interims and not authoritative:
        flags.append(
            f"auditor only interim/draft docs n={len(interims)}; "
            "draft reclassifications withheld from metric (notes/materiality/off-ledger kept)"
        )
        paired = list(paired_all)
        draft_for_materiality = list(interims)
    else:
        paired = list(paired_all)

    reclassifications: list[Reclassification] = []
    seen_reclass: set[tuple] = set()
    off_ledger: list = []
    seen_off: set[tuple] = set()
    scrape_notes: list[str] = []
    materiality: float | None = None
    any_present = False
    borrower = paired[0][0].borrower_name if paired else auditor_facts_list[0].borrower_name
    note_quotes: list[str] = []
    eur_usd_rate = eur_amount = usd_amount = None
    fx_quote = ""
    primary_doc_id = paired[0][1].doc_id if paired else auditor_docs[0].doc_id
    reclass_denied = False

    for facts, doc in paired:
        is_draft = auditor_doc_is_interim_or_draft(doc.normalized_text)
        # Authoritative path already dropped drafts from paired; keep guard.
        if is_draft and authoritative:
            continue
        if auditor_notes_deny_reclassifications(doc.normalized_text):
            reclass_denied = True
        any_present = any_present or facts.reclassifications_present
        note_quotes.extend(facts.reclassification_notes)
        # W5.1: never scrape draft body when authoritative sibling exists.
        if not is_draft:
            scrape_notes.extend(facts.reclassification_notes)
            scrape_notes.append(doc.normalized_text)
        if facts.materiality_threshold_usd is not None:
            m = abs(float(facts.materiality_threshold_usd))
            materiality = m if materiality is None else min(materiality, m)
        if not is_draft:
            for r in facts.reclassifications:
                key = (r.txn_id, round(float(r.amount), 2) if r.amount is not None else None, r.action)
                if key in seen_reclass:
                    continue
                seen_reclass.add(key)
                reclassifications.append(r)
        for add in getattr(facts, "off_ledger_additions", []) or []:
            # Abs-key so +X / -X from scrape vs LLM collapse (UPGRADE-PLAN-V2 A3).
            key = (round(abs(float(add.amount)), 2), getattr(add, "category", None))
            if key in seen_off:
                continue
            seen_off.add(key)
            off_ledger.append(add)
        if eur_usd_rate is None and (
            facts.eur_usd_rate is not None or (facts.eur_amount and facts.usd_amount)
        ):
            eur_usd_rate = facts.eur_usd_rate
            eur_amount = facts.eur_amount
            usd_amount = facts.usd_amount
            fx_quote = facts.fx_quote or ""

    non_draft_paired = [
        (f, d) for f, d in paired if not auditor_doc_is_interim_or_draft(d.normalized_text)
    ]
    scraped_fills = (
        scrape_missing_txn_amounts_from_notes(scrape_notes)
        if non_draft_paired or authoritative
        else []
    )
    existing_txn = {str(r.txn_id) for r in reclassifications if r.txn_id and r.amount is not None}
    existing_any = {str(r.txn_id) for r in reclassifications if r.txn_id}
    # W5.1 denial: authoritative «переклассификац… не требова» → no inventing new
    # category-move scrapes (amount-fill onto existing structured rows still ok).
    if reclass_denied and not any(
        (r.status or "applied") == "applied" and r.action == "reclassify" and r.to_category
        for r in reclassifications
    ):
        flags.append("auditor reclass denial: scraped category-move rows withheld")
        for rec in scraped_fills:
            tid = str(rec.txn_id) if rec.txn_id else ""
            if not tid or rec.amount is None or tid in existing_txn:
                continue
            if tid in existing_any:
                # Fill amount on an existing structured row.
                for r in reclassifications:
                    if str(r.txn_id) == tid and r.amount is None:
                        r.amount = rec.amount
                        flags.append(f"scraped missing amount for {tid} from auditor notes")
                        existing_txn.add(tid)
                        break
            else:
                # Amount-only fill for NaN ledger rows (treasury / notes) — not a category move.
                reclassifications.append(rec)
                flags.append(f"scraped missing amount for {tid} from notes/treasury (denial-safe fill)")
                existing_txn.add(tid)
    else:
        for rec in scraped_fills:
            if rec.txn_id and str(rec.txn_id) not in existing_txn:
                reclassifications.append(rec)
                flags.append(f"scraped missing amount for {rec.txn_id} from auditor notes")
                existing_txn.add(str(rec.txn_id))

    for add in scrape_off_ledger_from_notes(scrape_notes):
        key = (round(abs(float(add.amount)), 2), getattr(add, "category", None))
        if key not in seen_off:
            off_ledger.append(add)
            seen_off.add(key)
            flags.append("scraped off-ledger covenant addition from auditor notes")
    # Drop target_txn_id restorations + abs-dedup (ledger amount match happens later with scen_ledger).
    off_ledger, off_norm_flags = normalize_off_ledger_additions(off_ledger)
    flags.extend(off_norm_flags)

    if materiality is None:
        scraped_mat = scrape_materiality_from_notes(scrape_notes)
        if scraped_mat is None and draft_for_materiality:
            # W5.1: materiality from draft only when authoritative corpus had none.
            draft_notes = [d.normalized_text for _, d in draft_for_materiality]
            draft_notes.extend(
                n for f, _ in draft_for_materiality for n in (f.reclassification_notes or [])
            )
            scraped_mat = scrape_materiality_from_notes(draft_notes)
        if scraped_mat is not None:
            materiality = scraped_mat
            flags.append(f"scraped materiality_threshold_usd={scraped_mat} from auditor notes")

    if any_present and not reclassifications:
        flags.append(
            f"auditor reclassifications disclosed but unstructured notes only: {note_quotes[:3]}"
        )

    merged = AuditorNotesFacts(
        borrower_name=borrower,
        reclassifications_present=any_present and not reclass_denied,
        reclassification_notes=note_quotes,
        reclassifications=reclassifications,
        off_ledger_additions=off_ledger,
        materiality_threshold_usd=materiality,
        eur_usd_rate=eur_usd_rate,
        eur_amount=eur_amount,
        usd_amount=usd_amount,
        fx_quote=fx_quote,
    )
    # When denial: note-named TXNs are evidence-only candidates (no category apply).
    evidence_note_named = note_named_txn_ids if reclass_denied else []
    return (
        merged,
        reclassifications,
        off_ledger,
        primary_doc_id,
        flags,
        evidence_note_named,
    )


def _stub_cell(spec: CovenantSpec | None = None) -> dict:
    """Prefer covenant threshold as actual — except group-capex (never threshold)."""
    if spec is not None and spec.covenant_type == "ratio_group_capex_to_ebitda":
        return make_stub_cell(status="BREACH", actual=0.01, keep_status=True)
    if spec is not None and spec.threshold is not None:
        return make_stub_cell(threshold=spec.threshold)
    return make_stub_cell()


def missing_linked_type_flags(by_type: dict[str, list]) -> list[str]:
    """Warning flags when optional doc types are absent (G2). Does not change answers."""
    flags: list[str] = []
    if not by_type.get("kyc_dossier"):
        flags.append("no kyc_dossier linked to scenario")
    if not by_type.get("auditor_notes"):
        flags.append("no auditor_notes linked to scenario")
    return flags


def _dev_docs(
    scenario_id: str, linkages: list[DocumentLinkage], docs_by_id: dict[str, IngestedDocument]
) -> list[dict]:
    """Dev-dump section: every document linked to the scenario, with full normalized text."""
    entries: list[dict] = []
    for link in linkages:
        if link.scenario_id != scenario_id:
            continue
        doc = docs_by_id[link.doc_id]
        entries.append(
            {
                "doc_id": doc.doc_id,
                "path": doc.path,
                "type": classify_doc_type(doc),
                "link_channel": link.link_channel,
                "account_anchor": link.account_anchor,
                "company_name": link.company_name,
                "n_pages": len(doc.pages),
                "chars": sum(p.char_count for p in doc.pages),
                "is_scan": doc.is_scan,
                "text": doc.normalized_text,
            }
        )
    return sorted(entries, key=lambda e: (e["type"] or "~unclassified", e["doc_id"]))


def _dev_transactions(df, categories: dict[str, str], excluded_ids: set[str]) -> list[dict]:
    """Dev-dump section: every scenario ledger row with its category and reclass fate."""
    if df is None or df.empty:
        return []
    rows: list[dict] = []
    for r in df.sort_values(["date", "txn_id"]).itertuples():
        txn_id = str(r.txn_id)
        rows.append(
            {
                "txn_id": txn_id,
                "date": str(r.date.date()),
                "counterparty": str(r.counterparty),
                "description": str(r.description),
                "amount": float(r.amount),
                "currency": str(r.currency),
                "amount_usd": float(getattr(r, "amount_usd", r.amount)),
                "category": categories.get(txn_id),
                "excluded_by_reclass": txn_id in excluded_ids,
            }
        )
    return rows


def _select_loan_facts(
    loan_facts_list: list[LoanAgreementFacts],
) -> tuple[LoanAgreementFacts, list[str]]:
    """Prefer non-superseded edition with best coverage of its own covenant periods."""
    flags: list[str] = []
    current = [f for f in loan_facts_list if not f.is_superseded]
    if len(current) == 1:
        return current[0], flags
    if not current:
        flags.append("no non-superseded loan agreement; falling back to first extracted")
        return loan_facts_list[0], flags

    flags.append(f"expected exactly one non-superseded loan agreement, found {len(current)}")

    def _parse(d: str):
        try:
            return __import__("datetime").date.fromisoformat(d[:10])
        except Exception:
            return None

    def coverage_key(f: LoanAgreementFacts) -> tuple:
        ef, et = _parse(f.effective_from), _parse(f.effective_to)
        # Score how many covenants' periods fall inside [ef, et].
        exact = 0
        overlap_days = 0
        for c in f.covenants:
            cs, ce = _parse(c.period_start), _parse(c.period_end)
            if ef and et and cs and ce:
                if ef <= cs and et >= ce:
                    exact += 1
                lo = max(ef, cs)
                hi = min(et, ce)
                if hi >= lo:
                    overlap_days += (hi - lo).days + 1
        # Higher is better; stamp already filtered; doc_id for stability via contract_date.
        return (exact, overlap_days, f.effective_from or "", f.contract_date or "")

    chosen = sorted(current, key=coverage_key, reverse=True)[0]
    flags.append(f"selected loan effective_from={chosen.effective_from} account={chosen.account_id}")
    return chosen, flags


def _normalize_clause_keys(
    loan_facts: LoanAgreementFacts,
    template_clauses: list[str],
    document_text: str | None = None,
) -> tuple[LoanAgreementFacts, list[str]]:
    """Rewrites every covenant's clause to the canonical template key ("6.1").

    Weak models return the whole clause heading; without this, computed cells never
    match `submission_template.json` keys and the whole scenario is silently stubbed
    (root cause of the 2026-08-08 all-stubs run). Duplicate keys keep the first spec.
    """
    flags: list[str] = []
    seen: set[str] = set()
    normalized: list[CovenantSpec] = []
    for spec in loan_facts.covenants:
        key = clause_key(spec.clause, template_clauses)
        if key != spec.clause:
            flags.append(f"clause key normalized: {spec.clause!r} -> {key!r}")
            spec = spec.model_copy(update={"clause": key})
        if key in seen:
            flags.append(f"duplicate covenant clause {key} after normalization; kept the first")
            continue
        seen.add(key)
        if document_text and (spec.quote or "").strip():
            from agent.threshold_check import quote_in_document, threshold_appears

            if not quote_in_document(spec.quote, document_text):
                flags.append(f"QUOTE_NOT_IN_TEXT clause {key}")
            if not threshold_appears(float(spec.threshold), spec.quote) and not threshold_appears(
                float(spec.threshold), document_text
            ):
                flags.append(f"THRESHOLD_NOT_IN_TEXT clause {key} threshold={spec.threshold}")
            qlow = (spec.quote or "").casefold()
            if clause_has_carve_out_language(qlow):
                flags.append(f"CARVE_OUT_LANGUAGE clause {key}")
        normalized.append(refine_covenant_spec(spec, document_text=document_text))
    return loan_facts.model_copy(update={"covenants": normalized}), flags


def _report_clause_key_mismatch(
    sink: ProgressSink,
    *,
    scenario_id: str,
    template_clauses: list[str],
    loan_facts: LoanAgreementFacts,
    loan_doc_ids: list[str],
) -> list[str]:
    """Warn when template clause keys and S3 extract keys diverge.

    Missing template keys will be stubbed later; orphan extract keys are evaluated
    but dropped from submission. Both need operator visibility on battle day.
    """
    extracted_keys = [spec.clause for spec in loan_facts.covenants]
    extracted_set = set(extracted_keys)
    template_set = set(template_clauses)
    missing = sorted(template_set - extracted_set)
    orphan_specs = [spec for spec in loan_facts.covenants if spec.clause not in template_set]
    orphan_keys = [spec.clause for spec in orphan_specs]
    if not missing and not orphan_keys:
        return []

    docs = list(loan_doc_ids)
    extracted_summary = ",".join(extracted_keys) if extracted_keys else "(none)"
    orphan_quote_bits = [
        f"{spec.clause}:{(spec.quote or '')[:80]}" for spec in orphan_specs[:3]
    ]
    orphan_quotes_preview = " | ".join(orphan_quote_bits) if orphan_quote_bits else ""

    sink.log(
        f"[CLAUSE_KEY_MISMATCH] {scenario_id} "
        f"template={sorted(template_clauses)} extracted={extracted_keys} docs={docs}"
    )
    for spec in orphan_specs:
        sink.log(
            f"[CLAUSE_KEY_MISMATCH] orphan {scenario_id}/{spec.clause} "
            f"type={spec.covenant_type} threshold={spec.threshold} "
            f"period={spec.period_start}..{spec.period_end} quote={spec.quote!r}"
        )
    for clause in missing:
        sink.log(
            f"[CLAUSE_KEY_MISMATCH] missing {scenario_id}/{clause} "
            f"(will stub; extracted={extracted_summary})"
        )

    for clause in missing:
        msg = (
            f"template clause {clause} absent from loan extract; cell will be stubbed. "
            f"extracted=[{extracted_summary}]"
        )
        sink.alert(
            "CLAUSE_MISSING_FROM_EXTRACT",
            msg,
            severity="warning",
            scenario_id=scenario_id,
            clause=clause,
            docs=docs,
            quote_preview=orphan_quotes_preview[:160],
        )
    for spec in orphan_specs:
        formula = f"{spec.covenant_type} threshold={spec.threshold}"
        sink.alert(
            "CLAUSE_ORPHAN_EXTRACT",
            (
                f"extracted clause {spec.clause} not in submission_template keys; "
                f"computed but dropped from answers"
            ),
            severity="warning",
            scenario_id=scenario_id,
            clause=spec.clause,
            docs=docs,
            formula_description=formula,
            quote_preview=(spec.quote or "")[:160],
        )

    return [
        f"clause-key mismatch: missing={missing} orphan={orphan_keys}",
    ]


def _select_kyc(
    kyc_docs: list[IngestedDocument], loan_account_id: str | None, kyc_facts_list: list[KYCFacts]
) -> tuple[KYCFacts | None, list[str]]:
    flags: list[str] = []
    if not kyc_facts_list:
        return None, flags
    if loan_account_id:
        matched = [f for f in kyc_facts_list if f.account_id == loan_account_id]
        if len(matched) == 1:
            return matched[0], flags
        if len(matched) > 1:
            flags.append(f"{len(matched)} KYC dossiers match account {loan_account_id}; used the first")
            return matched[0], flags
    if len(kyc_docs) > 1:
        flags.append(
            f"{len(kyc_docs)} KYC dossiers linked; none matched account {loan_account_id}; used the first"
        )
    return kyc_facts_list[0], flags


def _resolve_embedded_fx(
    *,
    source: str,
    doc_id: str,
    eur_usd_rate: float | None,
    eur_amount: float | None,
    usd_amount: float | None,
    fx_quote: str,
) -> tuple[float | None, str, dict, list[str]]:
    """Build FxRateFacts from embedded loan/auditor fields and resolve."""
    fx_obj = fx_facts_from_embedded(
        eur_usd_rate=eur_usd_rate,
        eur_amount=eur_amount,
        usd_amount=usd_amount,
        fx_quote=fx_quote,
    )
    attempt = {
        "source": source,
        "doc_id": doc_id,
        "rate": fx_obj.eur_usd_rate,
        "eur_amount": fx_obj.eur_amount,
        "usd_amount": fx_obj.usd_amount,
        "quote": fx_obj.quote,
    }
    rate, resolve_flags = resolve_fx_rate(fx_obj)
    quote = fx_obj.quote if rate is not None else ""
    return rate, quote, attempt, resolve_flags


@dataclass
class _CaseExtract:
    """S3 extract bundle for one scenario — evaluate runs after global FX fallback is known."""

    scenario_id: str
    status: str  # "ready" | "unprocessed"
    flags: list[str] = field(default_factory=list)
    citations: dict[str, str] = field(default_factory=dict)
    by_type: dict[str, list[IngestedDocument]] = field(default_factory=dict)
    loan_docs: list[IngestedDocument] = field(default_factory=list)
    loan_facts_list: list[LoanAgreementFacts] = field(default_factory=list)
    loan_facts: LoanAgreementFacts | None = None
    selected_loan_doc: IngestedDocument | None = None
    kyc_docs: list[IngestedDocument] = field(default_factory=list)
    kyc_facts_list: list[KYCFacts] = field(default_factory=list)
    kyc_facts: KYCFacts | None = None
    auditor_docs: list[IngestedDocument] = field(default_factory=list)
    auditor_doc_id: str | None = None
    auditor_facts: AuditorNotesFacts | None = None
    reclassifications: list[Reclassification] = field(default_factory=list)
    off_ledger_additions: list = field(default_factory=list)
    # W5.2: TXN ids from auditor prose under reclass-denial — evidence triage only.
    note_named_txn_ids: list[str] = field(default_factory=list)
    own_fx_rate: float | None = None
    fx_quote: str = ""
    fx_attempts: list[dict] = field(default_factory=list)
    template_clauses: list[str] = field(default_factory=list)
    report_entry: dict | None = None
    case_trace: dict | None = None


async def _extract_case(
    scenario_id: str,
    linkages: list[DocumentLinkage],
    docs_by_id: dict[str, IngestedDocument],
    llm: LLMClient,
    cache_dir: Path,
    use_cache: bool,
    sink: ProgressSink,
    template_clauses: list[str],
    self_consistency: bool = False,
    dev: bool = False,
) -> _CaseExtract:
    """Parallel loan ∥ KYC ∥ auditor extract; FX read from embedded schema fields (no extra LLM)."""
    sink.scenario_start(scenario_id)
    scenario_docs = [docs_by_id[l.doc_id] for l in linkages if l.scenario_id == scenario_id]
    by_type: dict[str, list[IngestedDocument]] = {}
    for doc in scenario_docs:
        doc_type = classify_doc_type(doc)
        if doc_type:
            by_type.setdefault(doc_type, []).append(doc)

    flags: list[str] = []
    citations: dict[str, str] = {}
    loan_docs = by_type.get("loan_agreement", [])
    if not loan_docs:
        # A4: regex miss (e.g. English headers) — one LLM classify per untyped linked doc.
        from agent.schemas import DocTypeGuess

        untyped = [d for d in scenario_docs if classify_doc_type(d) is None]
        for doc in untyped[:5]:
            try:
                label = format_llm_label(scenario_id=scenario_id, doc_id=doc.doc_id, extra="doc-type")
                guess: DocTypeGuess = await llm.extract(
                    "Classify this corporate-loan corpus document. "
                    "Return doc_type and a verbatim quote that justifies it. "
                    "Use loan_agreement for bank loan contracts / facility agreements.",
                    doc.normalized_text[:6000],
                    DocTypeGuess,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    label=label,
                )
                if guess.doc_type != "other" and (guess.quote or "").strip():
                    by_type.setdefault(guess.doc_type, []).append(doc)
                    flags.append(f"LLM doc_type {guess.doc_type} for {doc.doc_id}")
            except Exception as exc:  # noqa: BLE001
                flags.append(f"LLM doc_type failed {doc.doc_id}: {type(exc).__name__}")
        loan_docs = by_type.get("loan_agreement", [])
    if not loan_docs:
        flags.append("no loan_agreement document linked to this scenario")
        sink.scenario_end(scenario_id, "unprocessed — no loan_agreement")
        case_trace = empty_case_trace(scenario_id, "unprocessed", flags, template_clauses)
        case_trace["binding"] = build_binding(scenario_id, linkages, by_type, None)
        if dev:
            case_trace["dev"] = {
                "documents": _dev_docs(scenario_id, linkages, docs_by_id),
                "extracts": None,
                "transactions": [],
            }
        return _CaseExtract(
            scenario_id=scenario_id,
            status="unprocessed",
            flags=flags,
            by_type=by_type,
            template_clauses=template_clauses,
            report_entry={"scenario_id": scenario_id, "status": "unprocessed", "flags": flags},
            case_trace=case_trace,
        )

    flags.extend(missing_linked_type_flags(by_type))
    sink.stage_start("S3_extract", f"{scenario_id}: loan / KYC / auditor")
    doc_anchor = {
        l.doc_id: (l.account_anchor or "")
        for l in linkages
        if l.scenario_id == scenario_id
    }
    kyc_docs = by_type.get("kyc_dossier", [])
    auditor_docs = by_type.get("auditor_notes", [])

    sink.log(
        f"[S3] extract loan×{len(loan_docs)} kyc×{len(kyc_docs)} "
        f"auditor×{len(auditor_docs)} ({scenario_id})"
    )

    async def _noop_auditor() -> list[AuditorNotesFacts]:
        return []

    async def _noop_kyc() -> list[tuple[KYCFacts, list[str]]]:
        return []

    loan_coro = asyncio.gather(
        *(
            extract_loan_agreement(
                llm,
                d,
                cache_dir,
                use_cache=use_cache,
                scenario_id=scenario_id,
                account_id=doc_anchor.get(d.doc_id, ""),
                template_clauses=template_clauses,
            )
            for d in loan_docs
        )
    )
    kyc_coro = (
        asyncio.gather(
            *(
                extract_kyc(
                    llm,
                    d,
                    cache_dir,
                    use_cache=use_cache,
                    scenario_id=scenario_id,
                    account_id=doc_anchor.get(d.doc_id, ""),
                )
                for d in kyc_docs
            )
        )
        if kyc_docs
        else _noop_kyc()
    )
    auditor_coro = (
        asyncio.gather(
            *(
                extract_auditor_notes(
                    llm,
                    d,
                    cache_dir,
                    use_cache=use_cache,
                    scenario_id=scenario_id,
                    account_id=doc_anchor.get(d.doc_id, ""),
                )
                for d in auditor_docs
            )
        )
        if auditor_docs
        else _noop_auditor()
    )

    loan_facts_list_t, kyc_facts_list_t, auditor_facts_list_t = await asyncio.gather(
        loan_coro, kyc_coro, auditor_coro
    )
    loan_facts_list = list(loan_facts_list_t)
    kyc_enriched = list(kyc_facts_list_t)
    kyc_facts_list = [facts for facts, _flags in kyc_enriched]
    kyc_enrich_flags = [_flags for _facts, _flags in kyc_enriched]
    auditor_facts_list = list(auditor_facts_list_t)

    loan_facts, loan_flags = _select_loan_facts(loan_facts_list)
    flags.extend(loan_flags)
    selected_loan_doc = next(d for f, d in zip(loan_facts_list, loan_docs) if f is loan_facts)
    # Sanitize expression trees + verify carve-out quotes against the selected loan.
    cleaned_covs: list[CovenantSpec] = []
    for spec in loan_facts.covenants:
        spec, expr_flags = sanitize_expr_on_spec(spec)
        flags.extend(expr_flags)
        section = longest_section_around_anchor(selected_loan_doc.normalized_text, spec.clause)
        spec, co_flags = verify_carve_out_quotes(
            spec,
            selected_loan_doc.normalized_text,
            section_text=section,
        )
        flags.extend(co_flags)
        cleaned_covs.append(spec)
    loan_facts = loan_facts.model_copy(update={"covenants": cleaned_covs})
    loan_facts, clause_flags_ = _normalize_clause_keys(
        loan_facts, template_clauses, document_text=selected_loan_doc.normalized_text if selected_loan_doc else None
    )
    flags.extend(clause_flags_)
    flags.extend(
        _report_clause_key_mismatch(
            sink,
            scenario_id=scenario_id,
            template_clauses=template_clauses,
            loan_facts=loan_facts,
            loan_doc_ids=[d.doc_id for d in loan_docs],
        )
    )
    citations["loan_agreement"] = loan_facts.account_id

    kyc_facts: KYCFacts | None = None
    if kyc_facts_list:
        kyc_facts, kyc_flags = _select_kyc(kyc_docs, loan_facts.account_id, kyc_facts_list)
        flags.extend(kyc_flags)
        for facts, enrich_flags in zip(kyc_facts_list, kyc_enrich_flags):
            if facts is kyc_facts:
                flags.extend(enrich_flags)
                break

    (
        auditor_facts,
        reclassifications,
        off_ledger_additions,
        auditor_doc_id,
        merge_flags,
        note_named_txn_ids,
    ) = _merge_auditor_facts(auditor_docs, auditor_facts_list)
    flags.extend(merge_flags)
    still_bad = [
        r
        for r in reclassifications
        if r.action == "reclassify"
        and (r.status or "applied") == "applied"
        and not r.to_category
        and not infer_to_category_from_quote(r.quote)
    ]
    if still_bad:
        flags.append(f"RECLASS_TO_CATEGORY_MISSING n={len(still_bad)}")
        sink.alert(
            "RECLASS_TO_CATEGORY_MISSING",
            f"{len(still_bad)} applied reclassify row(s) lack to_category after extract/repair",
            scenario_id=scenario_id,
            quote_preview=(still_bad[0].quote or "")[:160],
        )

    fx_rate: float | None = None
    fx_quote = ""
    fx_attempts: list[dict] = []

    # FX order: each auditor → selected loan → other loans (facts already extracted).
    fx_sources: list[tuple[str, str, AuditorNotesFacts | LoanAgreementFacts]] = []
    for facts, doc in zip(auditor_facts_list, auditor_docs):
        fx_sources.append(("auditor_notes", doc.doc_id, facts))
    fx_sources.append(("loan_agreement", selected_loan_doc.doc_id, loan_facts))
    for facts, doc in zip(loan_facts_list, loan_docs):
        if doc.doc_id == selected_loan_doc.doc_id:
            continue
        fx_sources.append(("loan_agreement", doc.doc_id, facts))

    for source, doc_id, facts in fx_sources:
        if fx_rate is not None:
            break
        rate, quote, attempt, fx_resolve_flags = _resolve_embedded_fx(
            source=source,
            doc_id=doc_id,
            eur_usd_rate=facts.eur_usd_rate,
            eur_amount=facts.eur_amount,
            usd_amount=facts.usd_amount,
            fx_quote=facts.fx_quote,
        )
        fx_attempts.append(attempt)
        flags.extend(fx_resolve_flags)
        if rate is not None:
            fx_rate = rate
            fx_quote = quote
        for f in fx_resolve_flags:
            if "conflict" in f.lower():
                sink.alert(
                    "FX_CHANNEL_CONFLICT",
                    f,
                    scenario_id=scenario_id,
                    quote_preview=(facts.fx_quote or "")[:120],
                )

    if self_consistency and loan_docs:
        alt = await extract_loan_agreement(
            llm,
            loan_docs[0],
            cache_dir,
            use_cache=False,
            scenario_id=scenario_id,
            account_id=loan_facts.account_id or "",
            template_clauses=template_clauses,
        )
        if [c.threshold for c in alt.covenants] != [c.threshold for c in loan_facts.covenants]:
            flags.append("self-consistency: loan covenant thresholds diverged across paraphrase pass")

    sink.stage_end("S3_extract", f"{scenario_id}: {loan_facts.borrower_name}")

    return _CaseExtract(
        scenario_id=scenario_id,
        status="ready",
        flags=flags,
        citations=citations,
        by_type=by_type,
        loan_docs=loan_docs,
        loan_facts_list=loan_facts_list,
        loan_facts=loan_facts,
        selected_loan_doc=selected_loan_doc,
        kyc_docs=kyc_docs,
        kyc_facts_list=kyc_facts_list,
        kyc_facts=kyc_facts,
        auditor_docs=auditor_docs,
        auditor_doc_id=auditor_doc_id,
        auditor_facts=auditor_facts,
        reclassifications=reclassifications,
        off_ledger_additions=off_ledger_additions,
        note_named_txn_ids=list(note_named_txn_ids),
        own_fx_rate=fx_rate,
        fx_quote=fx_quote,
        fx_attempts=fx_attempts,
        template_clauses=template_clauses,
    )


async def _finish_case(
    ext: _CaseExtract,
    linkages: list[DocumentLinkage],
    docs_by_id: dict[str, IngestedDocument],
    all_docs: list[IngestedDocument],
    ledger_df,
    llm: LLMClient,
    cache_dir: Path,
    use_cache: bool,
    journal: list[str],
    sink: ProgressSink,
    *,
    fallback_fx_rate: float | None = None,
    dev: bool = False,
    signal_index: dict | None = None,
    self_consistency: bool = False,
    engine: str = "shadow",
    engine_diff_counter: list | None = None,
) -> tuple[dict[str, dict], dict, dict]:
    """Categorize + evaluate using extract bundle; apply per-case FX then global fallback."""
    if ext.status != "ready" or ext.loan_facts is None or ext.selected_loan_doc is None:
        return {}, ext.report_entry or {}, ext.case_trace or {}

    scenario_id = ext.scenario_id
    flags = list(ext.flags)
    citations = dict(ext.citations)
    loan_facts = ext.loan_facts
    selected_loan_doc = ext.selected_loan_doc
    loan_facts_list = ext.loan_facts_list
    loan_docs = ext.loan_docs
    kyc_facts = ext.kyc_facts
    kyc_facts_list = ext.kyc_facts_list
    auditor_facts = ext.auditor_facts
    auditor_doc_id = ext.auditor_doc_id
    reclassifications = list(ext.reclassifications)
    off_ledger_additions = list(ext.off_ledger_additions)
    note_named_txn_ids = list(ext.note_named_txn_ids)
    fx_attempts = list(ext.fx_attempts)
    fx_rate = ext.own_fx_rate
    fx_quote = ext.fx_quote
    template_clauses = ext.template_clauses
    by_type = ext.by_type

    if fx_rate is None:
        flags.append("FX rate missing from borrower documents (no cross-borrower fallback)")

    account_ids = {loan_facts.account_id} if loan_facts.account_id else set()
    if not loan_facts.account_id:
        flags.append("loan agreement missing account_id; ledger filter empty")
    scen_ledger = scenario_ledger(ledger_df, account_ids)
    # A3: drop off-ledger amounts that already appear on the scenario ledger (±0.02).
    off_ledger_additions, off_ledger_flags = normalize_off_ledger_additions(
        off_ledger_additions, scen_ledger
    )
    flags.extend(off_ledger_flags)
    if scen_ledger.empty:
        flags.append(f"no ledger rows for account {loan_facts.account_id}")

    scen_ledger, fill_flags = fill_missing_amounts_from_reclassifications(scen_ledger, reclassifications)
    flags.extend(fill_flags)

    scen_ledger, fx_flags = apply_fx(scen_ledger, fx_rate)
    flags.extend(fx_flags)
    if fx_rate is not None:
        citations["fx_rate"] = f"{fx_rate}:{fx_quote[:80]}"

    sink.stage_start("S4_engine", f"{scenario_id}: categorize + evaluate")
    sink.log(f"[S4] categorize_transactions ({scenario_id}, {len(scen_ledger)} rows)")
    categories: dict[str, str] = {}
    relevance: dict[str, bool] = {}
    cat_flags: list[str] = []
    categorize_disagree: set[str] = set()
    if not scen_ledger.empty:
        categories, relevance, cat_flags = await categorize_transactions(
            llm,
            scen_ledger,
            cache_dir,
            use_cache=use_cache,
            scenario_id=scenario_id,
            account_id=loan_facts.account_id or "",
            signal_index=signal_index,
            self_consistency=self_consistency,
        )
        for f in cat_flags:
            if f.startswith("categorize_disagree_ids:"):
                raw = f.split(":", 1)[1].strip()
                if raw:
                    categorize_disagree = {x for x in raw.split(",") if x}
        categories, repair_flags = repair_degenerate_categories(scen_ledger, categories)
        cat_flags.extend(repair_flags)
        categories, relevance, rel_flags = apply_relevance_and_retainer_filters(
            scen_ledger, categories, relevance, signal_index=signal_index
        )
        cat_flags.extend(rel_flags)
        categories, filled_cat_flags = recategorize_filled_amount_rows(
            scen_ledger, categories, reclassifications
        )
        cat_flags.extend(filled_cat_flags)
        categories, formula_flags = ensure_formula_categories_present(
            scen_ledger, categories, list(loan_facts.covenants)
        )
        cat_flags.extend(formula_flags)
        for rf in repair_flags + rel_flags + filled_cat_flags + formula_flags:
            sink.log(f"[S4] {rf}")
    flags.extend(cat_flags)

    # D5: token-overlap related-party candidates → LLM confirm against KYC text (never auto-include).
    if kyc_facts is not None and not scen_ledger.empty:
        known = [p.counterparty_name for p in kyc_facts.counterparties]
        cps = sorted({str(c) for c in scen_ledger["counterparty"].astype(str)})
        candidates = related_party_token_overlap_candidates(
            loan_facts.borrower_name, cps, set(known)
        )
        if candidates:
            dossier_parts = [d.normalized_text for d in (ext.kyc_docs or [])]
            if selected_loan_doc is not None:
                dossier_parts.append(selected_loan_doc.normalized_text)
            dossier_text = "\n\n".join(dossier_parts)
            confirmed = await confirm_related_party_overlap(
                llm,
                dossier_text=dossier_text,
                borrower_name=loan_facts.borrower_name,
                candidates=candidates,
                cache_dir=cache_dir,
                use_cache=use_cache,
                scenario_id=scenario_id,
                account_id=loan_facts.account_id or "",
            )
            if confirmed:
                kyc_facts = kyc_facts.model_copy(
                    update={"counterparties": list(kyc_facts.counterparties) + confirmed}
                )
                flags.append(
                    f"related_party overlap confirmed {len(confirmed)}/{len(candidates)}: "
                    + ", ".join(p.counterparty_name for p in confirmed)
                )
            else:
                flags.append(
                    f"related_party overlap candidates not confirmed ({len(candidates)}): "
                    + ", ".join(candidates[:8])
                )

    ledger_before_reclass = scen_ledger.copy() if dev else None

    applied_keys: list[str] = []
    unmatched_indices: list[int] = []
    include_recs = [r for r in reclassifications if r.action == "include_in_period"]
    other_recs = [r for r in reclassifications if r.action != "include_in_period"]
    if other_recs:
        categories, scen_ledger, reclass_flags, applied_keys, unmatched_indices = apply_reclassifications(
            categories, scen_ledger, other_recs
        )
        flags.extend(reclass_flags)
    if not scen_ledger.empty:
        categories, repair_flags2 = repair_degenerate_categories(scen_ledger, categories)
        flags.extend(repair_flags2)
        for rf in repair_flags2:
            sink.log(f"[S4] post-reclass {rf}")
        # W3.2: description priors again after reclass to_category may have overwritten.
        categories, relevance, post_rel_flags = apply_relevance_and_retainer_filters(
            scen_ledger, categories, relevance, signal_index=signal_index
        )
        flags.extend(f"post-reclass {f}" for f in post_rel_flags if f.startswith("payroll_") or f.startswith("retainer_"))
        categories, pledge_flags, pledge_remapped = apply_pledge_based_transfer_categories(
            scen_ledger, categories, kyc_facts
        )
        flags.extend(pledge_flags)
        for pf in pledge_flags:
            sink.log(f"[S4] {pf}")
    else:
        pledge_remapped = []
    if unmatched_indices and other_recs:
        other_pos = [i for i, r in enumerate(reclassifications) if r.action != "include_in_period"]
        unmatched_indices = [other_pos[j] for j in unmatched_indices if j < len(other_pos)]

    # W3.7 / W5.2 / W5.4: mechanism = applied reclassify/exclude + Unrestricted
    # pledge remaps. include_in_period and null-to_category fills stay out.
    # Note-named TXNs under reclass-denial are NOT auto-mechanisms (draft traps
    # like utilities→taxes would otherwise pollute every BREACH cell).
    mechanism_from_reclass = sorted(
        {
            *(k for k in applied_keys if k and not str(k).startswith("idx:")),
            *pledge_remapped,
        }
    )
    if note_named_txn_ids:
        flags.append(
            f"auditor note-named txns (evidence triage, not auto-mechanism): "
            + ", ".join(note_named_txn_ids[:12])
        )
    group_capex_value: float | None = None
    group_capex_meta: dict | None = None
    # Demote false group_capex labels (Unrestricted-transfer clauses mis-typed by LLM).
    demoted = []
    demote_changed = False
    for c in loan_facts.covenants:
        q = f"{c.quote or ''} {c.formula_description or ''}"
        if c.covenant_type == "ratio_group_capex_to_ebitda" and not looks_like_group_capex_ebitda_ratio(q):
            demoted.append(refine_covenant_spec(c))
            demote_changed = True
        else:
            demoted.append(c)
    if demote_changed:
        loan_facts = loan_facts.model_copy(update={"covenants": demoted})
        flags.append("group_capex_type_demoted_via_quote")

    needs_group = any(
        c.covenant_type == "ratio_group_capex_to_ebitda" or getattr(c, "pct_of_group_capex", False)
        for c in loan_facts.covenants
    )
    if not needs_group:
        # Quote may still demand Group even if type refine missed (second line of defence).
        for c in loan_facts.covenants:
            q = f"{c.quote or ''} {c.formula_description or ''}"
            if looks_like_group_capex_ebitda_ratio(q):
                needs_group = True
                break
    if needs_group:
        parent_doc = find_group_capex_document(all_docs, loan_facts.borrower_name)
        if parent_doc:
            sink.log(f"[S3] extract group_capex ({parent_doc.doc_id})")
            sink.stage_start("S3_extract", f"{scenario_id}: group capex")
            capex_facts = await extract_group_capex(
                llm,
                parent_doc,
                cache_dir,
                use_cache=use_cache,
                scenario_id=scenario_id,
                account_id=loan_facts.account_id or "",
            )
            group_capex_value, capex_flags = compute_group_capex(capex_facts)
            flags.extend(capex_flags)
            for f in capex_flags:
                if "mismatch" in f.lower():
                    sink.alert(
                        "CAPEX_COMPONENT_MISMATCH",
                        f,
                        scenario_id=scenario_id,
                        quote_preview=(capex_facts.quote or "")[:120],
                    )
            if group_capex_value is None:
                sink.alert(
                    "CAPEX_COMPONENTS_MISSING",
                    "GroupCapexFacts returned no usable components; will try borrower own-capex fallback",
                    scenario_id=scenario_id,
                    quote_preview=(capex_facts.derivation or capex_facts.quote or "")[:160],
                )
            group_capex_meta = {
                "value": group_capex_value,
                "doc_id": parent_doc.doc_id,
                "derivation": capex_facts.derivation,
                "opening_nbv": capex_facts.opening_nbv,
                "closing_nbv": capex_facts.closing_nbv,
                "depreciation": capex_facts.depreciation,
                "disposals": capex_facts.disposals,
                "direct_capex_line": capex_facts.direct_capex_line,
            }
            citations["group_capex_doc"] = parent_doc.doc_id
            sink.stage_end("S3_extract", f"capex={group_capex_value}")
            # Ensure type is group so we never silently evaluate as plain capex/EBITDA.
            forced = []
            changed = False
            for c in loan_facts.covenants:
                q = f"{c.quote or ''} {c.formula_description or ''}".casefold()
                if c.covenant_type != "ratio_group_capex_to_ebitda" and looks_like_group_capex_ebitda_ratio(q):
                    forced.append(
                        c.model_copy(
                            update={
                                "covenant_type": "ratio_group_capex_to_ebitda",
                                "numerator_categories": [],
                                "denominator_categories": [],
                            }
                        )
                    )
                    changed = True
                else:
                    forced.append(c)
            if changed:
                loan_facts = loan_facts.model_copy(update={"covenants": forced})
        else:
            flags.append("GROUP_CAPEX_MISSING: no unique parent consolidated capex document")
            sink.alert(
                "GROUP_CAPEX_MISSING",
                "Group capex parent not found — using borrower own-capex fallback (never threshold stub)",
                scenario_id=scenario_id,
            )
            forced = []
            for c in loan_facts.covenants:
                q = f"{c.quote or ''} {c.formula_description or ''}".casefold()
                if c.covenant_type != "ratio_group_capex_to_ebitda" and looks_like_group_capex_ebitda_ratio(q):
                    forced.append(
                        c.model_copy(
                            update={
                                "covenant_type": "ratio_group_capex_to_ebitda",
                                "numerator_categories": [],
                                "denominator_categories": [],
                            }
                        )
                    )
                else:
                    forced.append(c)
            loan_facts = loan_facts.model_copy(update={"covenants": forced})

        # Loud own-capex fallback when Group numerator is still missing (V4 §1 / 1A).
        if group_capex_value is None and not scen_ledger.empty:
            period_src = scen_ledger
            for c in loan_facts.covenants:
                if c.covenant_type == "ratio_group_capex_to_ebitda":
                    period_src = filter_period(scen_ledger, c.period_start, c.period_end)
                    break
            own_val, own_ids = borrower_own_capex(period_src, categories)
            group_capex_value = own_val
            flags.append(
                f"GROUP_CAPEX_OWN_FALLBACK: borrower capex={own_val} txns={','.join(own_ids[:8])}"
            )
            sink.alert(
                "GROUP_CAPEX_OWN_FALLBACK",
                f"Using borrower ledger capex={own_val} as Group numerator — parent FS missing or empty",
                scenario_id=scenario_id,
            )
            if group_capex_meta is None:
                group_capex_meta = {"value": own_val, "doc_id": None, "fallback": "own_capex"}
            else:
                group_capex_meta = {**group_capex_meta, "value": own_val, "fallback": "own_capex"}

    cells: dict[str, dict] = {}
    cell_traces: dict[str, dict] = {}
    sink.stage_end("S4_engine", f"{scenario_id}: {len(loan_facts.covenants)} covenants")
    sink.stage_start("S5_evidence", f"{scenario_id}: flip-test")
    # W3.7: only applied reclass txn_ids (not amount-fill scrapes).
    doc_named_txns = list(mechanism_from_reclass)
    for spec in loan_facts.covenants:
        sink.log(f"[S5] evaluate+flip {scenario_id}/{spec.clause} ({spec.covenant_type})")
        # B2: per-clause copy so include_in_period category writes do not leak across clauses.
        clause_cats = dict(categories)
        period_df = filter_period(scen_ledger, spec.period_start, spec.period_end)
        if include_recs:
            clause_cats, period_df, inc_flags, _inc_keys, _inc_unmatched = apply_reclassifications(
                clause_cats, period_df, include_recs, full_df=scen_ledger
            )
            flags.extend(f"[{spec.clause}] {f}" for f in inc_flags)
            # include_* stay local to this clause — do not accumulate into applied_keys /
            # unmatched_indices (those feed auditor reporting for reclassify actions only).
        period_df, agg_flags = prepare_aggregation_frame(spec, period_df, full_df=scen_ledger)
        flags.extend(f"[{spec.clause}] {f}" for f in agg_flags)
        # W5.4: include_in_period is data hygiene, not an evidence mechanism.
        mech_ids = sorted({*doc_named_txns})
        try:
            cell, clause_flags, cell_trace = _evaluate_and_flip(
                spec,
                period_df,
                clause_cats,
                kyc_facts,
                group_capex_value,
                mech_ids,
                sink=sink,
                scenario_id=scenario_id,
                off_ledger=off_ledger_additions,
                categorize_disagree=categorize_disagree,
                materiality_threshold_usd=getattr(auditor_facts, "materiality_threshold_usd", None)
                if auditor_facts
                else None,
                engine=engine,
                engine_diff_counter=engine_diff_counter,
                relevance=relevance,
                mechanism_txn_ids=mech_ids,
            )
        except Exception as exc:  # B5: one clause failure must not stub the whole scenario
            flags.append(f"[{spec.clause}] evaluate_failed: {type(exc).__name__}: {exc}")
            cell = _stub_cell()
            clause_flags = [f"evaluate_failed: {type(exc).__name__}"]
            cell_trace = {"stub_reason": "evaluate_failed", "error": str(exc)}
        cells[spec.clause] = cell
        cell_traces[spec.clause] = cell_trace
        flags.extend(f"[{spec.clause}] {f}" for f in clause_flags)
    sink.stage_end("S5_evidence", f"{scenario_id}: {len(cells)} cells")

    for clause in template_clauses:
        if clause not in cells:
            stub = _stub_cell()
            cells[clause] = stub
            cell_traces[clause] = build_cell_trace(
                stub, stub_reason="clause not computed; stubbed"
            )
            flags.append(f"clause {clause} not computed; stubbed")

    journal.append(
        f"{scenario_id}: {loan_facts.borrower_name} ({loan_facts.account_id}) — "
        f"{len([c for c in cell_traces.values() if c.get('outcome') == 'computed'])} clauses computed"
    )
    journal.append(f"decision detail: trace.json → cases.{scenario_id}")
    sink.scenario_end(scenario_id, f"{len(cells)} clauses")

    case_trace = {
        "scenario_id": scenario_id,
        "status": "processed",
        "binding": build_binding(scenario_id, linkages, by_type, loan_facts),
        "loan": build_loan_section(loan_facts, template_clauses),
        "auditor": build_auditor_section(auditor_doc_id, reclassifications, applied_keys, unmatched_indices),
        "categorization": category_summary(scen_ledger, categories),
        "fx_rate": fx_rate,
        "group_capex": group_capex_meta,
        "cells": {c: cell_traces[c] for c in sorted(cell_traces)},
        "flags": list(flags),
    }

    if dev:
        excluded_ids: set[str] = set()
        if ledger_before_reclass is not None and not ledger_before_reclass.empty:
            excluded_ids = set(map(str, ledger_before_reclass["txn_id"])) - set(map(str, scen_ledger["txn_id"]))
        case_trace["dev"] = {
            "documents": _dev_docs(scenario_id, linkages, docs_by_id),
            "extracts": {
                "loan_agreements_all": [
                    {
                        **f.model_dump(),
                        "doc_id": d.doc_id,
                        "path": d.path,
                        "selected": d.doc_id == selected_loan_doc.doc_id,
                    }
                    for f, d in zip(loan_facts_list, loan_docs)
                ],
                "loan_selected": {
                    **loan_facts.model_dump(),
                    "doc_id": selected_loan_doc.doc_id,
                    "path": selected_loan_doc.path,
                },
                "kyc_all": [f.model_dump() for f in kyc_facts_list],
                "kyc_selected": kyc_facts.model_dump() if kyc_facts else None,
                "auditor": auditor_facts.model_dump() if auditor_facts else None,
                "fx_attempts": fx_attempts,
                "group_capex": group_capex_meta,
            },
            "transactions": _dev_transactions(
                ledger_before_reclass, {str(k): v for k, v in categories.items()}, excluded_ids
            ),
        }

    return (
        cells,
        {"scenario_id": scenario_id, "status": "processed", "flags": flags, "citations": citations},
        case_trace,
    )


def _evaluate_and_flip(
    spec: CovenantSpec,
    period_df,
    categories: dict[str, str],
    kyc_facts: KYCFacts | None,
    group_capex_value: float | None,
    doc_named_txn_ids: list[str],
    *,
    sink: ProgressSink | None = None,
    scenario_id: str = "",
    off_ledger: list | None = None,
    categorize_disagree: set[str] | None = None,
    materiality_threshold_usd: float | None = None,
    engine: str = "shadow",
    engine_diff_counter: list | None = None,
    relevance: dict[str, bool] | None = None,
    mechanism_txn_ids: list[str] | None = None,
) -> tuple[dict, list[str], dict]:
    """Enum path is submission default; tree/shadow compare via expression engine.

    Carve-outs apply before the type dispatcher (and before tree eval).
    Known limitation: COMPLIANT-via-carve-out does not feed evidence (BREACH flip only).
    """
    flags: list[str] = []
    stub = _stub_cell(spec)
    off_ledger = off_ledger or []
    categorize_disagree = categorize_disagree or set()
    materiality = materiality_threshold_usd
    engine = engine if engine in {"enum", "tree", "shadow"} else "shadow"
    relevance = relevance or {}
    mechanism_txn_ids = list(mechanism_txn_ids or doc_named_txn_ids or [])

    # V4 §4: RP quote must evaluate via KYC mechanism — re-refine before taxonomy gate.
    qlow_pre = f"{spec.quote or ''} {spec.formula_description or ''}".casefold()
    if any(
        tok in qlow_pre
        for tok in ("связанн", "аффилирован", "related-party", "related party", "аффилированн")
    ) and spec.covenant_type != "sum_related_party_payments":
        spec = refine_covenant_spec(spec)
        flags.append("rp_re_refined_before_taxonomy_gate")

    tax_flags = validate_spec_against_taxonomy(spec, categories)
    incomplete = spec_incomplete(spec)
    if incomplete:
        # Soft flag now; critical alert only if tree/enum cannot repair (W3.8).
        flags.extend(incomplete)
    if tax_flags:
        # R2: if quote names EBITDA and synthesize can build a tree, continue (tree/enum repair)
        # instead of soft-COMPLIANT stub — D4 still flagged loudly.
        qlow = f"{spec.quote or ''} {spec.formula_description or ''}".casefold()
        synth_probe = synthesize_expr_from_enum(spec)
        ebitda_repair = "ebitda" in qlow and synth_probe is not None
        desc_bucket_repair = bool(getattr(spec, "description_bucket", None))
        rp_repair = (
            spec.covenant_type == "sum_related_party_payments"
            or any(
                tok in qlow
                for tok in ("связанн", "аффилирован", "related-party", "related party")
            )
        )
        # RP ratio clauses (threshold < 1) have no synth tree — still must not stub on ['other'].
        if ebitda_repair or rp_repair or desc_bucket_repair:
            flags.extend(tax_flags)
            flags.append("taxonomy_repaired_via_ebitda_expr" if ebitda_repair else "taxonomy_repaired_via_rp")
            if sink is not None:
                sink.alert(
                    "SPEC_TAXONOMY_REPAIRED",
                    tax_flags[0],
                    scenario_id=scenario_id,
                    clause=spec.clause,
                )
            if rp_repair and spec.covenant_type != "sum_related_party_payments":
                spec = refine_covenant_spec(spec)
        else:
            reason = tax_flags[0]
            if sink is not None:
                sink.alert(
                    "SPEC_TAXONOMY_INVALID",
                    reason,
                    scenario_id=scenario_id,
                    clause=spec.clause,
                )
            cell = make_stub_cell(
                threshold=float(spec.threshold) if spec.threshold is not None else None
            )
            return cell, tax_flags, build_cell_trace(cell, spec=spec, stub_reason=reason, clause_flags=tax_flags)

    # R1: covenant aggregates use signal rows only (related-party retainers stay if relevant=true).
    # Description-bucket covenants (marketing/consulting) keep the full frame — those rows
    # are forced to other + relevance=false by design.
    period_full = period_df
    if getattr(spec, "description_bucket", None):
        rel_flags = ["relevance_bypass_description_bucket"]
        period_df = period_df
    else:
        period_df, rel_flags = filter_df_by_relevance(period_df, relevance)
    flags.extend(rel_flags)

    period_raw = period_df
    period_df, excluded_ids, carve_flags = apply_carve_outs(spec, period_df, categories)
    flags.extend(carve_flags)
    excluded_abs = 0.0
    if excluded_ids and not period_raw.empty:
        col = "amount_usd" if "amount_usd" in period_raw.columns else "amount"
        excluded_abs = float(
            period_raw[period_raw["txn_id"].astype(str).isin(excluded_ids)][col].abs().sum()
        )

    qlow_all = f"{spec.quote or ''} {spec.formula_description or ''}".casefold()
    adj_ebitda = (
        "скорректирован" in qlow_all
        or "adjusted ebitda" in qlow_all
        or "adj. ebitda" in qlow_all
        or "adj ebitda" in qlow_all
    )
    if adj_ebitda and materiality is None:
        # Soft triage only — missing materiality is normal when the loan does not define it.
        flags.append(
            "MATERIALITY_UNSET: Adj.EBITDA covenant; no materiality_threshold_usd "
            "(add-backs=0; normal if contract omits a floor)"
        )
    expr_ctx = ExprContext(
        group_capex=group_capex_value,
        related_parties=kyc_facts.counterparties if kyc_facts else None,
        related_threshold_pct=kyc_facts.related_party_threshold_pct if kyc_facts else None,
        off_ledger=off_ledger,
        period_end=spec.period_end,
        materiality_threshold_usd=materiality,
        adj_ebitda=adj_ebitda,
    )

    def _springing_trigger(frame, cats: dict[str, str]) -> float | None:
        mode = getattr(spec, "springing_mode", None)
        if mode in {"own_capex_above", "trigger_above"}:
            own, _ = borrower_own_capex(frame, cats)
            return own
        if mode == "leverage_above":
            from agent.schemas import EBITDA_OPEX_CATEGORIES
            from agent.s4_engine import _sum_cats

            rev_raw, _ = _sum_cats(frame, cats, ["revenue"])
            opex_raw, _ = _sum_cats(frame, cats, list(EBITDA_OPEX_CATEGORIES))
            ebitda = float(rev_raw) - abs(float(opex_raw))
            fin_raw, _ = _sum_cats(frame, cats, ["financing_proceeds"])
            if ebitda <= 0:
                return abs(float(fin_raw))  # undefined → treat as high leverage
            return abs(float(fin_raw)) / abs(float(ebitda))
        if mode == "min_quarter_revenue_below":
            def _rev_metric(qdf):
                from agent.s4_engine import _sum_cats

                rev, ids = _sum_cats(qdf, cats, ["revenue"])
                return float(rev), ids, []

            val, _ids, _flags = reduce_over_calendar_quarters(
                frame, spec.period_start, spec.period_end, "min", _rev_metric
            )
            return val
        return None

    def _enum_eval(frame):
        qred = getattr(spec, "quarter_reduce", None)
        bucket = getattr(spec, "description_bucket", None)

        def _with_quarters(base_eval, frm):
            """Wrap a (frame)->CovenantResult evaluator with calendar-quarter reduce."""
            if not qred:
                return base_eval(frm)

            def metric2(qdf):
                r = base_eval(qdf)
                raw = r.inputs.get("value_raw")
                if raw is None:
                    raw = r.inputs.get("total", r.actual)
                return float(raw), list(r.contributing_txn_ids), list(r.flags)

            val, ids, qflags = reduce_over_calendar_quarters(
                frm, spec.period_start, spec.period_end, qred, metric2
            )
            status = "COMPLIANT" if (
                (spec.comparator == "max" and val <= spec.threshold)
                or (spec.comparator == "min" and val >= spec.threshold)
            ) else "BREACH"
            sample = base_eval(frm)
            sample.status = status
            sample.actual = round(abs(val), 2) if val != 0 else 0.01
            sample.contributing_txn_ids = ids
            sample.flags = list(sample.flags) + qflags
            sample.inputs = {**sample.inputs, "value_raw": val, "total": abs(val), "numerator": abs(val)}
            return sample

        if bucket and spec.covenant_type in {"sum_category", "max_category", "unknown"}:
            def _bucket_eval(frm):
                return evaluate_description_bucket(spec, frm, bucket)

            res = _with_quarters(_bucket_eval, frame) if qred else _bucket_eval(frame)

            def recompute(subset, cats):
                # cats unused for description buckets
                r = (
                    _with_quarters(lambda f: evaluate_description_bucket(spec, f, bucket), subset)
                    if qred
                    else evaluate_description_bucket(spec, subset, bucket)
                )
                return apply_springing(spec, r, trigger_value=_springing_trigger(subset, cats)).status

            return res, None, recompute

        if bucket and spec.covenant_type == "ratio_generic":
            # consulting / EBITDA: numerator from description bucket
            def _ratio_bucket_eval(frm, cats):
                from agent.s4_engine import CovenantResult, verdict as _verdict
                from agent.schemas import EBITDA_OPEX_CATEGORIES
                from agent.s4_engine import _sum_cats, _currency_mix

                num, ids = sum_description_bucket(
                    frm, bucket, f"{spec.quote or ''} {spec.formula_description or ''}"
                )
                rev_raw, rev_ids = _sum_cats(frm, cats, ["revenue"])
                opex_raw, opex_ids = _sum_cats(frm, cats, list(EBITDA_OPEX_CATEGORIES))
                ebitda = float(rev_raw) - abs(float(opex_raw))
                flags_l = [f"description_bucket_num={bucket}"]
                if ebitda <= 0:
                    flags_l.append("EBITDA <= 0 — ratio undefined")
                    return CovenantResult(
                        clause=spec.clause,
                        status="BREACH" if spec.comparator == "max" else "COMPLIANT",
                        actual=round(num, 2) if num > 0 else 0.01,
                        contributing_txn_ids=ids + rev_ids + opex_ids,
                        flags=flags_l,
                        inputs={"numerator": num, "ebitda": ebitda, "value_raw": num, "total": num},
                    )
                ratio = num / ebitda
                return CovenantResult(
                    clause=spec.clause,
                    status=_verdict(ratio, spec.comparator, spec.threshold),
                    actual=round(abs(ratio), 2),
                    contributing_txn_ids=ids + rev_ids + opex_ids,
                    flags=flags_l,
                    inputs={
                        "numerator": num,
                        "ebitda": ebitda,
                        "value_raw": ratio,
                        "total": num,
                        "denominator": ebitda,
                    },
                )

            res = _ratio_bucket_eval(frame, categories)

            def recompute(subset, cats):
                return apply_springing(
                    spec, _ratio_bucket_eval(subset, cats), trigger_value=_springing_trigger(subset, cats)
                ).status

            return res, None, recompute

        if spec.covenant_type == "ratio_group_capex_to_ebitda":
            if group_capex_value is None:
                return None, "missing group capex; stubbed", None
            if is_absolute_group_capex_ceiling(spec):
                own_val, _own_ids = borrower_own_capex(frame, categories)
                res = evaluate_sum_group_capex(
                    spec, frame, categories, group_capex_value, own_capex=own_val
                )

                def recompute(subset, cats):
                    ov, _ = borrower_own_capex(subset, cats)
                    return apply_springing(
                        spec,
                        evaluate_sum_group_capex(
                            spec, subset, cats, group_capex_value, own_capex=ov
                        ),
                        trigger_value=_springing_trigger(subset, cats),
                    ).status

                return res, None, recompute
            res = evaluate_ratio_group_capex_to_ebitda(spec, frame, categories, group_capex_value)

            def recompute(subset, cats):
                return apply_springing(
                    spec,
                    evaluate_ratio_group_capex_to_ebitda(spec, subset, cats, group_capex_value),
                    trigger_value=_springing_trigger(subset, cats),
                ).status

            return res, None, recompute
        if spec.covenant_type == "sum_revenue":
            def _rev(frm):
                return evaluate_sum_revenue(spec, frm, categories)

            res = _with_quarters(_rev, frame) if qred else _rev(frame)

            def recompute(subset, cats):
                r = (
                    _with_quarters(lambda f: evaluate_sum_revenue(spec, f, cats), subset)
                    if qred
                    else evaluate_sum_revenue(spec, subset, cats)
                )
                return apply_springing(spec, r, trigger_value=_springing_trigger(subset, cats)).status

            return res, None, recompute
        if spec.covenant_type == "sum_related_party_payments":
            if kyc_facts is None:
                return None, "related-party covenant missing KYC dossier; stubbed", None
            # threshold_pct may be None — dossier-labelled affiliates still evaluate.
            res = evaluate_sum_related_party_payments(
                spec, frame, kyc_facts.counterparties, kyc_facts.related_party_threshold_pct, categories
            )

            def recompute(subset, cats):
                return apply_springing(
                    spec,
                    evaluate_sum_related_party_payments(
                        spec, subset, kyc_facts.counterparties, kyc_facts.related_party_threshold_pct, cats
                    ),
                    trigger_value=_springing_trigger(subset, cats),
                ).status

            return res, None, recompute
        if spec.covenant_type == "sum_category":
            if getattr(spec, "pct_of_group_capex", False):
                if group_capex_value is None:
                    return None, "pct_of_group_capex missing group capex; stubbed", None
                res = evaluate_sum_categories_vs_group_pct(
                    spec, frame, categories, group_capex_value
                )

                def recompute(subset, cats):
                    return apply_springing(
                        spec,
                        evaluate_sum_categories_vs_group_pct(spec, subset, cats, group_capex_value),
                        trigger_value=_springing_trigger(subset, cats),
                    ).status

                return res, None, recompute

            def _sum(frm):
                return evaluate_sum_category(
                    spec, frm, categories, off_ledger=off_ledger, materiality_threshold_usd=materiality
                )

            res = _with_quarters(_sum, frame) if qred else _sum(frame)

            def recompute(subset, cats):
                r = (
                    _with_quarters(
                        lambda f: evaluate_sum_category(
                            spec, f, cats, off_ledger=off_ledger, materiality_threshold_usd=materiality
                        ),
                        subset,
                    )
                    if qred
                    else evaluate_sum_category(
                        spec, subset, cats, off_ledger=off_ledger, materiality_threshold_usd=materiality
                    )
                )
                return apply_springing(spec, r, trigger_value=_springing_trigger(subset, cats)).status

            return res, None, recompute
        if spec.covenant_type == "max_category":
            if bucket:
                def _bmax(frm):
                    # max over quarters of bucket sum — if qred=max; else single sum vs thr
                    return evaluate_description_bucket(spec, frm, bucket)

                res = _with_quarters(_bmax, frame) if qred else _bmax(frame)

                def recompute(subset, cats):
                    r = _with_quarters(lambda f: evaluate_description_bucket(spec, f, bucket), subset) if qred else evaluate_description_bucket(spec, subset, bucket)
                    return apply_springing(spec, r, trigger_value=_springing_trigger(subset, cats)).status

                return res, None, recompute
            res = evaluate_max_category(spec, frame, categories)

            def recompute(subset, cats):
                return apply_springing(
                    spec, evaluate_max_category(spec, subset, cats),
                    trigger_value=_springing_trigger(subset, cats),
                ).status

            return res, None, recompute
        if spec.covenant_type == "revenue_minus_max_category":
            res = evaluate_revenue_minus_max_category(spec, frame, categories, off_ledger=off_ledger)

            def recompute(subset, cats):
                return apply_springing(
                    spec,
                    evaluate_revenue_minus_max_category(spec, subset, cats, off_ledger=off_ledger),
                    trigger_value=_springing_trigger(subset, cats),
                ).status

            return res, None, recompute
        if spec.covenant_type == "ratio_generic":
            if getattr(spec, "ebitda_plus_categories", None):
                res = evaluate_ebitda_plus_categories(
                    spec, frame, categories, materiality_threshold_usd=materiality
                )

                def recompute(subset, cats):
                    return apply_springing(
                        spec,
                        evaluate_ebitda_plus_categories(
                            spec, subset, cats, materiality_threshold_usd=materiality
                        ),
                        trigger_value=_springing_trigger(subset, cats),
                    ).status

                return res, None, recompute

            def _rg(frm):
                return evaluate_ratio_generic(
                    spec, frm, categories, materiality_threshold_usd=materiality
                )

            # Quarterly EBITDA floor: rebuild value_raw as ebitda per quarter.
            if qred and "ebitda" in f"{spec.quote or ''} {spec.formula_description or ''}".casefold():
                from agent.s4_engine import CovenantResult, verdict as _verdict
                from agent.schemas import EBITDA_OPEX_CATEGORIES
                from agent.s4_engine import _sum_cats

                def _ebitda_metric(qdf):
                    rev_raw, rev_ids = _sum_cats(qdf, categories, ["revenue"])
                    opex_raw, opex_ids = _sum_cats(qdf, categories, list(EBITDA_OPEX_CATEGORIES))
                    ebitda = float(rev_raw) - abs(float(opex_raw))
                    return ebitda, rev_ids + opex_ids, []

                val, ids, qflags = reduce_over_calendar_quarters(
                    frame, spec.period_start, spec.period_end, qred, _ebitda_metric
                )
                status = _verdict(val, spec.comparator, spec.threshold)
                res = CovenantResult(
                    clause=spec.clause,
                    status=status,
                    actual=round(abs(val), 2) if val != 0 else 0.01,
                    contributing_txn_ids=ids,
                    flags=["quarterly_ebitda"] + qflags,
                    inputs={"value_raw": val, "total": abs(val), "numerator": abs(val), "ebitda": val},
                )

                def recompute(subset, cats):
                    def _m(qdf):
                        rev_raw, rev_ids = _sum_cats(qdf, cats, ["revenue"])
                        opex_raw, opex_ids = _sum_cats(qdf, cats, list(EBITDA_OPEX_CATEGORIES))
                        ebitda = float(rev_raw) - abs(float(opex_raw))
                        return ebitda, rev_ids + opex_ids, []

                    v, _i, _f = reduce_over_calendar_quarters(
                        subset, spec.period_start, spec.period_end, qred, _m
                    )
                    st = _verdict(v, spec.comparator, spec.threshold)
                    return apply_springing(
                        spec,
                        CovenantResult(clause=spec.clause, status=st, actual=abs(v), contributing_txn_ids=[], flags=[], inputs={"value_raw": v}),
                        trigger_value=_springing_trigger(subset, cats),
                    ).status

                return res, None, recompute

            res = _rg(frame)

            def recompute(subset, cats):
                return apply_springing(
                    spec,
                    evaluate_ratio_generic(
                        spec, subset, cats, materiality_threshold_usd=materiality
                    ),
                    trigger_value=_springing_trigger(subset, cats),
                ).status

            return res, None, recompute
        if spec.covenant_type == "unknown":
            if bucket:
                res = evaluate_description_bucket(spec, frame, bucket)

                def recompute(subset, cats):
                    return apply_springing(
                        spec, evaluate_description_bucket(spec, subset, bucket),
                        trigger_value=_springing_trigger(subset, cats),
                    ).status

                return res, None, recompute
            return None, (
                f"unknown covenant_type; formula={spec.formula_description!r}; "
                f"quote={(spec.quote or '')[:160]!r}; stubbed — update S4"
            ), None
        return None, f"unknown covenant_type {spec.covenant_type}", None

    def _tree_eval(frame, tree_spec: CovenantSpec):
        res = apply_springing(
            spec,
            evaluate_expr(tree_spec, frame, categories, expr_ctx),
            trigger_value=_springing_trigger(frame, categories),
        )

        def recompute(subset, cats):
            return apply_springing(
                spec,
                evaluate_expr(tree_spec, subset, cats, expr_ctx),
                trigger_value=_springing_trigger(subset, cats),
            ).status

        return res, recompute

    tree_spec = spec
    if not (spec.expr_nodes and spec.expr_root is not None):
        synth = synthesize_expr_from_enum(spec)
        if synth is not None:
            tree_spec = synth
            # W3.8: tree repair → drop soft SPEC_INCOMPLETE noise from cell flags.
            if incomplete:
                flags = [f for f in flags if not str(f).startswith("SPEC_INCOMPLETE")]
                incomplete = []

    enum_result, stub_reason, enum_recompute = _enum_eval(period_df)
    if stub_reason:
        if sink is not None and "unknown covenant" in stub_reason:
            sink.alert(
                "UNKNOWN_COVENANT",
                "covenant does not fit known shapes; cell stubbed — update agent S4 logic",
                scenario_id=scenario_id,
                clause=spec.clause,
                formula_description=spec.formula_description or "",
                quote_preview=(spec.quote or "")[:160],
            )
        # W3.8: critical SPEC_INCOMPLETE only when the cell actually stubs.
        if incomplete and sink is not None:
            sink.alert(
                "SPEC_INCOMPLETE",
                incomplete[0],
                scenario_id=scenario_id,
                clause=spec.clause,
                quote_preview=(spec.quote or "")[:160],
            )
        return stub, [stub_reason], build_cell_trace(stub, spec=spec, stub_reason=stub_reason, clause_flags=[stub_reason])

    assert enum_result is not None and enum_recompute is not None
    enum_result = apply_springing(
        spec, enum_result, trigger_value=_springing_trigger(period_df, categories)
    )
    # W3.8 / V4 §6: cell computed → SPEC_INCOMPLETE is soft noise only (never critical).
    if incomplete:
        flags = [f for f in flags if not str(f).startswith("SPEC_INCOMPLETE")]
        incomplete = []

    if excluded_ids:
        raw_result, raw_reason, _ = _enum_eval(period_raw)
        if raw_result is not None and not raw_reason:
            raw_result = apply_springing(spec, raw_result, trigger_value=_springing_trigger(period_raw, categories))
            flip = carve_out_status_flip_flag(
                raw_result.status, enum_result.status, excluded_ids, excluded_abs
            )
            if flip:
                flags.append(flip)

    tree_result = None
    tree_recompute = None
    if tree_spec.expr_nodes and tree_spec.expr_root is not None:
        try:
            tree_result, tree_recompute = _tree_eval(period_df, tree_spec)
        except Exception as exc:  # noqa: BLE001
            flags.append(f"EXPR_EVAL_FAILED: {type(exc).__name__}: {exc}")

    if engine == "tree" and tree_result is not None:
        result, recompute = tree_result, tree_recompute
        flags.append("engine=tree")
    else:
        result, recompute = enum_result, enum_recompute
        if engine == "shadow" and tree_result is not None:
            status_diff = tree_result.status != enum_result.status
            actual_diff = abs(tree_result.actual - enum_result.actual) > max(
                0.001 * max(abs(enum_result.actual), 1e-9), 0.001
            )
            if status_diff or actual_diff:
                flags.append(
                    f"ENGINE_DIFF status {enum_result.status}->{tree_result.status} "
                    f"actual {enum_result.actual}->{tree_result.actual}"
                )
                if engine_diff_counter is not None:
                    engine_diff_counter.append(f"{scenario_id}/{spec.clause}")
            else:
                flags.append("ENGINE_DIFF=0")

    flags.extend(result.flags)
    hit_disagree = sorted(set(result.contributing_txn_ids) & categorize_disagree)
    if hit_disagree:
        flags.append(
            f"categorize self-consistency hit formula rows: {','.join(hit_disagree[:12])}"
        )

    # R7: expand mechanism pool with retainer/advisory rows only for related-party covenants.
    mech = list(mechanism_txn_ids)
    if (
        spec.covenant_type == "sum_related_party_payments"
        and not period_df.empty
        and "description" in period_df.columns
    ):
        contrib = set(result.contributing_txn_ids)
        for _, row in period_df.iterrows():
            tid = str(row["txn_id"])
            if tid not in contrib:
                continue
            desc = str(row["description"]).casefold()
            if "retainer" in desc or "advisory" in desc:
                mech.append(tid)

    evidence, flip_flags, evidence_detail = flip_test(
        period_df,
        result.contributing_txn_ids,
        categories,
        recompute,
        result.status,
        doc_named_txn_ids=doc_named_txn_ids,
        category_universe=CATEGORIES,
        mechanism_txn_ids=mech,
    )
    flags.extend(flip_flags)

    actual = result.actual
    clamped = False
    if actual <= 0:
        flags.append(f"actual clamped for validator (was {actual})")
        actual = 0.01
        clamped = True

    denom = result.inputs.get("denominator")
    if denom is None and "denominator" not in result.inputs:
        if "ebitda" in result.inputs:
            denom = result.inputs.get("ebitda")
    den_ids = None
    if spec.covenant_type == "ratio_generic":
        den_cats = result.inputs.get("denominator_categories") or []
        if den_cats and not period_df.empty:
            wanted = set(den_cats)
            den_ids = int(
                period_df["txn_id"].map(lambda t: categories.get(str(t)) in wanted).sum()
            )
    ratio_undefined = any(
        "denominator <= 0" in f or "EBITDA <= 0" in f or "undefined" in f for f in result.flags
    )
    gate_fail, gate_flags = apply_sanity_gates(
        status=result.status,
        actual=float(actual),
        threshold=float(spec.threshold),
        period_row_count=int(len(period_df)),
        denominator=float(denom) if denom is not None else None,
        denominator_row_count=den_ids,
        clamped=clamped,
        ratio_undefined=ratio_undefined,
    )
    if gate_fail:
        cell, gf = stub_cell_after_sanity(result.status, gate_flags)
        flags.extend(gf)
        cell_trace = build_cell_trace(
            cell, spec=spec, result=result, evidence_detail=evidence_detail, clause_flags=flags
        )
        return cell, flags, cell_trace

    cell = {"status": result.status, "actual": round(actual, 2), "evidence_txn_id": evidence}
    cell_trace = build_cell_trace(
        cell, spec=spec, result=result, evidence_detail=evidence_detail, clause_flags=flags
    )
    return cell, flags, cell_trace



def _skeleton_submission(config, template: dict) -> dict:
    """Fully filled stub submission — written on any fatal path so a file always exists."""
    return {
        "team": config.team,
        "contact_email": config.contact_email,
        "model": config.model_name,
        "answers": {
            sid: {clause: _stub_cell() for clause in clauses}
            for sid, clauses in template.get("answers", {}).items()
        },
    }


async def run(config, sink: ProgressSink) -> dict:
    random.seed(config.seed)
    dataset = config.dataset_dir
    template: dict = {"answers": {}}
    scenario_ids: list[str] = []
    paths = None
    submission = {
        "team": config.team,
        "contact_email": config.contact_email,
        "model": config.model_name,
        "answers": {},
    }
    answers: dict[str, dict] = submission["answers"]
    journal: list[str] = [
        f"seed={config.seed}",
        f"profile={config.profile}",
        f"use_cache={config.use_cache}",
        f"engine={config.engine}",
    ]
    report_entries: list[dict] = []
    vision_flags: list[str] = []
    linking_warnings: list[dict] = []
    ingest_failures: list[str] = []
    account_map_warnings: list[str] = []
    sanitize_fixes: list[str] = []
    linkages: list = []
    docs_by_id: dict = {}
    all_docs: list = []
    dev_cases: dict[str, dict] = {}
    engine_diff_counter: list[str] = []
    result = None

    def _emergency_dump(skel: dict) -> None:
        """Last-resort write without validator/ledger — file must exist."""
        out = Path(config.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(skel, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)

    def _finalize_write() -> object:
        nonlocal sanitize_fixes
        sink.stage_start("S6_write", f"sanitize+validate → {config.out_path}")
        if paths is None:
            raise RuntimeError("dataset paths unavailable for finalize")
        result_local, sanitize_fixes = write_submission_atomic(
            submission, template, paths.ledger, config.out_path
        )
        return result_local

    def _write_confidence(result_obj) -> Path:
        logs_dir = resolve_logs_dir(config.out_path)
        run_meta = build_run_config(config)
        sink.tracker.logs_dir = str(logs_dir)
        sink.tracker.run = run_meta
        n_diff = len(engine_diff_counter)
        # Point operators at the logs dir that actually received this run's artifacts.
        decision_detail = f"{logs_dir.as_posix()}/trace.json → cases.<scenario_id>"
        write_confidence_report(
            {
                "journal": journal,
                "scenarios": report_entries,
                "alerts": list(sink.tracker.alerts),
                "vision_flags": sorted(vision_flags),
                "linking_warnings": linking_warnings,
                "ingest_failures": ingest_failures,
                "account_map_warnings": account_map_warnings,
                "sanitize_fixes": sanitize_fixes,
                "validation_ok": getattr(result_obj, "ok", False) if result_obj is not None else False,
                "validation_violations": (
                    [str(v) for v in result_obj.violations] if result_obj is not None else ["no_validation"]
                ),
                "seed": config.seed,
                "profile": config.profile,
                "engine": config.engine,
                "git_sha": run_meta["git_sha"],
                "engine_diff": f"ENGINE_DIFF {n_diff}",
                "engine_diff_cells": sorted(engine_diff_counter),
                "decision_detail": decision_detail,
                "submission": submission,
                "template": template,
                "case_traces": dict(sink.tracker.cases),
            },
            logs_dir,
        )
        write_run_config(run_meta, logs_dir)
        return logs_dir

    try:
        # B3: dataset discovery inside try so BOM/KeyError still yields a file.
        paths = discover_dataset(dataset)
        template = json.loads(paths.template.read_text(encoding="utf-8"))
        scenario_ids = list(template["answers"].keys())
        sink.set_scenarios_total(len(scenario_ids) if not config.only_scenario else 1)
        submission = _skeleton_submission(config, template)
        answers = submission["answers"]
        journal.extend(
            [
                f"dataset_ledger={paths.ledger.name}",
                f"dataset_template={paths.template.name}",
                f"dataset_documents={paths.documents.name}",
            ]
        )

        sink.stage_start("S1_ingest", f"parsing {paths.documents}")
        all_docs, ingest_failures = ingest_directory(
            paths.documents,
            cache_dir=config.cache_dir,
            vision_char_threshold=config.vision_char_threshold,
            use_cache=config.use_cache,
        )
        if ingest_failures:
            journal.append(f"ingest_failures={len(ingest_failures)}")
            for msg in ingest_failures[:20]:
                sink.log(f"[S1] fail {msg}")
        sink.stage_end("S1_ingest", f"{len(all_docs)} documents")
        sink.log(f"[S1] ingested {len(all_docs)} docs")

        llm = LLMClient(config, sink=sink)

        scan_docs = [d for d in all_docs if d.is_scan or any(p.is_scan for p in d.pages)]
        if scan_docs:
            sink.stage_start("S1_vision", f"OCR {len(scan_docs)} scan docs")
            refreshed = await asyncio.gather(
                *(
                    ocr_scan_document(
                        llm,
                        d,
                        config.cache_dir,
                        use_cache=config.use_cache,
                        vision_char_threshold=config.vision_char_threshold,
                    )
                    for d in scan_docs
                )
            )
            by_id: dict[str, IngestedDocument] = {}
            for doc, flags in refreshed:
                by_id[doc.doc_id] = doc
                vision_flags.extend(flags)
            all_docs = [by_id.get(d.doc_id, d) for d in all_docs]
            for flag in vision_flags:
                sink.log(f"[S1_vision] {flag}")
                if flag.startswith("VISION_PAGE_FAILED"):
                    # W5.5: soft warning — short transcript is triage noise when
                    # sibling pages / other docs already carry the numbers.
                    sink.alert("VISION_PAGE_FAILED", flag, severity="warning")
            sink.stage_end("S1_vision", f"{len(scan_docs)} processed · {len(vision_flags)} flags")

        docs_by_id = {d.doc_id: d for d in all_docs}
        sink.stage_start("S2_link", "account ↔ scenario + document binding")
        known_scenarios = set(scenario_ids)
        account_to_scenario, account_map_warnings = load_account_to_scenario(
            paths.ledger, known_scenarios
        )
        for w in account_map_warnings:
            sink.log(f"[S2] {w}")
            journal.append(w)
        linkages = link_documents(all_docs, account_to_scenario)
        linked = sum(1 for l in linkages if l.scenario_id)
        s2_entries, s2_unlinked = build_s2_entries(docs_by_id, linkages)
        sink.set_stage_payload(
            "S2_link",
            {"summary": s2_summary(s2_entries, s2_unlinked), "unlinked": s2_unlinked},
        )
        if s2_unlinked:
            sink.log_inventory("S2_unlinked", [format_s2_unlinked_log_line(e) for e in s2_unlinked])
            sink.log(f"[S2] unlinked · {len(s2_unlinked)} docs")
        sink.stage_end("S2_link", f"{linked}/{len(linkages)} linked")
        sink.log(f"[S2] linked {linked}/{len(linkages)} documents")

        # consolidated parent reports already consumed by the group-capex path are
        # expected-unlinked by design — exclude from linking_warnings.
        used_consolidated: set[str] = set()
        for name in sorted({l.company_name for l in linkages if l.scenario_id and l.company_name}):
            found = find_group_capex_document(all_docs, name)
            if found:
                used_consolidated.add(found.doc_id)
        linking_warnings = build_linking_warnings(
            docs_by_id, linkages, used_consolidated_doc_ids=used_consolidated
        )

        ledger_df = load_ledger(paths.ledger)
        signal_index = build_ledger_signal_index(ledger_df)
        sink.log(f"[S4] ledger loaded · {len(ledger_df)} rows")

        if vision_flags:
            journal.extend(vision_flags)
        if linking_warnings:
            journal.append(f"linking_warnings={len(linking_warnings)}")

        async def _extract_one(sid: str, clauses: dict) -> tuple[str, dict, _CaseExtract | None, dict | None]:
            """Returns (sid, clauses, extract|None, failed_trace|None)."""
            template_clauses = list(clauses.keys())
            if config.only_scenario and sid != config.only_scenario:
                case_trace = empty_case_trace(sid, "unprocessed", ["out of --only scope"], template_clauses)
                return (
                    sid,
                    clauses,
                    _CaseExtract(
                        scenario_id=sid,
                        status="unprocessed",
                        flags=["out of --only scope"],
                        template_clauses=template_clauses,
                        report_entry={"scenario_id": sid, "status": "unprocessed", "flags": ["out of --only scope"]},
                        case_trace=case_trace,
                    ),
                    None,
                )
            try:
                ext = await _extract_case(
                    sid,
                    linkages,
                    docs_by_id,
                    llm,
                    config.cache_dir,
                    config.use_cache,
                    sink,
                    template_clauses,
                    self_consistency=config.self_consistency,
                    dev=config.dev,
                )
                return sid, clauses, ext, None
            except Exception as exc:  # noqa: BLE001 — per-scenario recovery is the CLI contract
                sink.log(f"[FAIL] {sid} extract: {type(exc).__name__}: {exc}")
                flags = [f"{type(exc).__name__}: {exc}"]
                case_trace = empty_case_trace(sid, "failed", flags, template_clauses)
                if config.dev:
                    case_trace["dev"] = {
                        "documents": _dev_docs(sid, linkages, docs_by_id),
                        "extracts": None,
                        "transactions": [],
                    }
                return sid, clauses, None, case_trace

        extract_results = await asyncio.gather(
            *(_extract_one(sid, clauses) for sid, clauses in template["answers"].items())
        )

        # Deterministic: no cross-borrower FX fallback (CLAUDE.md — rate only from own docs).
        journal.append("global_fx_fallback=disabled")

        async def _finish_one(
            sid: str, clauses: dict, ext: _CaseExtract | None, failed_trace: dict | None
        ) -> tuple[str, dict[str, dict], dict, dict]:
            template_clauses = list(clauses.keys())
            if failed_trace is not None:
                return (
                    sid,
                    {clause: _stub_cell() for clause in clauses},
                    {"scenario_id": sid, "status": "failed", "flags": list(failed_trace.get("flags") or [])},
                    failed_trace,
                )
            assert ext is not None
            if ext.status != "ready":
                return (
                    sid,
                    {clause: _stub_cell() for clause in clauses},
                    ext.report_entry
                    or {"scenario_id": sid, "status": "unprocessed", "flags": list(ext.flags)},
                    ext.case_trace
                    or empty_case_trace(sid, "unprocessed", list(ext.flags), template_clauses),
                )
            try:
                cells, entry, case_trace = await _finish_case(
                    ext,
                    linkages,
                    docs_by_id,
                    all_docs,
                    ledger_df,
                    llm,
                    config.cache_dir,
                    config.use_cache,
                    journal,
                    sink,
                    fallback_fx_rate=None,
                    dev=config.dev,
                    signal_index=signal_index,
                    self_consistency=config.self_consistency,
                    engine=config.engine,
                    engine_diff_counter=engine_diff_counter,
                )
            except Exception as exc:  # noqa: BLE001 — per-scenario recovery is the CLI contract
                sink.log(f"[FAIL] {sid} finish: {type(exc).__name__}: {exc}")
                flags = [f"{type(exc).__name__}: {exc}"]
                case_trace = empty_case_trace(sid, "failed", flags, template_clauses)
                if config.dev:
                    case_trace["dev"] = {
                        "documents": _dev_docs(sid, linkages, docs_by_id),
                        "extracts": None,
                        "transactions": [],
                    }
                return (
                    sid,
                    {clause: _stub_cell() for clause in clauses},
                    {"scenario_id": sid, "status": "failed", "flags": flags},
                    case_trace,
                )
            filled: dict[str, dict] = {}
            for clause in clauses:
                filled[clause] = cells.get(clause, _stub_cell())
                if clause not in cells:
                    entry.setdefault("flags", []).append(f"clause {clause} not computed; stubbed")
            return sid, filled, entry, case_trace

        results = await asyncio.gather(
            *(_finish_one(sid, clauses, ext, fail) for sid, clauses, ext, fail in extract_results)
        )
        for sid, filled, entry, case_trace in results:
            answers[sid] = filled
            report_entries.append(entry)
            dev_section = case_trace.pop("dev", None)
            if config.dev and dev_section is not None:
                dev_cases[sid] = dev_section
            sink.tracker.record_case(sid, case_trace)

        report_entries.sort(key=lambda e: e.get("scenario_id", ""))
        if engine_diff_counter:
            journal.append(f"ENGINE_DIFF {len(engine_diff_counter)}/{sum(len(template['answers'][s]) for s in template['answers'])}")
        result = _finalize_write()

    except Exception as exc:  # noqa: BLE001 — file must exist even on fatal pre-cell failure
        sink.log(f"[FATAL] {type(exc).__name__}: {exc}")
        journal.append(f"fatal={type(exc).__name__}: {exc}")
        report_entries = sorted(report_entries, key=lambda e: e.get("scenario_id", ""))
        if not report_entries:
            report_entries = [
                {"scenario_id": sid, "status": "failed", "flags": [f"fatal: {type(exc).__name__}: {exc}"]}
                for sid in (scenario_ids or ["UNKNOWN"])
            ]
        try:
            result = _finalize_write()
        except Exception as write_exc:  # B4: finalize itself may be the fatal — raw dump
            journal.append(f"finalize_failed={type(write_exc).__name__}: {write_exc}")
            skel = submission if submission.get("answers") else _skeleton_submission(config, template)
            if not skel.get("answers") and not template.get("answers"):
                skel = {
                    "team": config.team,
                    "contact_email": config.contact_email,
                    "model": config.model_name,
                    "answers": {},
                }
            try:
                _emergency_dump(skel)
                journal.append("emergency_dump_ok")
            except Exception as dump_exc:
                journal.append(f"emergency_dump_failed={type(dump_exc).__name__}: {dump_exc}")
            result = None

    logs_dir = _write_confidence(result)
    if config.dev:
        logs_dir.mkdir(parents=True, exist_ok=True)
        unlinked = sorted(
            (
                {
                    "doc_id": link.doc_id,
                    "path": docs_by_id[link.doc_id].path,
                    "type": classify_doc_type(docs_by_id[link.doc_id]),
                    "account_anchor": link.account_anchor,
                    "company_name": link.company_name,
                    "unlink_reason": link.unlink_reason,
                    "preview": docs_by_id[link.doc_id].normalized_text[:400],
                }
                for link in linkages
                if not link.scenario_id and link.doc_id in docs_by_id
            ),
            key=lambda e: e["doc_id"],
        )
        dev_dump = {
            "dataset": str(dataset),
            "model": config.model_name,
            "profile": config.profile,
            "scenario_ids": scenario_ids,
            "unlinked_documents": unlinked,
            "scenarios": dev_cases,
            "ingest_failures": ingest_failures,
            "sanitize_fixes": sanitize_fixes,
        }
        dev_path = logs_dir / "dev_dump.json"
        dev_path.write_text(json.dumps(dev_dump, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        sink.log(f"[DEV] dev dump written to {dev_path}")

    sink.write_artifacts(logs_dir)
    if result is not None:
        sink.stage_end("S6_write", "ok" if result.ok else f"{len(result.violations)} violations")
    else:
        sink.stage_end("S6_write", "emergency_dump")

    return {
        "validation": result,
        "report_entries": report_entries,
        "submission": submission,
        "template": template,
        "sanitize_fixes": sanitize_fixes,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agent.run",
        description=(
            "Halyk AI Challenge — covenant-compliance agent. "
            "Читает documents/ + master_ledger_*.csv + submission_template.json "
            "и пишет submission.json (status / actual / evidence_txn_id)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python -m agent.run --dataset data/agentic-bank-public --out submission.json\n"
            "  python -m agent.run --dataset <DIR> --out submission.json --no-cache\n"
            "  python -m agent.run --dataset <DIR> --out submission.json --only P5 --dev\n"
            "\n"
            "Справка: -h / --help. Переменные окружения — .env.example."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="каталог датасета (ledger + documents/ + submission_template.json)",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="путь к итоговому submission.json (атомарная запись после validate)",
    )
    p.add_argument(
        "--profile",
        default="default",
        choices=["default", "fast", "careful"],
        help="пресет effort/concurrency: default | fast | careful (по умолчанию: default)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache"),
        help="каталог sidecar-кэша parsed/ + llm/ (по умолчанию: .cache)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="reproducibility seed (any stochastic path) + confidence metadata (default: 42)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="не читать и не писать sidecar-кэш (живой прогон API)",
    )
    p.add_argument(
        "--only",
        dest="only_scenario",
        default=None,
        metavar="SCENARIO_ID",
        help="dev-only: обработать один scenario_id (остальные — stubs)",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="dev-only: писать <logs>/dev_dump.json для tools/dev.py (на ответы не влияет)",
    )
    p.add_argument(
        "--engine",
        choices=["enum", "shadow", "tree"],
        default="shadow",
        help="движок ковенантов: enum (submit), shadow (enum+tree diff, default), tree (submit из expr)",
    )
    p.add_argument(
        "--no-tui",
        action="store_true",
        help="отключить Rich Live TUI (также: AGENT_NO_TUI=1 или non-TTY stdout)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from agent.config import load_config

    args = build_arg_parser().parse_args(argv)
    config = load_config(
        dataset_dir=args.dataset,
        out_path=args.out,
        cache_dir=args.cache,
        profile=args.profile,
        seed=args.seed,
        use_cache=not args.no_cache,
        only_scenario=args.only_scenario,
        dev=args.dev,
        engine=args.engine,
    )
    sink = create_progress_sink(
        model_name=config.model_name,
        out_path=str(config.out_path),
        force_quiet=args.no_tui,
    )
    try:
        outcome = asyncio.run(run(config, sink))
    except BaseException as exc:
        sink.finish(f"failed: {type(exc).__name__}: {exc}")
        # Last-resort: if we somehow escape run() without a file, surface the error.
        # Per-scenario failures are already recovered inside run().
        raise
    validation = outcome.get("validation")
    if validation is None:
        # Emergency dump path: file may exist but we have no ValidationResult.
        sink.finish(
            f"emergency dump written to {config.out_path} (no validation result)"
        )
        return 1
    if not validation.ok:
        for v in validation.violations:
            sink.log(f"FAIL {v}")
            print(f"FAIL {v}")
        sink.finish("validation failed")
        return 1
    sink.finish(f"submission written to {config.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
