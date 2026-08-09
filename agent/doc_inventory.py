"""Per-document inventory builders for S2 observability (agent.log + trace.json)."""

from __future__ import annotations

from collections import Counter

from agent.s2_classify import classify_doc_type
from agent.schemas import DocumentLinkage, IngestedDocument


def build_s2_entries(
    docs_by_id: dict[str, IngestedDocument],
    linkages: list[DocumentLinkage],
) -> tuple[list[dict], list[dict]]:
    all_entries: list[dict] = []
    unlinked: list[dict] = []
    for link in sorted(linkages, key=lambda l: l.doc_id):
        doc = docs_by_id[link.doc_id]
        entry = {
            "doc_id": link.doc_id,
            "type": classify_doc_type(doc),
            "scenario_id": link.scenario_id,
            "link_channel": link.link_channel,
            "account_anchor": link.account_anchor,
            "company_name": link.company_name,
            "unlink_reason": link.unlink_reason,
        }
        all_entries.append(entry)
        if link.scenario_id is None:
            unlinked.append(
                {
                    "doc_id": link.doc_id,
                    "type": entry["type"],
                    "account_anchor": link.account_anchor,
                    "company_name": link.company_name,
                    "unlink_reason": link.unlink_reason,
                }
            )
    return all_entries, unlinked


def format_s2_unlinked_log_line(entry: dict) -> str:
    reason = entry.get("unlink_reason") or "?"
    return (
        f"[S2] unlinked {entry['doc_id']} type={entry['type']} "
        f"reason={reason} account={entry['account_anchor']} company={entry['company_name']}"
    )


_IMPORTANT_UNLINKED_TYPES = frozenset({"loan_agreement", "kyc_dossier", "auditor_notes"})


def build_linking_warnings(
    docs_by_id: dict[str, IngestedDocument],
    linkages: list[DocumentLinkage],
    *,
    used_consolidated_doc_ids: set[str] | None = None,
) -> list[dict]:
    """Unlinked docs that look important (typed covenant docs or full scans).

    consolidated_parent_report is excluded when already consumed by
    find_group_capex_document for some scenario (expected-unlinked by design).
    """
    used = used_consolidated_doc_ids or set()
    warnings: list[dict] = []
    for link in sorted(linkages, key=lambda l: l.doc_id):
        if link.scenario_id is not None:
            continue
        doc = docs_by_id[link.doc_id]
        doc_type = classify_doc_type(doc)
        if doc_type == "consolidated_parent_report" and link.doc_id in used:
            continue
        if doc_type not in _IMPORTANT_UNLINKED_TYPES and not doc.is_scan:
            continue
        warnings.append(
            {
                "doc_id": link.doc_id,
                "type": doc_type,
                "unlink_reason": link.unlink_reason,
                "is_scan": doc.is_scan,
            }
        )
    return warnings


def s2_summary(entries: list[dict], unlinked: list[dict]) -> dict:
    linked = sum(1 for e in entries if e["scenario_id"])
    return {
        "n_documents": len(entries),
        "n_linked": linked,
        "n_unlinked": len(unlinked),
        "by_channel": dict(
            sorted(Counter(e["link_channel"] or "unlinked" for e in entries).items())
        ),
        "by_type": dict(sorted(Counter(e["type"] or "unclassified" for e in entries).items())),
    }
