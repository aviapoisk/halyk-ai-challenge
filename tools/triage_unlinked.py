"""Offline triage of unlinked documents — no LLM, not imported by agent.run.

Usage:
    python -m tools.triage_unlinked --dataset data/agentic-bank-public [--cache .cache]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from agent.doc_inventory import build_linking_warnings, build_s2_entries, s2_summary
from agent.s1_ingest import ingest_directory
from agent.s2_classify import classify_doc_type, find_group_capex_document
from agent.s2_link import link_documents, load_account_to_scenario


def _header_lines(text: str, n: int = 4) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:n]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline triage of unlinked documents (no LLM)")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=Path(".cache"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset: Path = args.dataset
    from agent.dataset import discover_dataset

    paths = discover_dataset(dataset)
    docs, failures = ingest_directory(paths.documents, cache_dir=args.cache)
    for f in failures:
        print(f"FAIL ingest {f}")
    docs_by_id = {d.doc_id: d for d in docs}
    account_to_scenario, map_warnings = load_account_to_scenario(paths.ledger)
    for w in map_warnings:
        print(f"WARN {w}")
    linkages = link_documents(docs, account_to_scenario)

    entries, unlinked = build_s2_entries(docs_by_id, linkages)
    summary = s2_summary(entries, unlinked)

    used_consolidated: set[str] = set()
    for name in sorted({l.company_name for l in linkages if l.scenario_id and l.company_name}):
        found = find_group_capex_document(docs, name)
        if found:
            used_consolidated.add(found.doc_id)
    candidates = build_linking_warnings(
        docs_by_id, linkages, used_consolidated_doc_ids=used_consolidated
    )
    # Also surface consolidated that were NOT consumed (may be noise or a miss).
    for link in linkages:
        if link.scenario_id is not None:
            continue
        doc = docs_by_id[link.doc_id]
        if classify_doc_type(doc) == "consolidated_parent_report" and link.doc_id not in used_consolidated:
            if not any(c["doc_id"] == link.doc_id for c in candidates):
                candidates.append(
                    {
                        "doc_id": link.doc_id,
                        "type": "consolidated_parent_report",
                        "unlink_reason": link.unlink_reason,
                        "is_scan": doc.is_scan,
                    }
                )

    print(
        f"documents={summary['n_documents']} linked={summary['n_linked']} "
        f"unlinked={summary['n_unlinked']}"
    )
    print(f"by_channel={summary['by_channel']}")
    print(f"by_type={summary['by_type']}")
    print()

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in unlinked:
        key = entry.get("company_name") or "(no name)"
        groups[key].append(entry)

    print("=== unlinked by company name ===")
    for name in sorted(groups):
        print(f"\n[{name}] ×{len(groups[name])}")
        for entry in sorted(groups[name], key=lambda e: e["doc_id"]):
            doc = docs_by_id[entry["doc_id"]]
            chars = sum(p.char_count for p in doc.pages)
            print(
                f"  {entry['doc_id']} type={entry['type']} reason={entry.get('unlink_reason')} "
                f"pages={len(doc.pages)} chars={chars} scan={doc.is_scan}"
            )
            for ln in _header_lines(doc.normalized_text):
                print(f"    | {ln[:120]}")

    print("\n=== non-noise candidates (important type / scan / unused consolidated) ===")
    if not candidates:
        print("  (none)")
    for c in sorted(candidates, key=lambda e: e["doc_id"]):
        print(
            f"  {c['doc_id']} type={c['type']} reason={c.get('unlink_reason')} is_scan={c['is_scan']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
