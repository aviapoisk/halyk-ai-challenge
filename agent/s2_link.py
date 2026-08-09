"""S2 (linking half only) — deterministic borrower linking, no LLM required.

Two channels (docs/FINDINGS-2026-08-06.md §2):
  1. Account anchors from the document body, matched against ledger account_id
     values (pattern built at runtime — never hardcoded ACC-\\d{4} for private sets).
  2. Exact normalized company legal name from the page-1 header -> resolved via a
     bootstrap map built from documents that carry *both* signals (channel 1 + a name).
     Deliberately exact matching only: near-duplicate names in the corpus
     ("Shymkent Refinery Services JSC" vs "Shymkent Refinery JSC") make fuzzy matching
     dangerous (it would transplant one borrower's covenants onto another).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from agent.normalize import (
    find_account_anchors,
    find_company_name,
    make_anchor_pattern,
    normalize_company_name,
)
from agent.s1_ingest import ingest_directory
from agent.schemas import DocumentLinkage, IngestedDocument

_TXN_RE = re.compile(r"^TXN-(?P<scenario>.+)-(?P<seq>\d+)$")
# Legacy open-set form — kept so unknown ACC-NNNN still surfaces as anchor_unknown_account
# when the ledger pattern (private-set ids) does not match the token.
_LEGACY_ACC = re.compile(r"A\s*C\s*C\s*-\s*(\d\s*){3,6}")


def _legacy_acc_anchors(text: str) -> list[str]:
    out: list[str] = []
    for m in _LEGACY_ACC.finditer(text):
        digits = re.sub(r"\s+", "", m.group(0).split("-", 1)[1])
        out.append(f"ACC-{digits}")
    return out


def load_account_to_scenario(ledger_path: Path) -> tuple[dict[str, str], list[str]]:
    """account_id -> scenario_id from the ledger. Collisions → majority vote + warning."""
    votes: dict[str, Counter[str]] = {}
    with open(ledger_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = _TXN_RE.match(row.get("txn_id", ""))
            account_id = row.get("account_id", "")
            if not m or not account_id:
                continue
            scenario = m.group("scenario")
            votes.setdefault(account_id, Counter())[scenario] += 1

    mapping: dict[str, str] = {}
    warnings: list[str] = []
    for account_id, counter in votes.items():
        winner, _n = counter.most_common(1)[0]
        if len(counter) > 1:
            warnings.append(f"account_id {account_id} → {dict(counter)}; выбран {winner}")
        mapping[account_id] = winner
    return mapping, warnings


def link_documents(docs: list[IngestedDocument], account_to_scenario: dict[str, str]) -> list[DocumentLinkage]:
    linkages: dict[str, DocumentLinkage] = {}
    name_to_scenario: dict[str, str] = {}
    anchor_pattern = make_anchor_pattern(account_to_scenario.keys())

    # Pass 1: anchor channel, and bootstrap the name->scenario map from docs that have both.
    for doc in docs:
        text = doc.normalized_text
        anchors = find_account_anchors(text, anchor_pattern)
        # Merge legacy ACC-* hits so unknown accounts remain diagnosable.
        for a in _legacy_acc_anchors(text):
            if a not in anchors:
                anchors.append(a)
        anchors = sorted(set(anchors), key=lambda x: (-len(x), x))
        name = find_company_name(text)
        scenarios = {account_to_scenario[a] for a in anchors if a in account_to_scenario}
        if len(scenarios) > 1:
            linkage = DocumentLinkage(
                doc_id=doc.doc_id,
                account_anchor=anchors[0] if anchors else None,
                company_name=name,
                unlink_reason="anchor_ambiguous",
            )
            linkages[doc.doc_id] = linkage
            continue

        anchor = None
        if len(scenarios) == 1:
            # Prefer an anchor that maps to the unique scenario.
            for a in anchors:
                if a in account_to_scenario and account_to_scenario[a] in scenarios:
                    anchor = a
                    break
            if anchor is None and anchors:
                anchor = anchors[0]
        elif anchors:
            # Anchors present but none in map — keep first for unlink diagnostics.
            anchor = anchors[0]

        linkage = DocumentLinkage(doc_id=doc.doc_id, account_anchor=anchor, company_name=name)

        if anchor and anchor in account_to_scenario:
            linkage.scenario_id = account_to_scenario[anchor]
            linkage.link_channel = "account_anchor"
            if name:
                key = normalize_company_name(name)
                existing = name_to_scenario.get(key)
                if existing is not None and existing != linkage.scenario_id:
                    name_to_scenario[key] = "__AMBIGUOUS__"
                else:
                    name_to_scenario[key] = linkage.scenario_id

        linkages[doc.doc_id] = linkage

    # Pass 2: name channel for docs without a resolved anchor.
    for linkage in linkages.values():
        if linkage.scenario_id is not None or not linkage.company_name:
            continue
        if linkage.unlink_reason == "anchor_ambiguous":
            continue
        key = normalize_company_name(linkage.company_name)
        scenario = name_to_scenario.get(key)
        if scenario and scenario != "__AMBIGUOUS__":
            linkage.scenario_id = scenario
            linkage.link_channel = "company_name"

    # Pass 3: unlink_reason for docs that never resolved (LINK-GUARDS G3).
    for linkage in linkages.values():
        if linkage.scenario_id is not None:
            continue
        if linkage.unlink_reason:
            continue
        if linkage.account_anchor and linkage.account_anchor not in account_to_scenario:
            linkage.unlink_reason = "anchor_unknown_account"
            continue
        if linkage.company_name:
            key = normalize_company_name(linkage.company_name)
            boot = name_to_scenario.get(key)
            if boot == "__AMBIGUOUS__":
                linkage.unlink_reason = "name_ambiguous"
            else:
                linkage.unlink_reason = "name_not_bootstrapped"
            continue
        linkage.unlink_reason = "no_anchor"

    return list(linkages.values())


def _main() -> int:
    import argparse
    from collections import Counter as Ctr

    from agent.dataset import discover_dataset

    p = argparse.ArgumentParser(description="S1+S2 linking smoke run: coverage stats over the whole corpus")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=Path(".cache"))
    args = p.parse_args()

    paths = discover_dataset(args.dataset)
    docs, failures = ingest_directory(paths.documents, cache_dir=args.cache)
    account_to_scenario, map_warnings = load_account_to_scenario(paths.ledger)
    for w in map_warnings:
        print(f"WARN {w}")
    for f in failures:
        print(f"FAIL ingest {f}")
    linkages = link_documents(docs, account_to_scenario)

    n_anchor_found = sum(1 for l in linkages if l.account_anchor)
    n_name_found = sum(1 for l in linkages if l.company_name)
    by_channel = Ctr(l.link_channel or "unlinked" for l in linkages)

    print(f"documents: {len(linkages)}")
    print(f"anchor found in body: {n_anchor_found}")
    print(f"company name found: {n_name_found}")
    print(f"by channel: {dict(by_channel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
