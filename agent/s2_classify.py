"""S2 (classification half) — deterministic document-type routing.

docs/FINDINGS-2026-08-06.md §2 counts a small, distinctive set of document
types in the open corpus (loan agreements, "АУДИТОРСКОЕ ДЕЛО", KYC dossiers,
"НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ" stamps, consolidated/Group reports) that are each
identifiable from a couple of fixed, corpus-wide phrases — cheap regex
classification is used instead of spending an LLM call per document just to
learn its type. LLM calls (S3) are reserved for the facts *inside* each
document, which regex cannot get right.
"""

from __future__ import annotations

import re

from agent.normalize import normalize_company_name
from agent.schemas import IngestedDocument

# Priority is the list order — first match wins. Declared as an ordered list so
# the contract is explicit (dict insertion order alone is not the documented API).
# W3.1: consolidated/Group reports before auditor_notes — parent FS often contain
# "Auditor's Report" in the body and must not be mis-typed as notes.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("loan_agreement", re.compile(r"ДОГОВОР\s+БАНКОВСКОГО\s+ЗАЙМА|LOAN\s+AGREEMENT|BANK\s+LOAN\s+AGREEMENT", re.IGNORECASE)),
    ("kyc_dossier", re.compile(r"(?:Знай\s+своего\s+клиент|KYC)", re.IGNORECASE)),
    ("consolidated_parent_report", re.compile(r"CONSOLIDATED|консолидирова", re.IGNORECASE)),
    (
        "auditor_notes",
        re.compile(
            r"АУДИТОРСКОЕ\s+ДЕЛО|Примечания\s+к\s+финансовой\s+отчётности|AUDITOR.?S?\s+NOTES|"
            r"Аудиторск\w+\s+отчёт|Auditor.?s?\s+Report|AR-\d{4}|"
            r"служебн\w+\s+записк|казначейств|Treasury\s+memo|treasury\s+note",
            re.IGNORECASE,
        ),
    ),
]


# W3.4: status-formula draft markers only — bare «промежуточн» / «проект» appear in
# final AUP procedure language ("промежуточного и завершающего этапов").
_DRAFT_STATUS = (
    "не является окончательн",
    "не является окончательной",
    "заменена окончательным",
    "заменён окончательным",
    "заменен окончательным",
    "interim working paper",
    "superseded by the final",
    "replaced by the final",
    "draft — not for covenant",
    "draft - not for covenant",
    "not a final position",
    "not the final position",
)
_DRAFT_LABEL = re.compile(
    r"(?:^|[\n.;:])\s*(?:проект|черновик)\b|"
    r"\bdraft\b(?:\s*[—\-–]\s*|\s+)",
    re.IGNORECASE,
)
_FINAL_STATUS = (
    "окончательной позиц",
    "окончательная позиц",
    "окончательн",
    "final auditor",
    "final report",
    "final position",
    "вывод в отчёте",
    "заменяет промежуточн",
    "replaces the interim",
    "replaces interim",
)
# W5.1: unstamped year-end notes (no «окончательн» token) still beat draft siblings.
_AUTHORITATIVE_NOTES_RE = re.compile(
    r"аудитор\s*ское\s*дело|примечания\s+к\s+финансовой|auditor'?s?\s+notes",
    re.IGNORECASE,
)
_RECLASS_DENIED_RE = re.compile(
    r"переклассификац\w*.{0,40}не\s+требова|reclassifications?\s+(?:were\s+)?not\s+required",
    re.IGNORECASE | re.DOTALL,
)


def auditor_doc_is_interim_or_draft(text: str) -> bool:
    """True when the note's *status* is interim/draft and must not override a final AR."""
    t = (text or "").casefold()
    if any(tok in t for tok in _DRAFT_STATUS) or bool(_DRAFT_LABEL.search(text or "")):
        # A superseded draft saying «заменена окончательным» stays draft even if
        # the word «окончательн» appears in that same phrase.
        return True
    # Procedural «промежуточн» alone is not a draft marker.
    return False


def auditor_doc_is_final(text: str) -> bool:
    t = (text or "").casefold()
    if auditor_doc_is_interim_or_draft(text):
        return False
    return any(tok in t for tok in _FINAL_STATUS)


def auditor_doc_is_authoritative(text: str) -> bool:
    """W5.1: final stamp OR non-draft year-end notes body (beats draft siblings)."""
    if auditor_doc_is_interim_or_draft(text):
        return False
    if auditor_doc_is_final(text):
        return True
    return bool(_AUTHORITATIVE_NOTES_RE.search(text or ""))


def auditor_notes_deny_reclassifications(text: str) -> bool:
    """True when authoritative prose says reclassifications were not required."""
    return bool(_RECLASS_DENIED_RE.search(text or ""))


def classify_doc_type(doc: IngestedDocument) -> str | None:
    text = doc.normalized_text
    for doc_type, pattern in _PATTERNS:
        if pattern.search(text):
            return doc_type
    return None


def find_group_capex_document(all_docs: list[IngestedDocument], borrower_name: str) -> IngestedDocument | None:
    """Exact normalized-name containment in a consolidated parent report.

    Uses normalize_company_name so «Shymkent Refinery JSC» does not match inside
    «Shymkent Refinery Services JSC» (contiguous needle after OPF fold). Multiple
    hits → None (ambiguous).
    """
    needle = normalize_company_name(borrower_name)
    if not needle:
        return None
    hits: list[IngestedDocument] = []
    for doc in all_docs:
        if classify_doc_type(doc) != "consolidated_parent_report":
            continue
        haystack = normalize_company_name(doc.normalized_text)
        if needle in haystack:
            hits.append(doc)
    if len(hits) == 1:
        return hits[0]
    return None
