"""Deterministic KYC pledge-table enricher.

Cold LLM extracts sometimes return unrestricted_pledge_threshold_pct but omit
subsidiary rows that appear only in the collateral table (voting_rights=0).
Python scrapes those rows from normalized dossier text and merges them into
KYCFacts so apply_pledge_based_transfer_categories can run.
"""

from __future__ import annotations

import re

from agent.normalize import normalize_company_name
from agent.schemas import KYCFacts, RelatedParty

_SECTION_MARKERS = (
    "обеспечительное покрытие",
    "доля активов в залоге",
    "в залоге",
    "collateral coverage",
    "pledged assets",
    "assets pledged",
)

_PIPE_ROW = re.compile(
    r"^\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$",
    re.MULTILINE,
)
_PCT_IN_CELL = re.compile(r"(\d+(?:[.,]\d+)?)\s*%?")
_FALLBACK_LINE = re.compile(
    r"^(.{8,}?)\s+(\d+(?:[.,]\d+)?)\s*%\s*$",
    re.MULTILINE,
)
_THR_BELOW = re.compile(
    r"(?:ниже|below)\s+(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)
_UNRESTRICTED_NEAR = re.compile(
    r"неограничен|unrestricted",
    re.IGNORECASE,
)


def _parse_pct(raw: str) -> float | None:
    m = _PCT_IN_CELL.search(raw.replace(" ", ""))
    if not m:
        m = _PCT_IN_CELL.search(raw)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def _is_header_or_sep(name: str, pct_cell: str) -> bool:
    n = name.casefold().strip()
    p = pct_cell.casefold().strip()
    if set(n) <= {"-", ":"} or set(p.replace("|", "")) <= {"-", ":"}:
        return True
    if "---" in n or "---" in p:
        return True
    header_bits = (
        "дочерн",
        "организац",
        "доля",
        "залог",
        "subsidiary",
        "pledged",
        "assets",
        "coverage",
        "name",
    )
    if any(b in n for b in header_bits) and _parse_pct(pct_cell) is None:
        return True
    return False


def _find_section_window(text: str) -> tuple[str, bool]:
    """Return (window, markers_present). Window is text from first marker onward."""
    folded = text.casefold()
    positions = [folded.find(m) for m in _SECTION_MARKERS if folded.find(m) >= 0]
    if not positions:
        return "", False
    start = min(positions)
    # Cap window so ownership tables below don't leak in; pledge blocks are short.
    return text[start : start + 2500], True


def _rows_from_markdown(window: str) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for m in _PIPE_ROW.finditer(window):
        name = m.group(1).strip()
        pct_cell = m.group(2).strip()
        if _is_header_or_sep(name, pct_cell):
            continue
        pct = _parse_pct(pct_cell)
        if pct is None:
            continue
        # Company-like: at least two tokens, not a bare number.
        tokens = [t for t in re.split(r"\s+", name) if t]
        if len(tokens) < 2:
            continue
        out.append((name, pct, m.group(0).strip()))
    return out


def _rows_from_fallback(window: str) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for m in _FALLBACK_LINE.finditer(window):
        name = m.group(1).strip().lstrip("|").strip()
        if "|" in name:
            continue
        pct = float(m.group(2).replace(",", "."))
        tokens = [t for t in re.split(r"\s+", name) if t]
        if len(tokens) < 2:
            continue
        if _is_header_or_sep(name, m.group(2)):
            continue
        out.append((name, pct, m.group(0).strip()))
    return out


def _threshold_from_text(window: str, full_text: str) -> float | None:
    search_in = window if window else full_text
    for m in _THR_BELOW.finditer(search_in):
        # Prefer matches near unrestricted wording.
        tail = search_in[m.start() : m.start() + 200]
        if _UNRESTRICTED_NEAR.search(tail) or _UNRESTRICTED_NEAR.search(
            search_in[max(0, m.start() - 80) : m.end() + 120]
        ):
            return float(m.group(1).replace(",", "."))
    # Fallback: first "ниже X%" in the pledge window.
    if window:
        m = _THR_BELOW.search(window)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def enrich_kyc_pledge_from_text(facts: KYCFacts, text: str) -> tuple[KYCFacts, list[str]]:
    """Merge pledge-table subsidiaries / threshold from dossier text into KYCFacts."""
    flags: list[str] = []
    if not text or not text.strip():
        return facts, flags

    window, markers = _find_section_window(text)
    rows = _rows_from_markdown(window) if window else []
    if not rows and window:
        rows = _rows_from_fallback(window)

    counterparties = list(facts.counterparties or [])
    by_norm = {
        normalize_company_name(p.counterparty_name): i for i, p in enumerate(counterparties) if p.counterparty_name
    }

    merged = 0
    for name, pct, quote in rows:
        key = normalize_company_name(name)
        if not key:
            continue
        if key in by_norm:
            i = by_norm[key]
            existing = counterparties[i]
            if existing.pledged_assets_pct is None:
                counterparties[i] = existing.model_copy(
                    update={"pledged_assets_pct": pct, "quote": existing.quote or quote}
                )
                merged += 1
            elif abs(float(existing.pledged_assets_pct) - pct) > 1e-9:
                flags.append(
                    f"kyc_pledge_conflict {name!r}: llm={existing.pledged_assets_pct} text={pct}"
                )
            continue
        counterparties.append(
            RelatedParty(
                counterparty_name=name,
                voting_rights_pct=0.0,
                pledged_assets_pct=pct,
                quote=quote,
            )
        )
        by_norm[key] = len(counterparties) - 1
        merged += 1

    thr = facts.unrestricted_pledge_threshold_pct
    thr_from_text = False
    if thr is None:
        parsed = _threshold_from_text(window, text)
        if parsed is not None:
            thr = parsed
            thr_from_text = True

    updates: dict = {}
    if counterparties != list(facts.counterparties or []):
        updates["counterparties"] = counterparties
    if thr_from_text:
        updates["unrestricted_pledge_threshold_pct"] = thr
        flags.append(f"kyc_pledge_threshold_from_text={thr}")

    if merged:
        flags.append(f"kyc_pledge_table_merged n={merged}")

    out = facts.model_copy(update=updates) if updates else facts

    n_pledged = sum(1 for p in (out.counterparties or []) if p.pledged_assets_pct is not None)
    if markers and out.unrestricted_pledge_threshold_pct is not None and n_pledged == 0:
        flags.append("kyc_pledge_table_unparsed")

    return out, flags
