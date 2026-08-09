"""Text sanitizer + normalization (docs/mcp-approach-review.md §5.2, docs/FINDINGS-2026-08-06.md §2).

Two representations of every document are kept everywhere downstream:
  - raw text: exactly as extracted, used only for pulling citation quotes.
  - normalized text: sanitized (NBSP/invisible/control chars removed, NFC form),
    used for all regex matching (anchors, clause numbers, amounts, dates).

Rule (CLAUDE.md / FINDINGS §2): never regex-match on raw text. The parser choice
and PDF letter-spacing (`А У Д И Т О Р С К О Е`) silently drop ~18% of anchor
matches on raw text; always normalize first.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width, bidi control, soft-hyphen characters — stripped entirely.
_INVISIBLE_CHARS = dict.fromkeys(
    ord(ch)
    for ch in (
        "​‌‍﻿­"  # zero-width space/joiners, BOM, soft hyphen
        "‪‫‬‭‮"  # bidi embedding/override controls
        "⁦⁧⁨⁩"  # bidi isolate controls
    )
)

_NBSP_CHARS = {0x00A0: " ", 0x202F: " ", 0x2007: " "}  # NBSP, narrow NBSP, figure space


def sanitize(text: str) -> str:
    """NBSP->space, strip invisible/bidi/control chars (except \\n\\t), NFC-normalize."""
    text = text.translate(_NBSP_CHARS)
    text = text.translate(_INVISIBLE_CHARS)
    text = "".join(ch for ch in text if not (unicodedata.category(ch) == "Cc" and ch not in "\n\t"))
    return unicodedata.normalize("NFC", text)


# Fallback when no ledger account ids are available (open-set ACC-NNNN).
_ANCHOR_PATTERN = re.compile(r"A\s*C\s*C\s*-\s*(\d\s*){4}")


def find_account_anchor(normalized_text: str) -> str | None:
    """Returns the first "ACC-NNNN" anchor found, tolerant of PDF letter-spacing."""
    m = _ANCHOR_PATTERN.search(normalized_text)
    if not m:
        return None
    digits = re.sub(r"\s+", "", m.group(0).split("-", 1)[1])
    return f"ACC-{digits}"


def make_anchor_pattern(account_ids: list[str] | tuple[str, ...] | set[str]) -> re.Pattern[str] | None:
    """Build a letter-spacing-tolerant pattern from ledger account_id values.

    Longer ids first so ACC-7801-02 is not truncated by ACC-7801. Lookarounds
    keep ACC-78051 from matching ACC-7805.
    """
    ids = sorted({a.strip() for a in account_ids if a and str(a).strip()}, key=len, reverse=True)
    if not ids:
        return None
    alts = "|".join(r"\s*".join(re.escape(ch) for ch in a) for a in ids)
    return re.compile(rf"(?<![\w-])(?:{alts})(?![\w-])")


def find_account_anchors(normalized_text: str, pattern: re.Pattern[str] | None) -> list[str]:
    """All anchors in the document. Empty pattern → fall back to legacy ACC-\\d{4}."""
    if pattern is None:
        one = find_account_anchor(normalized_text)
        return [one] if one else []
    return sorted({re.sub(r"\s+", "", m.group(0)) for m in pattern.finditer(normalized_text)})


# Legal-entity suffixes OR prefixes (ТОО "Name" / JSC Name) observed in the corpus.
_COMPANY_LINE_PATTERN = re.compile(
    r"^(?:"
    r"(?P<name>[A-ZА-Я][\w .,'\"&\-]{2,80}?\s+(?:JSC|LLP|LLC|Ltd\.?|ТОО|АО\.?))"
    r"|"
    r"(?:(?:ТОО|АО|JSC|LLP|LLC|Ltd\.?)\s+[\"«]?(?P<name_pref>[A-ZА-Я][\w .,'\"&\-]{2,80}?))"
    r")\s*$",
    re.IGNORECASE,
)

# Dotted OPF spellings seen in KYC vs ledger (L.L.P. vs LLP) — apply before word-boundary aliases.
_DOTTED_OPF = (
    (re.compile(r"\bL\.?\s*L\.?\s*P\.?\b", re.IGNORECASE), "llp"),
    (re.compile(r"\bL\.?\s*L\.?\s*C\.?\b", re.IGNORECASE), "llc"),
)

# Canonical OPF aliases — suffix-only, never fuzzy on the whole name.
_OPF_ALIASES = (
    (re.compile(r"\bтоо\b", re.IGNORECASE), "llp"),
    (re.compile(r"\bао\.?\b", re.IGNORECASE), "jsc"),
    (re.compile(r"\bltd\.?\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\bllc\b", re.IGNORECASE), "llc"),
    (re.compile(r"\bjsc\b", re.IGNORECASE), "jsc"),
    (re.compile(r"\bllp\b", re.IGNORECASE), "llp"),
)


def collapse_letter_spacing(line: str) -> str:
    """Collapses "A U D I T ...  D E A L" -> "AUDIT DEAL" when a line is >=70% single
    letters separated by single spaces (heuristic for PDF letter-spacing rendering)."""
    tokens = line.split(" ")
    if not tokens:
        return line
    single_char = sum(1 for t in tokens if len(t) == 1)
    if single_char / len(tokens) < 0.7:
        return line
    return re.sub(r"(?<=\w) (?=\w)", "", line)


def fold_quotes(text: str) -> str:
    """Normalize curly/guillemet quotes to ASCII for matching."""
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("«", '"')
        .replace("»", '"')
        .replace("’", "'")
    )


def fold_for_substring(text: str) -> str:
    """NFKC + dash/quote fold + whitespace collapse for substring searches."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = fold_quotes(text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_company_name(name: str) -> str:
    """Canonical form for exact matching: collapsed whitespace, quote/comma stripped,
    L.L.P./L.L.C. folded, OPF suffix aliases, case-folded. Deliberately exact (not fuzzy)
    — near-duplicate names in the corpus (e.g. "Shymkent Refinery Services JSC" vs
    "Shymkent Refinery JSC") must not collide."""
    name = fold_quotes(name)
    name = re.sub(r"\s+", " ", name).strip()
    for pattern, canon in _DOTTED_OPF:
        name = pattern.sub(canon, name)
    for pattern, canon in _OPF_ALIASES:
        name = pattern.sub(canon, name)
    # Strip quotes/commas and residual punctuation that differs between KYC and ledger.
    name = re.sub(r'["«»\',]', "", name)
    name = re.sub(r"\.(?!\d)", "", name)  # drop periods except decimal points
    name = re.sub(r"\s+", " ", name).strip()
    return name.casefold()


_OPF_CORE_STRIP = re.compile(
    r"\b(?:jsc|llp|llc|ltd|тоо|ао)\b\.?",
    re.IGNORECASE,
)

# Wide stop-list for related-party token overlap (legal OPF + corporate noise).
_NAME_CONTENT_STOP = frozenset(
    {
        "llp",
        "llc",
        "jsc",
        "ao",
        "too",
        "inc",
        "ltd",
        "co",
        "company",
        "group",
        "holding",
        "holdings",
        "partners",
        "capital",
        "services",
        "service",
    }
)


def core_company_name(name: str) -> str:
    """Normalized name with OPF suffix/prefix tokens removed (C4 second-tier match)."""
    norm = normalize_company_name(name)
    core = _OPF_CORE_STRIP.sub(" ", norm)
    return re.sub(r"\s+", " ", core).strip()


def company_name_content_tokens(name: str, *, min_len: int = 3) -> set[str]:
    """Content tokens from a legal name after OPF/corporate-noise stop-list filter."""
    norm = normalize_company_name(name)
    return {t for t in norm.split() if t and t not in _NAME_CONTENT_STOP and len(t) >= min_len}


_CLAUSE_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)+")

# «первый|…|четвёртый / Q1–Q4 финансовый квартал» ending on a date → quarter window.
_QUARTER_ORDINAL = {
    "перв": 1,
    "втор": 2,
    "трет": 3,
    "четвёрт": 4,
    "четверт": 4,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
}
_QUARTER_RE = re.compile(
    r"(?:"
    r"(?P<ord>перв\w*|втор\w*|трет\w*|четвёрт\w*|четверт\w*|first|second|third|fourth|1st|2nd|3rd|4th)"
    r"|(?:q|к)\s*(?P<qn>[1-4])"
    r")"
    r".{0,40}?"
    r"(?:квартал|quarter)",
    re.IGNORECASE | re.DOTALL,
)
_AS_OF_RE = re.compile(
    r"(?:по\s+состоянию\s+на|остаток\s+на|as\s+of)\s+"
    r"(?P<d>\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4})",
    re.IGNORECASE,
)


def quarter_period_from_quote(
    quote: str,
    period_end: str | None = None,
) -> tuple[str, str] | None:
    """If quote names an N-th fiscal quarter ending on a date, return (start, end) ISO dates.

    Uses period_end from the clause when the ending date is not embedded next to the
    quarter phrase. Never scenario-specific — pure text rule.
    """
    if not quote:
        return None
    m = _QUARTER_RE.search(quote)
    if not m:
        return None
    if m.group("qn"):
        qn = int(m.group("qn"))
    else:
        ord_raw = (m.group("ord") or "").casefold()
        qn = None
        for prefix, n in _QUARTER_ORDINAL.items():
            if ord_raw.startswith(prefix):
                qn = n
                break
        if qn is None:
            return None

    end_s = period_end
    # Prefer an ISO date near the match (within ~80 chars after).
    tail = quote[m.end() : m.end() + 80]
    dm = re.search(r"(20\d{2})-(\d{2})-(\d{2})", tail) or re.search(
        r"(20\d{2})-(\d{2})-(\d{2})", quote
    )
    if dm:
        end_s = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
    if not end_s:
        return None
    try:
        y, mo, d = (int(x) for x in end_s[:10].split("-"))
    except ValueError:
        return None
    # Map quarter number → calendar quarter start in the same year as period_end.
    starts = {1: (y, 1, 1), 2: (y, 4, 1), 3: (y, 7, 1), 4: (y, 10, 1)}
    sy, sm, sd = starts[qn]
    start_s = f"{sy:04d}-{sm:02d}-{sd:02d}"
    # If stated end is not the quarter-end month, still keep the parsed end (clause wins).
    return start_s, end_s[:10]


def as_of_balance_from_quote(quote: str) -> bool:
    """True when the clause measures a stock/balance as of a date."""
    if not quote:
        return False
    return _AS_OF_RE.search(quote) is not None or bool(
        re.search(r"по\s+состоянию\s+на|остаток\s+на|\bas\s+of\b", quote, re.IGNORECASE)
    )


def clause_key(raw_clause: str, template_clauses: list[str] | None = None) -> str:
    """Canonical clause key ("6.1") from whatever the LLM returned for `clause`.

    Weak models return the full heading ("Пункт 6.1 Минимальный коэффициент …")
    instead of the bare number; submission_template.json keys are bare numbers, so
    an un-normalized key silently stubs every computed cell (run 2026-08-08).
    Preference order: a template clause that appears as a number in the raw string;
    otherwise the first number-like token; otherwise the raw string stripped.
    """
    raw = raw_clause.strip()
    numbers = _CLAUSE_NUMBER_PATTERN.findall(raw)
    if template_clauses:
        for num in numbers:
            if num in template_clauses:
                return num
        if raw in template_clauses:
            return raw
    if numbers:
        return numbers[0]
    return raw


def find_company_name(normalized_text: str, window_lines: int = 12) -> str | None:
    """Extracts the legal-entity name from the page-1 header.

    Searches the first `window_lines` non-blank lines (default widened past the
    FINDINGS §2 default of 6, to cover the ~19 docs where a "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ"
    stamp pushes the header down).
    """
    lines = [ln.strip() for ln in normalized_text.splitlines() if ln.strip()]
    for line in lines[:window_lines]:
        collapsed = collapse_letter_spacing(line)
        m = _COMPANY_LINE_PATTERN.match(collapsed)
        if m:
            return (m.group("name") or m.group("name_pref") or "").strip() or None
    return None
