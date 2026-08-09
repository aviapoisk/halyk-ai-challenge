"""C3 — localize the longest body section for an anchor (not TOC)."""

from __future__ import annotations

import re


def longest_section_around_anchor(
    text: str,
    anchor: str,
    *,
    min_chars: int = 400,
    window: int = 2500,
) -> str | None:
    """Return the longest haystack window among anchor hits (prefer body over TOC).

    Used for carve-out quote verification and future B2 — never as the sole
    extraction input (preamble dates must stay visible to S3).
    """
    if not text or not anchor:
        return None
    # Case-insensitive literal search; collapse internal whitespace in anchor for PDFs.
    parts = re.split(r"\s+", anchor.strip())
    if not parts:
        return None
    pat = re.compile(r"\s+".join(re.escape(p) for p in parts), re.IGNORECASE)
    hits = list(pat.finditer(text))
    if not hits:
        return None
    best = ""
    for m in hits:
        start = max(0, m.start() - window // 4)
        end = min(len(text), m.end() + window)
        frag = text[start:end]
        if len(frag) > len(best):
            best = frag
    if len(best) < min_chars and len(hits) == 1:
        return best or None
    return best if len(best) >= min_chars or best else None
