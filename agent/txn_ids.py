"""Parse ledger txn_id → scenario_id, resolved against submission_template keys.

Private sets may use multi-segment ids (e.g. TXN-KC-CAP-29) where the template
key is only the leading segment (KC). Open-set TXN-P5-0001 stays P5.
"""

from __future__ import annotations

import re

# Trailing -<digits> is the sequence; everything between TXN- and that is "rest".
_TXN_RE = re.compile(r"^TXN-(?P<rest>.+)-(?P<seq>\d+)$")

# Auditor / note scrape: allow hyphenated middle segments (KC-CAP, …).
TXN_ID_IN_TEXT = re.compile(r"TXN-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+", re.IGNORECASE)


def parse_txn_scenario(
    txn_id: str,
    known_scenarios: set[str] | frozenset[str] | None = None,
) -> str | None:
    """Return scenario_id for a txn_id, or None if the string is not TXN-shaped.

    When ``known_scenarios`` is provided (template ``answers`` keys), pick the
    longest key ``k`` such that ``rest == k`` or ``rest.startswith(k + "-")``.
    Otherwise return the full ``rest`` (legacy greedy behaviour).
    """
    m = _TXN_RE.match((txn_id or "").strip())
    if not m:
        return None
    rest = m.group("rest")
    if known_scenarios:
        candidates = [k for k in known_scenarios if rest == k or rest.startswith(f"{k}-")]
        if candidates:
            return max(candidates, key=len)
    return rest
