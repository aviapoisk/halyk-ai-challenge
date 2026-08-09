"""Corpus-level ledger hints for LLM categorization (signal vs filler rows).

Frequencies are statistics over the full ledger, never scenario hardcodes.
Numbers are passed to the LLM as hints — not boolean cutoffs.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from agent.normalize import normalize_company_name

# Location/period tails after em/en dash or a late comma (noise marker in open set).
_TAIL_DASH = (" — ", " – ", " - ")


def first_counterparty_token(name: str) -> str:
    norm = normalize_company_name(name)
    if not norm:
        return ""
    return norm.split(" ", 1)[0]


def description_has_qualifier_tail(description: str) -> bool:
    """True when description has a location/period qualifier after — or a late comma."""
    d = description or ""
    for dash in _TAIL_DASH:
        if dash in d:
            after = d.split(dash, 1)[1].strip()
            if after:
                return True
    # Trailing ", Q1" / ", period 02" style — comma in the second half of the string.
    if "," in d:
        left, right = d.rsplit(",", 1)
        if len(left) >= 12 and right.strip():
            return True
    return False


def build_ledger_signal_index(ledger_df: pd.DataFrame) -> dict[str, object]:
    """Precompute corpus-wide and per-account frequencies once per run."""
    token_freq: Counter[str] = Counter()
    # (account_id, normalized_counterparty) -> count
    account_cp_n: Counter[tuple[str, str]] = Counter()

    if ledger_df.empty:
        return {"token_freq": token_freq, "account_cp_n": account_cp_n}

    for row in ledger_df.itertuples():
        cp = str(getattr(row, "counterparty", "") or "")
        acct = str(getattr(row, "account_id", "") or "")
        tok = first_counterparty_token(cp)
        if tok:
            token_freq[tok] += 1
        norm_cp = normalize_company_name(cp)
        if acct and norm_cp:
            account_cp_n[(acct, norm_cp)] += 1

    return {"token_freq": token_freq, "account_cp_n": account_cp_n}


def row_signal_fields(
    counterparty: str,
    description: str,
    account_id: str,
    index: dict[str, object],
) -> dict[str, int]:
    """Numeric hint fields for one ledger row (no boolean cutoff)."""
    token_freq: Counter[str] = index.get("token_freq") or Counter()  # type: ignore[assignment]
    account_cp_n: Counter[tuple[str, str]] = index.get("account_cp_n") or Counter()  # type: ignore[assignment]
    tok = first_counterparty_token(counterparty)
    norm_cp = normalize_company_name(counterparty)
    return {
        "cp_token_freq": int(token_freq.get(tok, 0)) if tok else 0,
        "desc_tail": 1 if description_has_qualifier_tail(description) else 0,
        "cp_account_n": int(account_cp_n.get((account_id, norm_cp), 0)) if account_id and norm_cp else 0,
    }
