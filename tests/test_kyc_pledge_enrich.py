"""Unit tests for deterministic KYC pledge-table enrich."""

from __future__ import annotations

import pandas as pd

from agent.kyc_pledge import enrich_kyc_pledge_from_text
from agent.s4_engine import apply_pledge_based_transfer_categories
from agent.schemas import KYCFacts, RelatedParty

_SYNTHETIC_KYC = """\
Связанные стороны

| Контрагент | Голосующие права |
|---|---|
| Acme Holdings JSC | 6.9% |
| Beta Capital LLP. | 39.7% |
| Gamma Haul Systems LLP | 31.4% |

Обеспечительное покрытие дочерних организаций

Ниже приведена доля активов каждой дочерней организации, переданных в залог.

| Дочерняя организация | Доля активов в залоге |
|---|---|
| North Conveyor Assets LLP | 87.6% |
| South Processing Holdings LLP | 11.4% |

Дочерние организации, у которых доля активов в залоге ниже 50.0%, находятся вне
периметра обеспечения и для целей Договора рассматриваются как неограниченные.
"""


def _ownership_only_kyc() -> KYCFacts:
    return KYCFacts(
        borrower_name="Synthetic Mining Services JSC",
        account_id="ACC-9999",
        related_party_threshold_pct=34.0,
        unrestricted_pledge_threshold_pct=50.0,
        counterparties=[
            RelatedParty(
                counterparty_name="Acme Holdings JSC",
                voting_rights_pct=6.9,
                pledged_assets_pct=None,
                quote="Acme Holdings JSC 6.9%",
            ),
            RelatedParty(
                counterparty_name="Beta Capital LLP.",
                voting_rights_pct=39.7,
                pledged_assets_pct=None,
                quote="39.7%",
            ),
            RelatedParty(
                counterparty_name="Gamma Haul Systems LLP",
                voting_rights_pct=31.4,
                pledged_assets_pct=None,
                quote="31.4%",
            ),
        ],
    )


def test_enrich_merges_pledge_table_rows():
    facts, flags = enrich_kyc_pledge_from_text(_ownership_only_kyc(), _SYNTHETIC_KYC)
    by_name = {p.counterparty_name: p for p in facts.counterparties}
    assert "North Conveyor Assets LLP" in by_name
    assert "South Processing Holdings LLP" in by_name
    assert by_name["North Conveyor Assets LLP"].pledged_assets_pct == 87.6
    assert by_name["South Processing Holdings LLP"].pledged_assets_pct == 11.4
    assert by_name["North Conveyor Assets LLP"].voting_rights_pct == 0.0
    assert any(f.startswith("kyc_pledge_table_merged n=") for f in flags)
    assert "kyc_pledge_table_unparsed" not in flags


def test_enrich_fills_threshold_when_missing():
    base = _ownership_only_kyc().model_copy(update={"unrestricted_pledge_threshold_pct": None})
    facts, flags = enrich_kyc_pledge_from_text(base, _SYNTHETIC_KYC)
    assert facts.unrestricted_pledge_threshold_pct == 50.0
    assert any(f.startswith("kyc_pledge_threshold_from_text=") for f in flags)


def test_enrich_noop_without_section():
    text = "Only ownership.\n| Acme Holdings JSC | 6.9% |\n"
    facts, flags = enrich_kyc_pledge_from_text(_ownership_only_kyc(), text)
    assert len(facts.counterparties) == 3
    assert all(p.pledged_assets_pct is None for p in facts.counterparties)
    assert flags == []


def test_enrich_noop_empty_text():
    before = _ownership_only_kyc()
    facts, flags = enrich_kyc_pledge_from_text(before, "")
    assert facts is before or facts.model_dump() == before.model_dump()
    assert flags == []


def test_enrich_does_not_overwrite_llm_pledged():
    base = _ownership_only_kyc()
    # Pre-seed one pledge row with a different value than the table.
    cps = list(base.counterparties) + [
        RelatedParty(
            counterparty_name="North Conveyor Assets LLP",
            voting_rights_pct=0.0,
            pledged_assets_pct=90.0,
            quote="llm",
        )
    ]
    base = base.model_copy(update={"counterparties": cps})
    facts, flags = enrich_kyc_pledge_from_text(base, _SYNTHETIC_KYC)
    conveyor = next(p for p in facts.counterparties if "Conveyor" in p.counterparty_name)
    assert conveyor.pledged_assets_pct == 90.0
    assert any("kyc_pledge_conflict" in f for f in flags)
    south = next(p for p in facts.counterparties if "Processing" in p.counterparty_name)
    assert south.pledged_assets_pct == 11.4


def test_enrich_then_pledge_remap():
    facts, _ = enrich_kyc_pledge_from_text(_ownership_only_kyc(), _SYNTHETIC_KYC)
    df = pd.DataFrame(
        {
            "txn_id": ["u1", "r1"],
            "amount": [-418_204.37, -302_118.64],
            "amount_usd": [-418_204.37, -302_118.64],
            "description": [
                "Transfer of capital assets to subsidiary",
                "Transfer of capital assets to subsidiary",
            ],
            "counterparty": [
                "South Processing Holdings LLP",
                "North Conveyor Assets LLP",
            ],
            "currency": ["USD", "USD"],
        }
    )
    cats = {
        "u1": "unrestricted_subsidiary_transfer",
        "r1": "unrestricted_subsidiary_transfer",
    }
    new_cats, flags, remapped = apply_pledge_based_transfer_categories(df, cats, facts)
    assert new_cats["r1"] == "other"
    assert new_cats["u1"] == "unrestricted_subsidiary_transfer"
    assert "u1" in remapped
    assert "r1" not in remapped
    assert any("Restricted" in f for f in flags)
    assert any("Unrestricted" in f for f in flags)


def test_unparsed_flag_when_markers_but_no_rows():
    text = (
        "Обеспечительное покрытие дочерних организаций\n"
        "Доля активов в залоге не раскрыта в табличной форме.\n"
        "ниже 50.0% — неограниченные.\n"
    )
    facts, flags = enrich_kyc_pledge_from_text(_ownership_only_kyc(), text)
    assert all(p.pledged_assets_pct is None for p in facts.counterparties)
    assert "kyc_pledge_table_unparsed" in flags
