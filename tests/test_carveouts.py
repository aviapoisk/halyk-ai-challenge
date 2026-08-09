"""B10 carve-out unit tests — no LLM."""

from __future__ import annotations

import pandas as pd

from agent.carveouts import apply_carve_outs, suspect_carve_outs, verify_carve_out_quotes
from agent.s4_engine import evaluate_sum_category
from agent.schemas import CarveOut, CovenantSpec


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _spec(**kw) -> CovenantSpec:
    base = dict(
        clause="6.2",
        covenant_type="sum_category",
        comparator="max",
        threshold=1000.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="сумма capex за исключением платежей контрагенту Acme LLP",
        filter_category="capex",
    )
    base.update(kw)
    return CovenantSpec(**base)


def test_carve_out_flips_breach_to_compliant():
    df = _df(
        [
            {
                "txn_id": "TXN-X-1",
                "date": "2025-06-01",
                "account_id": "ACC-1",
                "counterparty": "Acme LLP",
                "description": "capex",
                "amount": -800.0,
                "currency": "USD",
                "amount_usd": -800.0,
            },
            {
                "txn_id": "TXN-X-2",
                "date": "2025-06-02",
                "account_id": "ACC-1",
                "counterparty": "Other JSC",
                "description": "capex",
                "amount": -300.0,
                "currency": "USD",
                "amount_usd": -300.0,
            },
        ]
    )
    cats = {"TXN-X-1": "capex", "TXN-X-2": "capex"}
    quote = "за исключением платежей контрагенту Acme LLP"
    spec = _spec(
        carve_outs=[
            CarveOut(kind="counterparty", value="Acme LLP", quote=quote),
        ],
        quote=f"сумма capex {quote}",
    )
    # Without carve-out: 1100 > 1000 → BREACH
    assert evaluate_sum_category(spec, df, cats).status == "BREACH"
    filtered, excluded, flags = apply_carve_outs(spec, df, cats)
    assert excluded == ["TXN-X-1"]
    assert evaluate_sum_category(spec, filtered, cats).status == "COMPLIANT"


def test_carve_out_matched_nothing():
    df = _df(
        [
            {
                "txn_id": "TXN-X-1",
                "date": "2025-06-01",
                "account_id": "ACC-1",
                "counterparty": "Other JSC",
                "description": "capex",
                "amount": -2000.0,
                "currency": "USD",
                "amount_usd": -2000.0,
            },
        ]
    )
    cats = {"TXN-X-1": "capex"}
    quote = "за исключением платежей контрагенту Missing Co LLP"
    spec = _spec(
        carve_outs=[CarveOut(kind="counterparty", value="Missing Co LLP", quote=quote)],
        quote=quote,
    )
    _, excluded, flags = apply_carve_outs(spec, df, cats)
    assert excluded == []
    assert any("CARVEOUT_MATCHED_NOTHING" in f for f in flags)


def test_carve_out_quote_not_in_document_dropped():
    spec = _spec(
        carve_outs=[
            CarveOut(
                kind="counterparty",
                value="Acme LLP",
                quote="за исключением платежей контрагенту Acme LLP",
            )
        ]
    )
    cleaned, flags = verify_carve_out_quotes(spec, "документ без этой оговорки совсем")
    assert cleaned.carve_outs == []
    assert any("CARVEOUT_QUOTE_NOT_FOUND" in f for f in flags)


def test_carveout_suspected_when_hint_without_list():
    spec = _spec(carve_outs=[], quote="лимит capex за исключением связанных сторон")
    assert suspect_carve_outs(spec) == ["CARVEOUT_SUSPECTED"]
