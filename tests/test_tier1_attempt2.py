"""Unit tests for Tier-1 attempt-2 rules (no LLM, no open-set literals in agent/)."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.schemas import CovenantSpec, RelatedParty
from agent.s4_engine import (
    apply_springing,
    evaluate_description_bucket,
    evaluate_sum_categories_vs_group_pct,
    evaluate_sum_group_capex,
    evaluate_sum_related_party_payments,
    fill_missing_amounts_from_reclassifications,
    is_absolute_group_capex_ceiling,
    looks_like_any_quarter_metric,
    reduce_over_calendar_quarters,
    refine_covenant_spec,
    scrape_missing_txn_amounts_from_notes,
    scrape_quarterly_revenue_waiver_floor,
    sum_description_bucket,
)


def _ledger(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


def _spec(**kw) -> CovenantSpec:
    base = dict(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=1.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="q",
    )
    base.update(kw)
    return CovenantSpec(**base)


class TestAbsoluteGroupCapex:
    def test_detects_monetary_ceiling(self):
        spec = _spec(
            covenant_type="ratio_group_capex_to_ebitda",
            threshold=20_000_000.0,
            quote="Капитальные затраты консолидированной Группы не более $20,000,000",
        )
        assert is_absolute_group_capex_ceiling(spec)

    def test_ratio_threshold_not_absolute(self):
        spec = _spec(
            covenant_type="ratio_group_capex_to_ebitda",
            threshold=9.0,
            quote="отношение капитальных затрат Группы к EBITDA Заёмщика не более 9.00x",
        )
        assert not is_absolute_group_capex_ceiling(spec)

    def test_absolute_breach(self):
        df = _ledger(
            [
                {
                    "txn_id": "t1",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "d",
                    "amount": -100.0,
                    "currency": "USD",
                }
            ]
        )
        spec = _spec(
            covenant_type="ratio_group_capex_to_ebitda",
            comparator="max",
            threshold=20_000_000.0,
            quote="Капитальные затраты консолидированной Группы ≤ $20,000,000",
        )
        res = evaluate_sum_group_capex(spec, df, {"t1": "capex"}, group_capex=24_679_282.33)
        assert res.status == "BREACH"
        assert res.actual == pytest.approx(24_679_282.33)

    def test_outside_borrower_subtracts_own(self):
        df = _ledger(
            [
                {
                    "txn_id": "c1",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "equipment",
                    "amount": -2_551_757.64,
                    "currency": "USD",
                }
            ]
        )
        spec = _spec(
            covenant_type="ratio_group_capex_to_ebitda",
            comparator="max",
            threshold=20_000_000.0,
            quote="Капитальные затраты Группы за пределами Заёмщика ≤ $20,000,000",
        )
        res = evaluate_sum_group_capex(
            spec, df, {"c1": "capex"}, group_capex=32_053_535.58, own_capex=2_551_757.64
        )
        assert res.status == "BREACH"
        assert res.actual == pytest.approx(29_501_777.94)


class TestFinancingAlias:
    def test_refine_aliases_financing_to_proceeds(self):
        spec = _spec(
            covenant_type="ratio_generic",
            threshold=3.0,
            quote="отношение совокупного долга (привлечённое финансирование) к EBITDA",
            numerator_categories=["financing"],
            denominator_categories=["revenue", "operating_expense"],
        )
        out = refine_covenant_spec(spec)
        assert out.numerator_categories == ["financing_proceeds"]


class TestRelatedPartyNullThreshold:
    def test_explicit_affiliate_included(self):
        df = _ledger(
            [
                {
                    "txn_id": "p1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "Taraz Digital Ventures LLP",
                    "description": "advisory",
                    "amount": -100_000.0,
                    "currency": "USD",
                }
            ]
        )
        parties = [
            RelatedParty(
                counterparty_name="Taraz Digital Ventures LLP",
                voting_rights_pct=100.0,
                quote="классифицирован как АФФИЛИРОВАННОЕ ЛИЦО Заёмщика, дочерняя организация",
            )
        ]
        spec = _spec(
            covenant_type="sum_related_party_payments",
            comparator="max",
            threshold=20_000_000.0,
            quote="платежи связанным сторонам",
        )
        res = evaluate_sum_related_party_payments(spec, df, parties, None)
        assert res.actual == pytest.approx(100_000.0)
        assert res.status == "COMPLIANT"

    def test_zero_pct_without_label_excluded(self):
        df = _ledger(
            [
                {
                    "txn_id": "p1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "Altyn Capital L.L.P.",
                    "description": "payment",
                    "amount": -50_000.0,
                    "currency": "USD",
                }
            ]
        )
        parties = [
            RelatedParty(
                counterparty_name="Altyn Capital L.L.P.",
                voting_rights_pct=0.0,
                quote="",
            )
        ]
        spec = _spec(
            covenant_type="sum_related_party_payments",
            comparator="max",
            threshold=275_000.0,
            quote="платежи связанным сторонам",
        )
        res = evaluate_sum_related_party_payments(spec, df, parties, None)
        assert res.inputs["payments_sum"] == pytest.approx(0.0)
        assert res.status == "COMPLIANT"


class TestQuarterlyReduce:
    def test_min_quarter_revenue(self):
        assert looks_like_any_quarter_metric(
            "Выручка за любой финансовый квартал не ниже $3,000,000"
        )
        df = _ledger(
            [
                {
                    "txn_id": "r1",
                    "date": "2025-02-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "sales",
                    "amount": 5_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "r2",
                    "date": "2025-05-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "sales",
                    "amount": 4_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "r3",
                    "date": "2025-08-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "sales",
                    "amount": 3_106_656.60,
                    "currency": "USD",
                },
                {
                    "txn_id": "r4",
                    "date": "2025-11-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "sales",
                    "amount": 4_500_000.0,
                    "currency": "USD",
                },
            ]
        )
        cats = {f"r{i}": "revenue" for i in range(1, 5)}

        def metric(qdf):
            from agent.s4_engine import _sum_cats

            v, ids = _sum_cats(qdf, cats, ["revenue"])
            return float(v), ids, []

        val, _, flags = reduce_over_calendar_quarters(
            df, "2025-01-01", "2025-12-31", "min", metric
        )
        assert val == pytest.approx(3_106_656.60)
        assert any("quarter_reduce=min" in f for f in flags)

    def test_refine_sets_quarter_reduce(self):
        spec = _spec(
            covenant_type="sum_revenue",
            comparator="min",
            threshold=3_000_000.0,
            quote="Выручка за любой финансовый квартал не ниже $3,000,000",
        )
        out = refine_covenant_spec(spec)
        assert out.quarter_reduce == "min"


class TestMarketingBucket:
    def test_sums_marketing_outflows_skips_refund(self):
        df = _ledger(
            [
                {
                    "txn_id": "m1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "Brand marketing campaign Q1",
                    "amount": -100_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "m2",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "Marketing overbilling refund",
                    "amount": 50_000.0,
                    "currency": "USD",
                },
            ]
        )
        total, ids = sum_description_bucket(df, "marketing")
        assert total == pytest.approx(100_000.0)
        assert ids == ["m1"]

    def test_refine_clears_other_filter(self):
        spec = _spec(
            covenant_type="max_category",
            comparator="max",
            threshold=600_000.0,
            filter_categories=["other"],
            quote='расходами по статье "Маркетинговые расходы"',
        )
        out = refine_covenant_spec(spec)
        assert out.description_bucket == "marketing"
        assert "other" not in (out.filter_categories or [])

    def test_truncated_quote_recovers_marketing_from_document(self):
        doc = (
            'под расходами по статье "Маркетинговые расходы" понимаются суммы, '
            "отнесённые к данной статье. Заёмщик не вправе допускать, чтобы "
            "такие совокупные расходы за период превышали $600,000."
        )
        spec = _spec(
            covenant_type="max_category",
            comparator="max",
            threshold=600_000.0,
            filter_categories=["other"],
            quote="чтобы такие совокупные расходы за период превышали $600,000",
        )
        out = refine_covenant_spec(spec, document_text=doc)
        assert out.description_bucket == "marketing"


class TestSpringing:
    def test_own_capex_not_triggered_keeps_requirement_actual(self):
        from agent.s4_engine import CovenantResult

        spec = _spec(
            covenant_type="sum_category",
            comparator="min",
            threshold=250_000.0,
            filter_categories=["insurance"],
            quote="Если капитальные затраты превышают $1,500,000, страховые премии не менее $250,000",
            springing_min_numerator=1_500_000.0,
            springing_mode="own_capex_above",
        )
        result = CovenantResult(
            clause="6.2",
            status="BREACH",
            actual=100_000.0,
            contributing_txn_ids=[],
            flags=[],
            inputs={"total": 100_000.0, "numerator": 100_000.0},
        )
        out = apply_springing(spec, result, trigger_value=1_000_000.0)
        assert out.status == "COMPLIANT"
        assert out.actual == pytest.approx(100_000.0)

    def test_insurance_floor_refine(self):
        spec = _spec(
            covenant_type="unknown",
            comparator="min",
            threshold=250_000.0,
            formula_description="Springing insurance",
            quote=(
                "Если совокупные капитальные затраты Заёмщика превышают $1,500,000.00, "
                "Заёмщик обязуется поддерживать страховые премии в размере не менее $250,000.00"
            ),
            springing_min_numerator=1_500_000.0,
        )
        out = refine_covenant_spec(spec)
        assert out.covenant_type == "sum_category"
        assert out.filter_categories == ["insurance"]
        assert out.springing_mode == "own_capex_above"

    def test_revenue_waiver_scrape_and_refine(self):
        doc = (
            "отношение Капитальных затрат к Выручке значения 0.32x. "
            "До тех пор пока Выручка за каждый финансовый квартал, оканчивающийся в "
            "течение периода, составляет не менее $4,000,000.00, изложенное ограничение "
            "Капитальных затрат не применяется."
        )
        assert scrape_quarterly_revenue_waiver_floor(doc) == pytest.approx(4_000_000.0)
        spec = _spec(
            covenant_type="ratio_group_capex_to_ebitda",
            comparator="max",
            threshold=0.32,
            quote="отношение Капитальных затрат к Выручке за указанный период в целом значения 0.32x",
            formula_description="Капитальные затраты / Выручка",
            numerator_categories=["capex"],
            denominator_categories=["revenue"],
        )
        out = refine_covenant_spec(spec, document_text=doc)
        assert out.covenant_type == "ratio_generic"
        assert out.numerator_categories == ["capex"]
        assert out.denominator_categories == ["revenue"]
        assert out.springing_mode == "min_quarter_revenue_below"
        assert out.springing_min_numerator == pytest.approx(4_000_000.0)

    def test_min_quarter_revenue_below_not_triggered(self):
        from agent.s4_engine import CovenantResult

        spec = _spec(
            springing_min_numerator=4_000_000.0,
            springing_mode="min_quarter_revenue_below",
            threshold=0.32,
        )
        result = CovenantResult(
            clause="6.4",
            status="BREACH",
            actual=0.39,
            contributing_txn_ids=[],
            flags=[],
            inputs={"value_raw": 0.39},
        )
        out = apply_springing(spec, result, trigger_value=4_650_129.31)
        assert out.status == "COMPLIANT"


class TestTreasuryAmountFill:
    def test_scrape_and_fill_nan_amount(self):
        note = (
            "Операция TXN-AA-0029 (KTZ Freight Services JSC): сумма не отражена в выгрузке "
            "реестра; фактическая сумма операции составляет $6,082,920.03 (поступление)."
        )
        found = scrape_missing_txn_amounts_from_notes([note])
        assert found and found[0].txn_id.upper().endswith("0029")
        assert found[0].amount == pytest.approx(6_082_920.03)
        df = _ledger(
            [
                {
                    "txn_id": found[0].txn_id,
                    "date": "2025-09-30",
                    "account_id": "A",
                    "counterparty": "KTZ",
                    "description": "Freight haulage sales settlement Q3",
                    "amount": float("nan"),
                    "currency": "USD",
                }
            ]
        )
        filled, flags = fill_missing_amounts_from_reclassifications(df, found)
        assert filled.iloc[0]["amount"] == pytest.approx(6_082_920.03)
        assert any("filled" in f for f in flags)


class TestPctOfGroupCapex:
    def test_sum_vs_pct_limit(self):
        df = _ledger(
            [
                {
                    "txn_id": "i1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "interest",
                    "amount": -1_128_378.65,
                    "currency": "USD",
                },
                {
                    "txn_id": "r1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "rent",
                    "amount": -412_648.46,
                    "currency": "USD",
                },
            ]
        )
        cats = {"i1": "interest_expense", "r1": "rent"}
        spec = _spec(
            covenant_type="sum_category",
            comparator="max",
            threshold=5.0,
            filter_categories=["interest_expense", "rent"],
            quote="расходы не должны превышать 5% капитальных затрат Группы",
            pct_of_group_capex=True,
        )
        res = evaluate_sum_categories_vs_group_pct(spec, df, cats, group_capex=32_053_535.58)
        assert res.status == "COMPLIANT"
        assert res.actual == pytest.approx(1_541_027.11)

    def test_pct_of_group_accepts_fraction_threshold(self):
        """LLM may emit 0.05 instead of 5.0 — must not divide by 100 again."""
        df = pd.DataFrame(
            [
                {
                    "txn_id": "i1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "interest",
                    "amount": -1_128_378.65,
                    "currency": "USD",
                },
                {
                    "txn_id": "r1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "rent",
                    "amount": -412_648.46,
                    "currency": "USD",
                },
            ]
        )
        cats = {"i1": "interest_expense", "r1": "rent"}
        spec = _spec(
            covenant_type="sum_category",
            comparator="max",
            threshold=0.05,
            filter_categories=["interest_expense", "rent"],
            quote="расходы не должны превышать 5% капитальных затрат Группы",
            pct_of_group_capex=True,
        )
        res = evaluate_sum_categories_vs_group_pct(spec, df, cats, group_capex=32_053_535.58)
        assert res.status == "COMPLIANT"
        assert res.actual == pytest.approx(1_541_027.11)
        assert any("limit=1602676" in f or "limit=1602676.77" in f for f in res.flags) or any(
            "1602676" in f for f in res.flags
        )

    def test_refine_sets_pct_flag(self):
        spec = _spec(
            covenant_type="unknown",
            comparator="max",
            threshold=5.0,
            filter_categories=["interest_expense", "rent"],
            quote=(
                "Совокупные расходы на финансирование и содержание помещений "
                "не должны превышать 5% капитальных затрат Группы"
            ),
            formula_description="interest_expense + rent vs 5% group capex",
        )
        out = refine_covenant_spec(spec)
        assert out.pct_of_group_capex is True
        assert out.covenant_type == "sum_category"


class TestUnrestrictedAbsolute:
    def test_refine_absolute_cap(self):
        spec = _spec(
            covenant_type="ratio_generic",
            comparator="max",
            threshold=250_000.0,
            quote=(
                "Borrower shall not permit transfers to Unrestricted Subsidiaries "
                "to exceed $250,000 in the aggregate"
            ),
        )
        out = refine_covenant_spec(spec)
        assert out.covenant_type == "sum_category"
        assert out.filter_categories == ["unrestricted_subsidiary_transfer"]

    def test_sum_breaches(self):
        from agent.s4_engine import evaluate_sum_category

        df = _ledger(
            [
                {
                    "txn_id": "u1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "Sub",
                    "description": "Transfer to unrestricted subsidiary",
                    "amount": -2_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "u2",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "Sub",
                    "description": "Transfer to unrestricted subsidiary",
                    "amount": -1_589_760.41,
                    "currency": "USD",
                },
            ]
        )
        cats = {"u1": "unrestricted_subsidiary_transfer", "u2": "unrestricted_subsidiary_transfer"}
        spec = _spec(
            covenant_type="sum_category",
            comparator="max",
            threshold=250_000.0,
            filter_categories=["unrestricted_subsidiary_transfer"],
            quote="transfers to Unrestricted Subsidiaries ≤ $250,000",
        )
        res = evaluate_sum_category(spec, df, cats)
        assert res.status == "BREACH"
        assert res.actual == pytest.approx(3_589_760.41)

    def test_leverage_springing_inactive(self):
        from agent.s4_engine import CovenantResult, apply_springing

        spec = _spec(
            covenant_type="sum_category",
            comparator="max",
            threshold=500_000.0,
            filter_categories=["unrestricted_subsidiary_transfer"],
            quote=(
                "If Net Leverage Ratio exceeds 2.5x, transfers to the Group "
                "shall not exceed $500,000"
            ),
            springing_min_numerator=2.5,
            springing_mode="leverage_above",
        )
        out = refine_covenant_spec(
            _spec(
                covenant_type="unknown",
                comparator="max",
                threshold=500_000.0,
                quote=spec.quote,
            )
        )
        assert out.springing_mode == "leverage_above"
        assert out.springing_min_numerator == pytest.approx(2.5)
        result = CovenantResult(
            clause="5.1",
            status="BREACH",
            actual=1_643_165.84,
            contributing_txn_ids=[],
            flags=[],
            inputs={"total": 1_643_165.84},
        )
        demoted = apply_springing(spec, result, trigger_value=2.0)
        assert demoted.status == "COMPLIANT"
        assert demoted.actual == pytest.approx(1_643_165.84)


class TestEbitdaPlusFinancing:
    def test_refine_and_evaluate(self):
        from agent.s4_engine import evaluate_ebitda_plus_categories

        spec = _spec(
            covenant_type="ratio_generic",
            comparator="min",
            threshold=8_000_000.0,
            quote=(
                "EBITDA plus financing proceeds shall not be less than $8,000,000"
            ),
            formula_description="EBITDA + financing proceeds",
        )
        out = refine_covenant_spec(spec)
        assert out.ebitda_plus_categories == ["financing_proceeds"]
        df = _ledger(
            [
                {
                    "txn_id": "r1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "sales",
                    "amount": 10_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "o1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "opex",
                    "amount": -2_387_115.59,
                    "currency": "USD",
                },
                {
                    "txn_id": "f1",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "Bank",
                    "description": "facility proceeds",
                    "amount": 5_264_338.19,
                    "currency": "USD",
                },
            ]
        )
        cats = {
            "r1": "revenue",
            "o1": "operating_expense",
            "f1": "financing_proceeds",
        }
        res = evaluate_ebitda_plus_categories(out, df, cats)
        assert res.status == "COMPLIANT"
        assert res.actual == pytest.approx(12_877_222.60)


class TestPercentThreshold:
    def test_normalize_80_to_fraction(self):
        spec = _spec(
            covenant_type="ratio_generic",
            comparator="max",
            threshold=80.0,
            numerator_categories=["capex"],
            denominator_categories=["revenue"],
            quote="Капитальные затраты не должны превышать 80% Выручки",
        )
        out = refine_covenant_spec(spec)
        assert out.threshold == pytest.approx(0.80)


class TestMarketingReclass:
    def test_infer_and_tag_description(self):
        from agent.schemas import Reclassification
        from agent.s4_engine import apply_reclassifications, infer_to_category_from_quote

        quote = "Сумма переклассифицирована как маркетинговые расходы по статье"
        assert infer_to_category_from_quote(quote) == "other"
        df = _ledger(
            [
                {
                    "txn_id": "t1",
                    "date": "2025-04-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "Misc settlement",
                    "amount": -50_000.0,
                    "currency": "USD",
                }
            ]
        )
        rec = Reclassification(
            action="reclassify",
            txn_id="t1",
            amount=50_000.0,
            to_category=None,
            quote=quote,
        )
        cats, out, flags, _keys, _unm = apply_reclassifications({"t1": "other"}, df, [rec])
        assert cats["t1"] == "other"
        assert "marketing" in str(out.iloc[0]["description"]).casefold()
        assert any("marketing reclass" in f for f in flags)

    def test_advisory_needle_only_when_quote_names_it(self):
        from agent.s4_engine import sum_description_bucket

        df = _ledger(
            [
                {
                    "txn_id": "a1",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "Brand advisory engagement Q1",
                    "amount": -100_000.0,
                    "currency": "USD",
                }
            ]
        )
        total_plain, _ = sum_description_bucket(df, "marketing")
        assert total_plain == pytest.approx(0.0)
        total_named, ids = sum_description_bucket(
            df, "marketing", 'статья "Маркетинговые расходы" включая brand advisory'
        )
        assert total_named == pytest.approx(100_000.0)
        assert ids == ["a1"]
