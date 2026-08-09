"""Unit tests for submission2 fix plan (A1–A9, B3–B6) — no LLM."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.carveouts import apply_carve_outs
from agent.expr_tree import ExprContext, evaluate_expr, synthesize_expr_from_enum
from agent.s4_engine import (
    description_bucket_mask,
    evaluate_ratio_generic,
    evaluate_sum_categories_vs_group_pct,
    looks_like_dscr,
    looks_like_ebitda_over_opex_lines,
    looks_like_net_debt,
    looks_like_total_debt_leverage,
    normalize_percent_threshold,
    refine_covenant_spec,
    sum_description_bucket,
)
from agent.schemas import CarveOut, CovenantSpec, RelatedParty


def _spec(**kw) -> CovenantSpec:
    base = dict(
        clause="6.1",
        covenant_type="ratio_generic",
        comparator="max",
        threshold=1.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="",
    )
    base.update(kw)
    return CovenantSpec(**base)


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if "amount_usd" not in df.columns:
        df["amount_usd"] = df["amount"]
    return df


class TestA1PctOfGroup:
    def test_fraction_threshold(self):
        df = _df(
            [
                {
                    "txn_id": "a",
                    "date": "2025-03-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "interest",
                    "amount": -1_000_000.0,
                    "currency": "USD",
                }
            ]
        )
        cats = {"a": "interest_expense"}
        spec = _spec(
            covenant_type="sum_category",
            comparator="max",
            threshold=0.05,
            filter_categories=["interest_expense"],
            pct_of_group_capex=True,
        )
        res = evaluate_sum_categories_vs_group_pct(spec, df, cats, group_capex=40_000_000.0)
        assert res.status == "COMPLIANT"
        assert res.actual == pytest.approx(1_000_000.0)


class TestA2MoneyLinear:
    def test_revenue_minus_capex(self):
        df = _df(
            [
                {
                    "txn_id": "r",
                    "date": "2025-01-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "sales",
                    "amount": 10_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "c",
                    "date": "2025-02-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "capex",
                    "amount": -2_000_000.0,
                    "currency": "USD",
                },
            ]
        )
        cats = {"r": "revenue", "c": "capex"}
        # LLM swapped num/den — engine must still do revenue − capex.
        spec = _spec(
            threshold=6_000_000.0,
            comparator="min",
            numerator_categories=["capex"],
            denominator_categories=["revenue"],
            quote="Revenue minus Capex shall not be less than $6,000,000",
            formula_description="Revenue minus Capex",
        )
        res = evaluate_ratio_generic(spec, df, cats)
        assert res.actual == pytest.approx(8_000_000.0)
        assert res.status == "COMPLIANT"
        assert any("money_linear" in f for f in res.flags)


class TestA5Marketing:
    def test_billboard_ad_campaign(self):
        df = _df(
            [
                {
                    "txn_id": "m",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "Almaty Media Partners LLP",
                    "description": "Regional billboard ad campaign",
                    "amount": -254_812.36,
                    "currency": "USD",
                }
            ]
        )
        mask = description_bucket_mask(df, "marketing")
        assert bool(mask.iloc[0])
        total, ids = sum_description_bucket(df, "marketing")
        assert total == pytest.approx(254_812.36)
        assert ids == ["m"]


class TestA6A7A9Refine:
    def test_total_debt_leverage_detect(self):
        q = "Общий долг / EBITDA Заёмщика не более 3.25x"
        assert looks_like_total_debt_leverage(q)
        out = refine_covenant_spec(
            _spec(quote=q, formula_description=q, threshold=3.25, comparator="max")
        )
        assert out.numerator_categories == ["financing_proceeds"]

    def test_dscr_detect(self):
        q = "EBITDA / Debt Service (Interest Expense plus scheduled principal repayments) ≥ 1.25"
        assert looks_like_dscr(q)
        out = refine_covenant_spec(
            _spec(quote=q, formula_description=q, threshold=1.25, comparator="min")
        )
        assert set(out.denominator_categories or []) == {"interest_expense", "financing"}
        assert not out.numerator_categories

    def test_ebitda_over_opex_lines(self):
        q = "EBITDA / (rent + utilities + insurance) shall exceed 2.28"
        assert looks_like_ebitda_over_opex_lines(q)
        out = refine_covenant_spec(
            _spec(quote=q, formula_description=q, threshold=2.28, comparator="min")
        )
        assert set(out.denominator_categories or []) == {"rent", "utilities", "insurance"}


class TestA8SpringingRent:
    def test_refine_payroll_rent(self):
        q = (
            "Springing: if payroll exceeds 30% of revenue, then rent shall not exceed $900,000"
        )
        out = refine_covenant_spec(
            _spec(
                quote=q,
                formula_description=q,
                threshold=900_000.0,
                comparator="max",
                covenant_type="ratio_generic",
            )
        )
        assert out.covenant_type == "sum_category"
        assert out.filter_categories == ["rent"]
        assert out.springing_mode == "payroll_ratio_above"
        assert out.springing_min_numerator == pytest.approx(0.30)


class TestB3RelatedOutflows:
    def test_tree_allows_null_threshold(self):
        df = _df(
            [
                {
                    "txn_id": "p",
                    "date": "2025-04-01",
                    "account_id": "A",
                    "counterparty": "Taraz Digital Ventures LLP",
                    "description": "affiliate payment",
                    "amount": -100_000.0,
                    "currency": "USD",
                }
            ]
        )
        cats = {"p": "other"}
        spec = synthesize_expr_from_enum(
            _spec(
                covenant_type="sum_related_party_payments",
                threshold=20_000_000.0,
                comparator="max",
                quote="related-party payments",
            )
        )
        assert spec is not None
        ctx = ExprContext(
            related_parties=[
                RelatedParty(
                    counterparty_name="Taraz Digital Ventures LLP",
                    voting_rights_pct=100.0,
                    quote="аффилированное лицо, 100% дочка",
                )
            ],
            related_threshold_pct=None,
        )
        # Attach categories via evaluate path using period_df counterparty match.
        res = evaluate_expr(spec, df, cats, ctx)
        assert "missing KYC" not in " ".join(res.flags)
        assert res.actual == pytest.approx(100_000.0)


class TestB4NormalizePercent:
    def test_leverage_not_divided_by_100(self):
        spec = _spec(
            threshold=3.0,
            comparator="max",
            quote=(
                "Совокупный долг / скорректированная EBITDA не более 3.00x; "
                "исключая платежи до 5% выручки"
            ),
            formula_description="Total Debt / Adjusted EBITDA ≤ 3.00x",
        )
        assert normalize_percent_threshold(spec) is None


class TestB5NetDebt:
    def test_net_debt_detect(self):
        q = "Net Debt (drawn less scheduled repayments) / EBITDA ≤ 3.0"
        assert looks_like_net_debt(q)


class TestSubmission3LeverageSpringingCapex:
    def test_scrape_ru_dolgovoy_nagruzki(self):
        from agent.s4_engine import scrape_leverage_springing_threshold

        q = (
            "Если Коэффициент долговой нагрузки Заёмщика (отношение совокупного долга "
            "к EBITDA, определяемое согласно Пункту 5.1) превышает 2.40x, то Капитальными "
            "затратами не допускать превышения величины $5,050,000.00."
        )
        assert scrape_leverage_springing_threshold(q) == pytest.approx(2.40)

    def test_refine_b2_style(self):
        q = (
            "Если по состоянию на любую дату Коэффициент долговой нагрузки Заёмщика "
            "(отношение совокупного долга к EBITDA) превышает 3.00x, то "
            "обязуется не допускать превышения совокупными Капитальными затратами "
            "за период величины $2,500,000.00."
        )
        out = refine_covenant_spec(
            _spec(
                covenant_type="ratio_generic",
                threshold=2_500_000.0,
                comparator="max",
                quote=q,
                formula_description="Springing capex cap when Debt/EBITDA exceeds 3.00x",
            )
        )
        assert out.covenant_type == "sum_category"
        assert out.filter_categories == ["capex"]
        assert out.springing_mode == "leverage_above"
        assert out.springing_min_numerator == pytest.approx(3.0)

    def test_refine_f1_dual_default(self):
        q = (
            "Случаем неисполнения обязательств признаётся ситуация, когда Заёмщик "
            "одновременно (а) допускает превышение Коэффициентом долговой нагрузки "
            "(отношение совокупного долга к EBITDA) величины 3.50x и (б) допускает "
            "превышение совокупными Капитальными затратами величины $2,000,000.00."
        )
        out = refine_covenant_spec(
            _spec(
                covenant_type="ratio_generic",
                threshold=2_000_000.0,
                comparator="max",
                quote=q,
                formula_description=(
                    "Dual default: leverage > 3.50x AND CapEx > $2,000,000; "
                    "Reported metric is cumulative Capital Expenditures."
                ),
            )
        )
        assert out.covenant_type == "sum_category"
        assert out.filter_categories == ["capex"]
        assert out.springing_mode == "leverage_above"
        assert out.springing_min_numerator == pytest.approx(3.5)

    def test_refine_j5_group_leverage_springing(self):
        q = (
            "не допускать превышения Капитальными затратами суммы $1,800,000.00 "
            "в любой момент, когда Консолидированный коэффициент долговой нагрузки "
            "Группы превышает 3.50x."
        )
        out = refine_covenant_spec(
            _spec(
                covenant_type="ratio_group_capex_to_ebitda",
                threshold=1_800_000.0,
                comparator="max",
                quote=q,
                formula_description="Springing capex when Consolidated Leverage exceeds 3.50x",
            )
        )
        assert out.covenant_type == "sum_category"
        assert out.filter_categories == ["capex"]
        assert out.springing_mode == "leverage_above"
        assert out.springing_min_numerator == pytest.approx(3.5)

    def test_evaluate_springing_inactive_keeps_capex_actual(self):
        from agent.s4_engine import CovenantResult, apply_springing, evaluate_sum_category

        df = _df(
            [
                {
                    "txn_id": "c1",
                    "date": "2025-06-01",
                    "account_id": "A",
                    "counterparty": "C",
                    "description": "equipment",
                    "amount": -2_632_543.20,
                    "currency": "USD",
                }
            ]
        )
        cats = {"c1": "capex"}
        spec = _spec(
            covenant_type="sum_category",
            comparator="max",
            threshold=2_500_000.0,
            filter_categories=["capex"],
            springing_mode="leverage_above",
            springing_min_numerator=3.0,
            quote="capex springing",
        )
        res = evaluate_sum_category(spec, df, cats)
        assert res.status == "BREACH"
        demoted = apply_springing(spec, res, trigger_value=2.99)
        assert demoted.status == "COMPLIANT"
        assert demoted.actual == pytest.approx(2_632_543.20)

    def test_scale_mismatch_helper(self):
        from agent.s4_engine import is_scale_mismatch

        assert is_scale_mismatch(2_500_000.0, 2.99)
        assert is_scale_mismatch(3.0, 2_500_000.0)
        assert not is_scale_mismatch(3.0, 2.99)
        assert not is_scale_mismatch(2_500_000.0, 2_632_543.20)


class TestSubmission3DuplicateActualFlag:
    def test_triage_duplicate_actuals(self):
        from agent.s6_report import triage_duplicate_actuals

        submission = {
            "answers": {
                "J5": {
                    "6.1": {"status": "COMPLIANT", "actual": 0.50, "evidence_txn_id": None},
                    "6.2": {"status": "COMPLIANT", "actual": 0.50, "evidence_txn_id": None},
                    "6.3": {"status": "COMPLIANT", "actual": 100.0, "evidence_txn_id": None},
                }
            }
        }
        template = {"answers": {"J5": ["6.1", "6.2", "6.3"]}}
        flags = triage_duplicate_actuals(submission, template)
        assert any("duplicate_actual" in f and "6.1" in f and "6.2" in f for f in flags)


class TestSubmission3RelatedPartyFallback:
    def test_evaluate_with_overlap_names(self):
        from agent.s4_engine import evaluate_sum_related_party_payments

        df = _df(
            [
                {
                    "txn_id": "r1",
                    "date": "2025-05-01",
                    "account_id": "A",
                    "counterparty": "Bridgeport Logistics Corp",
                    "description": "affiliate",
                    "amount": -307_410.14,
                    "currency": "USD",
                }
            ]
        )
        cats = {"r1": "other"}
        parties = [
            RelatedParty(
                counterparty_name="Bridgeport Logistics Corp",
                voting_rights_pct=100.0,
                quote="related-party token-overlap unconfirmed fallback",
            )
        ]
        res = evaluate_sum_related_party_payments(
            _spec(
                covenant_type="sum_related_party_payments",
                threshold=275_000.0,
                comparator="max",
                quote="related-party payments ≤ $275,000",
            ),
            df,
            parties,
            None,
            cats,
        )
        assert res.actual == pytest.approx(307_410.14)
        assert res.status == "BREACH"

