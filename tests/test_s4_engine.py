import pandas as pd
import pytest

from agent.s4_engine import (
    apply_fx,
    apply_reclassifications,
    category_summary,
    compute_group_capex,
    evaluate_ratio_generic,
    evaluate_ratio_group_capex_to_ebitda,
    evaluate_sum_category,
    evaluate_sum_related_party_payments,
    evaluate_sum_revenue,
    filter_period,
    resolve_fx_rate,
    scenario_ledger,
)
from agent.schemas import CovenantSpec, FxRateFacts, GroupCapexFacts, Reclassification, RelatedParty


def make_ledger(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


def make_spec(clause="6.1", covenant_type="sum_revenue", comparator="min", threshold=1000.0):
    return CovenantSpec(
        clause=clause,
        covenant_type=covenant_type,
        comparator=comparator,
        threshold=threshold,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="q",
    )


class TestScenarioAndPeriodFilters:
    def test_scenario_ledger_filters_by_account(self):
        df = make_ledger(
            [
                {"txn_id": "TXN-P1-0001", "date": "2025-01-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "TXN-P2-0001", "date": "2025-01-01", "account_id": "ACC-2", "counterparty": "B", "description": "d", "amount": 200.0, "currency": "USD"},
            ]
        )
        out = scenario_ledger(df, {"ACC-1"})
        assert list(out["txn_id"]) == ["TXN-P1-0001"]

    def test_filter_period_is_inclusive_of_boundaries(self):
        df = make_ledger(
            [
                {"txn_id": "a", "date": "2025-01-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1.0, "currency": "USD"},
                {"txn_id": "b", "date": "2025-12-31", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1.0, "currency": "USD"},
                {"txn_id": "c", "date": "2024-12-31", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1.0, "currency": "USD"},
                {"txn_id": "d", "date": "2026-01-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1.0, "currency": "USD"},
            ]
        )
        out = filter_period(df, "2025-01-01", "2025-12-31")
        assert set(out["txn_id"]) == {"a", "b"}


class TestEvaluateSumRevenue:
    def test_below_min_threshold_is_breach(self):
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 500.0, "currency": "USD"}]
        )
        spec = make_spec(comparator="min", threshold=1000.0)
        result = evaluate_sum_revenue(spec, df, {"t1": "revenue"})
        assert result.status == "BREACH"
        assert result.actual == 500.0
        assert result.contributing_txn_ids == ["t1"]

    def test_at_or_above_min_threshold_is_compliant(self):
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1500.0, "currency": "USD"}]
        )
        spec = make_spec(comparator="min", threshold=1000.0)
        result = evaluate_sum_revenue(spec, df, {"t1": "revenue"})
        assert result.status == "COMPLIANT"

    def test_non_revenue_rows_excluded_from_sum(self):
        df = make_ledger(
            [
                {"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1500.0, "currency": "USD"},
                {"txn_id": "t2", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 900.0, "currency": "USD"},
            ]
        )
        spec = make_spec(comparator="min", threshold=1000.0)
        result = evaluate_sum_revenue(spec, df, {"t1": "revenue", "t2": "other"})
        assert result.actual == 1500.0
        assert result.contributing_txn_ids == ["t1"]


class TestEvaluateSumRelatedPartyPayments:
    def test_matches_name_with_trailing_period_and_location_suffix(self):
        """Regression case modeled on real data: KYC dossier lists "Sarybel Capital LLP."
        (trailing period), ledger counterparty carries a location suffix in parens."""
        df = make_ledger(
            [
                {"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "Sarybel Capital LLP (Almaty office)", "description": "d", "amount": -300.0, "currency": "USD"},
                {"txn_id": "t2", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "Unrelated Vendor LLP", "description": "d", "amount": -900.0, "currency": "USD"},
            ]
        )
        spec = make_spec(clause="6.3", covenant_type="sum_related_party_payments", comparator="max", threshold=200.0)
        parties = [RelatedParty(counterparty_name="Sarybel Capital LLP.", voting_rights_pct=41.2, quote="q")]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0)
        assert result.status == "BREACH"
        assert result.actual == 300.0
        assert result.contributing_txn_ids == ["t1"]

    def test_below_ownership_threshold_not_counted_as_related(self):
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "Minor Holder LLP", "description": "d", "amount": -300.0, "currency": "USD"}]
        )
        spec = make_spec(clause="6.3", covenant_type="sum_related_party_payments", comparator="max", threshold=200.0)
        parties = [RelatedParty(counterparty_name="Minor Holder LLP", voting_rights_pct=10.0, quote="q")]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0)
        assert result.status == "COMPLIANT"
        assert result.actual == 0.0

    def test_positive_amounts_from_related_party_not_counted_as_payments(self):
        """Clause 6.3 is about payments *to* related parties (outflows); an inbound
        credit from the same counterparty is not a payment and must not count."""
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "Sarybel Capital LLP", "description": "refund", "amount": 300.0, "currency": "USD"}]
        )
        spec = make_spec(clause="6.3", covenant_type="sum_related_party_payments", comparator="max", threshold=200.0)
        parties = [RelatedParty(counterparty_name="Sarybel Capital LLP", voting_rights_pct=41.2, quote="q")]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0)
        assert result.status == "COMPLIANT"
        assert result.actual == 0.0

    def test_dotted_llp_and_quoted_names_match_ledger(self):
        df = make_ledger(
            [
                {
                    "txn_id": "t1",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Aktau Holdings LLP",
                    "description": "d",
                    "amount": -100.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "t2",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Turan Capital LLP",
                    "description": "d",
                    "amount": -50.0,
                    "currency": "USD",
                },
            ]
        )
        spec = make_spec(clause="6.3", covenant_type="sum_related_party_payments", comparator="max", threshold=200.0)
        parties = [
            RelatedParty(counterparty_name="Aktau Holdings L.L.P.", voting_rights_pct=40.0, quote="q"),
            RelatedParty(counterparty_name='"Turan Capital" LLP', voting_rights_pct=40.0, quote="q"),
        ]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0)
        assert result.actual == 150.0
        assert set(result.contributing_txn_ids) == {"t1", "t2"}

    def test_threshold_below_one_divides_by_revenue_by_default(self):
        df = make_ledger(
            [
                {
                    "txn_id": "pay",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Sarybel Capital LLP",
                    "description": "d",
                    "amount": -280000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "rev",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Customer",
                    "description": "sales",
                    "amount": 7000000.0,
                    "currency": "USD",
                },
            ]
        )
        cats = {"pay": "other", "rev": "revenue"}
        spec = make_spec(
            clause="6.3",
            covenant_type="sum_related_party_payments",
            comparator="max",
            threshold=0.04,
        )
        # Quote names revenue (open-set «Proportion of Revenue» clauses).
        spec = spec.model_copy(update={"quote": "Maximum Related-Party Payments as a Proportion of Revenue 0.04x от выручки"})
        parties = [RelatedParty(counterparty_name="Sarybel Capital LLP", voting_rights_pct=41.2, quote="q")]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0, categories=cats)
        assert result.actual == 0.04
        assert result.status == "COMPLIANT"
        assert any("revenue" in f for f in result.flags)

    def test_threshold_below_one_divides_by_opex_when_quote_says_so(self):
        df = make_ledger(
            [
                {
                    "txn_id": "pay",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Sarybel Capital LLP",
                    "description": "d",
                    "amount": -271884.36,
                    "currency": "USD",
                },
                {
                    "txn_id": "ox",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Vendor",
                    "description": "opex",
                    "amount": -7022667.65,
                    "currency": "USD",
                },
            ]
        )
        cats = {"pay": "other", "ox": "operating_expense"}
        spec = make_spec(clause="6.1", covenant_type="sum_related_party_payments", comparator="max", threshold=0.08)
        spec = spec.model_copy(
            update={"quote": "доля платежей связанным сторонам в операционных расходах не более 0.08x"}
        )
        parties = [RelatedParty(counterparty_name="Sarybel Capital LLP", voting_rights_pct=41.2, quote="q")]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0, categories=cats)
        assert result.actual == 0.04
        assert result.status == "COMPLIANT"
        assert any("operating_expense" in f for f in result.flags)

    def test_threshold_at_or_above_one_stays_absolute(self):
        df = make_ledger(
            [
                {
                    "txn_id": "pay",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Sarybel Capital LLP",
                    "description": "d",
                    "amount": -300.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "ox",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Vendor",
                    "description": "opex",
                    "amount": -10000.0,
                    "currency": "USD",
                },
            ]
        )
        cats = {"pay": "other", "ox": "operating_expense"}
        spec = make_spec(clause="6.3", covenant_type="sum_related_party_payments", comparator="max", threshold=200.0)
        parties = [RelatedParty(counterparty_name="Sarybel Capital LLP", voting_rights_pct=41.2, quote="q")]
        result = evaluate_sum_related_party_payments(spec, df, parties, threshold_pct=35.0, categories=cats)
        assert result.actual == 300.0
        assert not any("related_party_as_ratio" in f for f in result.flags)


class TestEvaluateRatioGroupCapexToEbitda:
    def test_ratio_within_threshold_is_compliant(self):
        df = make_ledger(
            [
                {"txn_id": "rev", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1000.0, "currency": "USD"},
                {"txn_id": "opex", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -100.0, "currency": "USD"},
            ]
        )
        # EBITDA = 1000 - 100 = 900; capex 1800 -> ratio 2.0 <= threshold 9.0
        spec = make_spec(clause="6.1", covenant_type="ratio_group_capex_to_ebitda", comparator="max", threshold=9.0)
        result = evaluate_ratio_group_capex_to_ebitda(spec, df, {"rev": "revenue", "opex": "operating_expense"}, group_capex=1800.0)
        assert result.status == "COMPLIANT"
        assert result.actual == pytest.approx(2.0)
        assert result.inputs["ebitda"] == pytest.approx(900.0)
        assert result.inputs["revenue_sum"] == pytest.approx(1000.0)
        assert result.inputs["group_capex"] == pytest.approx(1800.0)

    def test_ratio_exceeding_threshold_is_breach(self):
        df = make_ledger(
            [
                {"txn_id": "rev", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "opex", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -10.0, "currency": "USD"},
            ]
        )
        # EBITDA = 90; capex 1800 -> ratio 20.0 > threshold 9.0
        spec = make_spec(clause="6.1", covenant_type="ratio_group_capex_to_ebitda", comparator="max", threshold=9.0)
        result = evaluate_ratio_group_capex_to_ebitda(spec, df, {"rev": "revenue", "opex": "operating_expense"}, group_capex=1800.0)
        assert result.status == "BREACH"
        assert result.actual == pytest.approx(20.0)

    def test_zero_ebitda_flags_and_does_not_raise(self):
        df = make_ledger(
            [
                {"txn_id": "rev", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "opex", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -100.0, "currency": "USD"},
            ]
        )
        spec = make_spec(clause="6.1", covenant_type="ratio_group_capex_to_ebitda", comparator="max", threshold=9.0)
        result = evaluate_ratio_group_capex_to_ebitda(spec, df, {"rev": "revenue", "opex": "operating_expense"}, group_capex=1800.0)
        assert result.flags
        assert "EBITDA <= 0" in result.flags[0]
        # V4: never write threshold into actual (false COMPLIANT).
        assert result.actual == 1800.0
        assert result.actual != 9.0
        assert result.status == "BREACH"

    def test_negative_ebitda_is_breach_not_false_compliant(self):
        df = make_ledger(
            [
                {"txn_id": "rev", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 50.0, "currency": "USD"},
                {"txn_id": "opex", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -100.0, "currency": "USD"},
            ]
        )
        spec = make_spec(clause="6.1", covenant_type="ratio_group_capex_to_ebitda", comparator="max", threshold=9.0)
        result = evaluate_ratio_group_capex_to_ebitda(spec, df, {"rev": "revenue", "opex": "operating_expense"}, group_capex=50.0)
        assert "EBITDA <= 0" in result.flags[0]
        assert result.actual == 50.0
        assert result.actual != 9.0
        assert result.status == "BREACH"


class TestApplyFx:
    def test_converts_eur_when_rate_present(self):
        df = make_ledger(
            [
                {"txn_id": "u", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "e", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -50.0, "currency": "EUR"},
            ]
        )
        out, flags = apply_fx(df, 1.16)
        assert out.loc[out["txn_id"] == "u", "amount_usd"].iloc[0] == 100.0
        assert out.loc[out["txn_id"] == "e", "amount_usd"].iloc[0] == pytest.approx(-58.0)
        assert any("converted" in f for f in flags)

    def test_missing_rate_flags_and_leaves_amounts(self):
        df = make_ledger(
            [{"txn_id": "e", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -50.0, "currency": "EUR"}]
        )
        out, flags = apply_fx(df, None)
        assert out["amount_usd"].iloc[0] == -50.0
        assert any("FX rate missing" in f for f in flags)


class TestResolveFxRate:
    def test_named_rate(self):
        rate, flags = resolve_fx_rate(FxRateFacts(eur_usd_rate=1.16, quote="q"))
        assert rate == pytest.approx(1.16)
        assert flags == []

    def test_pair_division(self):
        # 83690.23 / 72146.75 ≈ 1.16
        rate, flags = resolve_fx_rate(
            FxRateFacts(eur_usd_rate=None, eur_amount=72146.75, usd_amount=83690.23, quote="pair")
        )
        assert rate == pytest.approx(83690.23 / 72146.75)
        assert any("derived from amount pair" in f for f in flags)

    def test_conflict_prefers_named(self):
        rate, flags = resolve_fx_rate(
            FxRateFacts(eur_usd_rate=1.16, eur_amount=100.0, usd_amount=200.0, quote="both")
        )
        assert rate == pytest.approx(1.16)
        assert any("conflict" in f.lower() for f in flags)

    def test_missing(self):
        rate, flags = resolve_fx_rate(FxRateFacts(eur_usd_rate=None, quote=""))
        assert rate is None


class TestComputeGroupCapex:
    def test_roll_forward(self):
        facts = GroupCapexFacts(
            parent_company_name="HoldCo",
            opening_nbv=100.0,
            closing_nbv=150.0,
            depreciation=20.0,
            disposals=0.0,
            derivation="rf",
            quote="q",
        )
        # 150 - 100 + 20 + 0 = 70
        value, flags = compute_group_capex(facts)
        assert value == pytest.approx(70.0)
        assert any("roll-forward" in f for f in flags)

    def test_direct_preferred_on_mismatch(self):
        facts = GroupCapexFacts(
            parent_company_name="HoldCo",
            opening_nbv=100.0,
            closing_nbv=150.0,
            depreciation=20.0,
            disposals=0.0,
            direct_capex_line=80.0,
            derivation="both",
            quote="q",
        )
        value, flags = compute_group_capex(facts)
        assert value == pytest.approx(80.0)
        assert any("mismatch" in f.lower() for f in flags)

    def test_legacy_capex_value(self):
        facts = GroupCapexFacts(
            parent_company_name="HoldCo",
            capex_value=42.0,
            derivation="legacy",
            quote="q",
        )
        value, flags = compute_group_capex(facts)
        assert value == pytest.approx(42.0)
        assert any("legacy" in f for f in flags)

    def test_scrape_from_derivation_when_fields_null(self):
        facts = GroupCapexFacts(
            parent_company_name="HoldCo",
            derivation=(
                "Note 7 provides the opening net book value ($148,028,989.69), "
                "depreciation charge ($15,826,229.43), and closing net book value "
                "($154,050,122.81). There were no disposals."
            ),
            quote="q",
        )
        value, flags = compute_group_capex(facts)
        assert value == pytest.approx(154050122.81 - 148028989.69 + 15826229.43)
        assert any("scraped" in f for f in flags)


class TestApplyReclassifications:
    def test_reclassify_by_txn_id(self):
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "Vendor LLP", "description": "d", "amount": -100.0, "currency": "USD"}]
        )
        cats = {"t1": "other"}
        rec = Reclassification(
            action="reclassify",
            txn_id="t1",
            from_category="other",
            to_category="operating_expense",
            quote="q",
        )
        new_cats, new_df, flags, applied, unmatched = apply_reclassifications(cats, df, [rec])
        assert new_cats["t1"] == "operating_expense"
        assert any("applied 1" in f for f in flags)
        assert applied == ["t1"]
        assert unmatched == []

    def test_exclude_from_period(self):
        df = make_ledger(
            [
                {"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "t2", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 200.0, "currency": "USD"},
            ]
        )
        rec = Reclassification(action="exclude_from_period", txn_id="t1", quote="cut")
        cats, new_df, flags, applied, unmatched = apply_reclassifications(
            {"t1": "revenue", "t2": "revenue"}, df, [rec]
        )
        assert list(new_df["txn_id"]) == ["t2"]
        assert applied == ["t1"]

    def test_include_in_period_from_full_df(self):
        full = make_ledger(
            [
                {"txn_id": "in", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "out", "date": "2026-02-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 50.0, "currency": "USD"},
            ]
        )
        period = filter_period(full, "2025-01-01", "2025-12-31")
        assert list(period["txn_id"]) == ["in"]
        rec = Reclassification(action="include_in_period", txn_id="out", quote="pull in")
        cats, new_df, flags, applied, unmatched = apply_reclassifications(
            {"in": "revenue", "out": "revenue"}, period, [rec], full_df=full
        )
        assert set(new_df["txn_id"]) == {"in", "out"}
        # W3.7: bare include without to_category is not a mechanism candidate.
        assert applied == []
        assert unmatched == []

    def test_unmatched_reclass_flags(self):
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1.0, "currency": "USD"}]
        )
        rec = Reclassification(action="exclude_from_period", txn_id="TXN-MISSING", quote="gone")
        _, _, flags, applied, unmatched = apply_reclassifications({"t1": "revenue"}, df, [rec])
        assert any("unmatched" in f for f in flags)
        assert applied == []
        assert unmatched == [0]


class TestGenericFallback:
    def test_sum_category(self):
        df = make_ledger(
            [{"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -400.0, "currency": "USD"}]
        )
        spec = make_spec(covenant_type="sum_category", comparator="max", threshold=300.0)
        spec = spec.model_copy(update={"filter_category": "capex"})
        result = evaluate_sum_category(spec, df, {"t1": "capex"})
        assert result.status == "BREACH"
        assert result.actual == 400.0

    def test_ratio_generic(self):
        df = make_ledger(
            [
                {"txn_id": "n", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -50.0, "currency": "USD"},
                {"txn_id": "d", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
            ]
        )
        spec = make_spec(covenant_type="ratio_generic", comparator="max", threshold=0.4)
        spec = spec.model_copy(update={"numerator_category": "capex", "denominator_category": "revenue"})
        result = evaluate_ratio_generic(spec, df, {"n": "capex", "d": "revenue"})
        assert result.status == "BREACH"
        assert result.actual == pytest.approx(0.5)


class TestCategorySummary:
    def test_counts_and_sums(self):
        df = make_ledger(
            [
                {"txn_id": "t1", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 100.0, "currency": "USD"},
                {"txn_id": "t2", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -40.0, "currency": "USD"},
            ]
        )
        summary = category_summary(df, {"t1": "revenue", "t2": "operating_expense"})
        assert summary["n_rows"] == 2
        assert summary["n_labeled"] == 2
        assert summary["by_category_count"]["revenue"] == 1
        assert summary["by_category_sum"]["revenue"] == pytest.approx(100.0)


class TestMaxCategoryAndSpringing:
    def test_max_category_takes_largest_line(self):
        from agent.s4_engine import evaluate_max_category

        df = make_ledger(
            [
                {"txn_id": "p", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -100.0, "currency": "USD"},
                {"txn_id": "u", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -500.0, "currency": "USD"},
                {"txn_id": "t", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -200.0, "currency": "USD"},
            ]
        )
        cats = {"p": "payroll", "u": "utilities", "t": "taxes"}
        spec = make_spec(covenant_type="max_category", comparator="max", threshold=400.0)
        spec = spec.model_copy(update={"filter_categories": ["payroll", "utilities", "taxes"]})
        result = evaluate_max_category(spec, df, cats)
        assert result.actual == 500.0
        assert result.status == "BREACH"

    def test_springing_not_triggered_returns_compliant_zero(self):
        df = make_ledger(
            [
                {"txn_id": "n", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 1000.0, "currency": "USD"},
                {"txn_id": "d", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -100.0, "currency": "USD"},
            ]
        )
        cats = {"n": "financing_proceeds", "d": "operating_expense"}
        spec = make_spec(covenant_type="ratio_generic", comparator="max", threshold=2.0)
        spec = spec.model_copy(
            update={
                "numerator_category": "financing_proceeds",
                "denominator_category": "operating_expense",
                "springing_min_numerator": 4_000_000.0,
            }
        )
        result = evaluate_ratio_generic(spec, df, cats)
        assert result.status == "COMPLIANT"
        assert result.actual == 10.0  # real ratio kept; springing only flips status
        assert any("springing" in f for f in result.flags)

    def test_revenue_minus_max_category(self):
        from agent.s4_engine import evaluate_revenue_minus_max_category

        df = make_ledger(
            [
                {"txn_id": "r", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": 7_204_882.16, "currency": "USD"},
                {"txn_id": "p", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -1_204_118.53, "currency": "USD"},
                {"txn_id": "t", "date": "2025-06-01", "account_id": "ACC-1", "counterparty": "A", "description": "d", "amount": -500_000.0, "currency": "USD"},
            ]
        )
        cats = {"r": "revenue", "p": "payroll", "t": "taxes"}
        spec = make_spec(covenant_type="revenue_minus_max_category", comparator="min", threshold=5_000_000.0)
        spec = spec.model_copy(update={"filter_categories": ["payroll", "taxes"]})
        result = evaluate_revenue_minus_max_category(spec, df, cats)
        assert result.actual == 6_000_763.63
        assert result.status == "COMPLIANT"

    def test_scrape_missing_txn_amount(self):
        from agent.s4_engine import scrape_missing_txn_amounts_from_notes

        notes = [
            "Операция TXN-P8-0031 (Kyzylorda Drilling Personnel LLP): сумма не отражена; "
            "фактическая сумма операции составляет $884,204.16 (расход)."
        ]
        found = scrape_missing_txn_amounts_from_notes(notes)
        assert len(found) == 1
        assert found[0].txn_id == "TXN-P8-0031"
        assert found[0].amount == 884204.16

    def test_off_ledger_normalize_skips_target_and_ledger_match(self):
        from agent.schemas import OffLedgerAddition
        from agent.s4_engine import normalize_off_ledger_additions, scrape_off_ledger_from_notes

        items = [
            OffLedgerAddition(amount=918447.52, category="payroll", quote="note 7.1", target_txn_id=None),
            OffLedgerAddition(amount=-918447.52, category="payroll", quote="dup signed", target_txn_id=None),
            OffLedgerAddition(
                amount=884204.16, category="payroll", quote="restored", target_txn_id="TXN-X-0031"
            ),
        ]
        out, flags = normalize_off_ledger_additions(items)
        assert len(out) == 1
        assert abs(out[0].amount) == pytest.approx(918447.52)
        assert any("target_txn" in f for f in flags)
        assert any("dedup" in f for f in flags)

        ledger = make_ledger(
            [
                {
                    "txn_id": "TXN-X-0001",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "d",
                    "amount": -918447.52,
                    "currency": "USD",
                }
            ]
        )
        out2, flags2 = normalize_off_ledger_additions(items[:1], ledger)
        assert out2 == []
        assert any("matches_ledger" in f for f in flags2)

        # Severance window must not jump across 200+ chars of filler to a distant $.
        far = "severance obligations disclosed elsewhere." + (" x" * 80) + " later amount $12,345.67"
        assert scrape_off_ledger_from_notes([far]) == []
        near = "severance programme liability of $918,447.52 not booked as a TXN."
        scraped = scrape_off_ledger_from_notes([near])
        assert len(scraped) == 1
        assert scraped[0].amount == pytest.approx(918447.52)

    def test_credit_line_demoted_from_revenue(self):
        from agent.s4_engine import demote_credit_revenue_lines

        df = make_ledger(
            [
                {
                    "txn_id": "c1",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "Sublet rent received — Q4",
                    "amount": 1124814.37,
                    "currency": "USD",
                },
                {
                    "txn_id": "r1",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "Core freight revenue",
                    "amount": 5000000.0,
                    "currency": "USD",
                },
            ]
        )
        cats, flags = demote_credit_revenue_lines(df, {"c1": "revenue", "r1": "revenue"})
        assert cats["c1"] == "other"
        assert cats["r1"] == "revenue"
        assert any("credit_line_demoted_from_revenue:c1" in f for f in flags)

    def test_springing_prefers_payments_sum_over_ratio(self):
        from agent.s4_engine import CovenantResult, apply_springing

        spec = make_spec(
            covenant_type="sum_related_party_payments",
            comparator="max",
            threshold=0.08,
        )
        spec = spec.model_copy(update={"springing_min_numerator": 250_000.0})
        result = CovenantResult(
            clause="6.3",
            status="BREACH",
            actual=0.04,
            contributing_txn_ids=[],
            flags=[],
            inputs={"payments_sum": 100_000.0, "value_raw": 0.04},
        )
        out = apply_springing(spec, result)
        assert out.status == "COMPLIANT"
        assert any("springing" in f for f in out.flags)


class TestSanityGates:
    def test_empty_denominator_flags_only(self):
        from agent.s4_engine import apply_sanity_gates

        fail, flags = apply_sanity_gates(
            status="BREACH",
            actual=93582451.0,
            threshold=0.2,
            period_row_count=50,
            denominator=0.0,
            ratio_undefined=True,
        )
        # B1: empty denom is a soft flag — do not stub a computed actual.
        assert fail is False
        assert any("empty_denominator" in f for f in flags)

    def test_actual_vs_threshold_flags_only(self):
        from agent.s4_engine import apply_sanity_gates

        fail, flags = apply_sanity_gates(
            status="BREACH",
            actual=93_582_451.0,
            threshold=0.20,
            period_row_count=40,
        )
        assert fail is False
        assert any("actual_vs_threshold" in f for f in flags)

    def test_denominator_row_frac_flags_only(self):
        from agent.s4_engine import apply_sanity_gates

        fail, flags = apply_sanity_gates(
            status="COMPLIANT",
            actual=0.5,
            threshold=1.0,
            period_row_count=100,
            denominator=10.0,
            denominator_row_count=40,
        )
        assert fail is False
        assert any("denominator_row_frac" in f for f in flags)

    def test_nan_stubs(self):
        from agent.s4_engine import apply_sanity_gates, stub_cell_after_sanity

        fail, flags = apply_sanity_gates(
            status="BREACH",
            actual=float("nan"),
            threshold=1.0,
            period_row_count=10,
        )
        assert fail is True
        assert any("nan_or_inf" in f for f in flags)
        cell, _ = stub_cell_after_sanity("BREACH", flags)
        assert cell["actual"] == 0.01

    def test_clamp_flags_only(self):
        from agent.s4_engine import apply_sanity_gates

        fail, flags = apply_sanity_gates(
            status="COMPLIANT",
            actual=0.01,
            threshold=1.0,
            period_row_count=10,
            clamped=True,
        )
        assert fail is False
        assert any("clamp" in f for f in flags)


class TestSpecTaxonomyValidation:
    def test_other_in_numerator_is_invalid(self):
        from agent.s4_engine import validate_spec_against_taxonomy

        spec = make_spec(covenant_type="ratio_generic", comparator="max", threshold=0.4)
        spec = spec.model_copy(
            update={"numerator_categories": ["other"], "denominator_categories": ["revenue"]}
        )
        flags = validate_spec_against_taxonomy(spec, {"t1": "revenue", "t2": "other"})
        assert flags
        assert any("garbage" in f for f in flags)

    def test_missing_formula_categories(self):
        from agent.s4_engine import validate_spec_against_taxonomy

        spec = make_spec(covenant_type="ratio_generic", comparator="max", threshold=0.4)
        spec = spec.model_copy(
            update={"numerator_categories": ["insurance"], "denominator_categories": ["rent"]}
        )
        flags = validate_spec_against_taxonomy(spec, {"t1": "revenue", "t2": "operating_expense"})
        assert flags
        assert any("spec_taxonomy_invalid" in f for f in flags)


class TestAddendumComposition:
    """Deterministic reconstructions from docs/G2-GAP-ADDENDUM-2026-08-08.md §1."""

    def test_p1_6_1_capex_over_opex_plus_rent(self):
        df = make_ledger(
            [
                {
                    "txn_id": "TXN-P1-0010",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Ural Crane Works LLP",
                    "description": "Purchase of quayside crane equipment",
                    "amount": -1_842_006.44,
                    "currency": "USD",
                },
                {
                    "txn_id": "TXN-P1-0042",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Mangystau Berth Services",
                    "description": "berth servicing and operating costs 2025",
                    "amount": -3_104_882.61,
                    "currency": "USD",
                },
                {
                    "txn_id": "TXN-P1-0014",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "Aktau Terminal Properties",
                    "description": "terminal land lease payments 2025",
                    "amount": -918_443.27,
                    "currency": "USD",
                },
            ]
        )
        cats = {
            "TXN-P1-0010": "capex",
            "TXN-P1-0042": "operating_expense",
            "TXN-P1-0014": "rent",
        }
        spec = make_spec(covenant_type="ratio_generic", comparator="max", threshold=0.42)
        spec = spec.model_copy(
            update={
                "numerator_categories": ["capex"],
                "denominator_categories": ["operating_expense", "rent"],
            }
        )
        result = evaluate_ratio_generic(spec, df, cats)
        assert result.actual == pytest.approx(0.46, abs=0.01)

    def test_p4_6_1_adjusted_ebitda_over_revenue(self):
        df = make_ledger(
            [
                {
                    "txn_id": "rev",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "sales",
                    "amount": 7_004_318.47,
                    "currency": "USD",
                },
                {
                    "txn_id": "TXN-P4-0054",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "terminal servicing",
                    "amount": -4_431_662.19,
                    "currency": "USD",
                },
                {
                    "txn_id": "TXN-P4-0001",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "berth silt cleaning",
                    "amount": -251_338.94,
                    "currency": "USD",
                },
            ]
        )
        cats = {
            "rev": "revenue",
            "TXN-P4-0054": "operating_expense",
            "TXN-P4-0001": "operating_expense",
        }
        from agent.s4_engine import _sum_cats

        rev, _ = _sum_cats(df, cats, ["revenue"])
        opex_raw, _ = _sum_cats(df, cats, ["operating_expense"])
        ebitda = rev - (-opex_raw)
        ratio = ebitda / rev
        assert round(abs(ratio), 2) == pytest.approx(0.33)

    def test_p2_6_1_coverage_ratio(self):
        df = make_ledger(
            [
                {
                    "txn_id": "r",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "sales",
                    "amount": 10_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "f",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "facility proceeds",
                    "amount": 2_096_781.28,
                    "currency": "USD",
                },
                {
                    "txn_id": "ox",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "opex",
                    "amount": -5_918_004.37,
                    "currency": "USD",
                },
                {
                    "txn_id": "cx",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "capex",
                    "amount": -3_204_881.55,
                    "currency": "USD",
                },
                {
                    "txn_id": "TXN-P2-0040",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "Advisory engagement on depot operations",
                    "amount": -1_104_663.28,
                    "currency": "USD",
                },
            ]
        )
        df.loc[df["txn_id"] == "r", "amount"] = 12_096_781.28 - 2_096_781.28
        cats = {
            "r": "revenue",
            "f": "financing_proceeds",
            "ox": "operating_expense",
            "cx": "capex",
            "TXN-P2-0040": "operating_expense",
        }
        spec = make_spec(covenant_type="ratio_generic", comparator="min", threshold=1.0)
        spec = spec.model_copy(
            update={
                "numerator_categories": ["revenue", "financing_proceeds"],
                "denominator_categories": ["operating_expense", "capex"],
            }
        )
        result = evaluate_ratio_generic(spec, df, cats)
        assert result.actual == pytest.approx(1.18, abs=0.01)

    def test_p5_6_1_group_capex_to_ebitda(self):
        df = make_ledger(
            [
                {
                    "txn_id": "rev",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "sales",
                    "amount": 10_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "ox",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "plant opex",
                    "amount": -1_000_000.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "pay",
                    "date": "2025-06-01",
                    "account_id": "ACC-1",
                    "counterparty": "A",
                    "description": "payroll",
                    "amount": -500_000.0,
                    "currency": "USD",
                },
            ]
        )
        cats = {"rev": "revenue", "ox": "operating_expense", "pay": "payroll"}
        result = evaluate_ratio_group_capex_to_ebitda(
            make_spec(covenant_type="ratio_group_capex_to_ebitda", comparator="max", threshold=9.0),
            df,
            cats,
            group_capex=85_050_000.0,
        )
        assert result.actual == pytest.approx(9.45)
        assert result.inputs["opex_sum"] == pytest.approx(1_000_000.0)


class TestLedgerSignals:
    def test_token_freq_and_tail(self):
        from agent.ledger_signals import (
            build_ledger_signal_index,
            description_has_qualifier_tail,
            row_signal_fields,
        )

        df = make_ledger(
            [
                {
                    "txn_id": "a",
                    "date": "2025-01-01",
                    "account_id": "ACC-1",
                    "counterparty": "Bridgeport Services LLP",
                    "description": "Office supplies — Almaty office, instalment 3",
                    "amount": -10.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "b",
                    "date": "2025-01-01",
                    "account_id": "ACC-1",
                    "counterparty": "Bridgeport Services LLP",
                    "description": "More noise — period 02",
                    "amount": -10.0,
                    "currency": "USD",
                },
                {
                    "txn_id": "c",
                    "date": "2025-01-01",
                    "account_id": "ACC-1",
                    "counterparty": "Ural Crane Works LLP",
                    "description": "Purchase of crane equipment 2025",
                    "amount": -100.0,
                    "currency": "USD",
                },
            ]
        )
        idx = build_ledger_signal_index(df)
        assert description_has_qualifier_tail("Office supplies — Almaty office")
        assert not description_has_qualifier_tail("Purchase of crane equipment 2025")
        bridge = row_signal_fields("Bridgeport Services LLP", "x — y", "ACC-1", idx)
        ural = row_signal_fields(
            "Ural Crane Works LLP", "Purchase of crane equipment 2025", "ACC-1", idx
        )
        assert bridge["cp_token_freq"] == 2
        assert ural["cp_token_freq"] == 1
        assert bridge["cp_account_n"] == 2
        assert ural["desc_tail"] == 0


class TestMajorityCategories:
    def test_majority_and_disagree_flag(self):
        from agent.s4_engine import _majority_categories

        votes = [
            {"t1": "rent", "t2": "utilities"},
            {"t1": "rent", "t2": "other"},
            {"t1": "other", "t2": "utilities"},
        ]
        cats, disagree, flags = _majority_categories(votes, ["t1", "t2"])
        assert cats["t1"] == "rent"
        assert cats["t2"] == "utilities"
        assert set(disagree) == {"t1", "t2"}
        assert flags


class TestRelatedPartyTokenOverlap:
    def test_candidates_share_toponym(self):
        from agent.s4_engine import related_party_token_overlap_candidates

        cands = related_party_token_overlap_candidates(
            "Aktau Port Services LLP",
            ["Aktau Holdings LLP", "Unrelated Vendor LLP", "Bridgeport Services LLP"],
            known_related=set(),
        )
        assert cands == ["Aktau Holdings LLP"]

    def test_known_related_excluded(self):
        from agent.s4_engine import related_party_token_overlap_candidates

        cands = related_party_token_overlap_candidates(
            "Aktau Port Services LLP",
            ["Aktau Holdings LLP"],
            known_related={"Aktau Holdings LLP"},
        )
        assert cands == []


class TestRefineCovenantSpec:
    def test_refine_unrestricted_subsidiary_ratio(self):
        from agent.s4_engine import refine_covenant_spec

        spec = make_spec(covenant_type="unknown", comparator="max", threshold=0.25)
        spec = spec.model_copy(
            update={
                "quote": "Передачи активов Неограниченным дочерним / совокупный capex не более 0.25x"
            }
        )
        out = refine_covenant_spec(spec)
        assert out.covenant_type == "ratio_generic"
        assert out.numerator_categories == ["unrestricted_subsidiary_transfer"]
        assert out.denominator_categories == ["capex", "unrestricted_subsidiary_transfer"]

    def test_refine_unrestricted_beats_false_group_capex_disclaimer(self):
        """LLM often labels unrestricted-transfer ratios as group_capex and puts a
        'not Group capex … EBITDA' disclaimer in formula_description — that must not
        win over the Unrestricted-subsidiary quote."""
        from agent.s4_engine import (
            looks_like_group_capex_ebitda_ratio,
            refine_covenant_spec,
        )

        spec = make_spec(
            covenant_type="ratio_group_capex_to_ebitda",
            comparator="max",
            threshold=0.15,
        )
        formula = (
            "ratio of capital assets transferred to Unrestricted Subsidiaries "
            "to Borrower's total capital expenditures "
            "(not Group capex to Borrower EBITDA)"
        )
        quote = (
            "не допускать, чтобы совокупная стоимость капитальных активов, "
            "переданных Неограниченным дочерним организациям, превышала 0.15x "
            "совокупных капитальных затрат Заёмщика"
        )
        assert not looks_like_group_capex_ebitda_ratio(f"{quote} {formula}")
        spec = spec.model_copy(update={"quote": quote, "formula_description": formula})
        out = refine_covenant_spec(spec)
        assert out.covenant_type == "ratio_generic"
        assert out.numerator_categories == ["unrestricted_subsidiary_transfer"]
        assert out.denominator_categories == [
            "capex",
            "unrestricted_subsidiary_transfer",
        ]
