"""Schema smoke tests for S3 fact models — no LLM."""

from __future__ import annotations

import json

from agent.schemas import CovenantSpec, FxRateFacts, GroupCapexFacts, KYCFacts, LoanAgreementFacts, Reclassification


def test_covenant_unknown_shape():
    spec = CovenantSpec(
        clause="6.4",
        covenant_type="unknown",
        comparator="max",
        threshold=1.0,
        period_start="2025-01-01",
        period_end="2025-12-31",
        quote="лимит на одну транзакцию",
        formula_description="max single txn amount",
    )
    assert spec.covenant_type == "unknown"
    assert spec.formula_description


def test_fx_pair_fields():
    facts = FxRateFacts(eur_usd_rate=None, eur_amount=100.0, usd_amount=116.0, quote="pair")
    assert facts.eur_amount == 100.0
    assert facts.eur_usd_rate is None


def test_kyc_nullable_threshold_and_account():
    facts = KYCFacts(
        borrower_name="X",
        account_id=None,
        related_party_threshold_pct=None,
        counterparties=[],
    )
    assert facts.related_party_threshold_pct is None


def test_include_in_period_action():
    rec = Reclassification(action="include_in_period", txn_id="TXN-X-0001", quote="pull")
    assert rec.action == "include_in_period"


def test_group_capex_components_nullable():
    facts = GroupCapexFacts(
        parent_company_name="HoldCo",
        opening_nbv=1.0,
        closing_nbv=2.0,
        depreciation=0.5,
        disposals=0.0,
        direct_capex_line=None,
        capex_value=None,
        derivation="roll",
        quote="q",
    )
    dumped = facts.model_dump()
    assert dumped["capex_value"] is None


def test_loan_account_id_nullable():
    loan = LoanAgreementFacts(
        borrower_name="B",
        account_id=None,
        contract_date="2025-01-01",
        effective_from="2025-01-01",
        effective_to="2025-12-31",
        is_superseded=False,
        covenants=[],
    )
    assert loan.account_id is None
    assert loan.eur_usd_rate is None
    assert loan.fx_quote == ""


def test_loan_and_auditor_embedded_fx_fields():
    from agent.schemas import AuditorNotesFacts
    from agent.s4_engine import fx_facts_from_embedded, resolve_fx_rate

    loan = LoanAgreementFacts(
        borrower_name="B",
        account_id="ACC-1",
        contract_date="2025-01-01",
        effective_from="2025-01-01",
        effective_to="2025-12-31",
        is_superseded=False,
        covenants=[],
        eur_usd_rate=None,
        eur_amount=100.0,
        usd_amount=116.0,
        fx_quote="pair",
    )
    rate, flags = resolve_fx_rate(
        fx_facts_from_embedded(
            eur_usd_rate=loan.eur_usd_rate,
            eur_amount=loan.eur_amount,
            usd_amount=loan.usd_amount,
            fx_quote=loan.fx_quote,
        )
    )
    assert rate is not None
    assert abs(rate - 1.16) < 1e-9

    auditor = AuditorNotesFacts(
        borrower_name="B",
        reclassifications_present=False,
        reclassification_notes=[],
        eur_usd_rate=1.16,
        fx_quote="rate 1.16",
    )
    rate2, _ = resolve_fx_rate(
        fx_facts_from_embedded(
            eur_usd_rate=auditor.eur_usd_rate,
            eur_amount=auditor.eur_amount,
            usd_amount=auditor.usd_amount,
            fx_quote=auditor.fx_quote,
        )
    )
    assert rate2 == 1.16


def test_loan_agreement_schema_under_anthropic_union_limit():
    """Anthropic rejects schemas with >16 nullable unions ('Schema is too complex')."""
    schema = LoanAgreementFacts.model_json_schema()
    blob = json.dumps(schema)

    def count_null_unions(obj) -> int:
        n = 0
        if isinstance(obj, dict):
            alts = obj.get("anyOf") or obj.get("oneOf")
            if isinstance(alts, list) and any(
                isinstance(a, dict) and a.get("type") == "null" for a in alts
            ):
                n += 1
            for v in obj.values():
                n += count_null_unions(v)
        elif isinstance(obj, list):
            for v in obj:
                n += count_null_unions(v)
        return n

    unions = count_null_unions(schema)
    assert unions <= 12, f"too many nullable unions for Anthropic: {unions}"
    assert "ExprNode" not in schema.get("$defs", {})
    assert "expr_nodes" not in blob
    assert "filter_category" not in blob  # singulars are SkipJsonSchema

