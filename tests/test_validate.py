import pytest

from tools.validate import validate

TEMPLATE = {
    "team": "",
    "contact_email": "",
    "model": "",
    "answers": {
        "P1": {
            "6.1": {"status": None, "actual": None, "evidence_txn_id": None},
            "6.2": {"status": None, "actual": None, "evidence_txn_id": None},
        }
    },
}

LEDGER_INDEX = {"P1": {"TXN-P1-0001", "TXN-P1-0002"}}


def good_submission():
    return {
        "team": "t",
        "contact_email": "a@b.com",
        "model": "claude-sonnet-5",
        "answers": {
            "P1": {
                "6.1": {"status": "BREACH", "actual": 100.5, "evidence_txn_id": "TXN-P1-0001"},
                "6.2": {"status": "COMPLIANT", "actual": 5.25, "evidence_txn_id": None},
            }
        },
    }


class TestValidate:
    def test_clean_submission_has_no_violations(self):
        result = validate(good_submission(), TEMPLATE, LEDGER_INDEX)
        assert result.ok
        assert result.violations == []

    def test_missing_top_level_key(self):
        sub = good_submission()
        del sub["model"]
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("model" in str(v) for v in result.violations)

    def test_extra_top_level_key(self):
        sub = good_submission()
        sub["extra"] = "x"
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("extra" in str(v) for v in result.violations)

    def test_missing_scenario(self):
        sub = good_submission()
        del sub["answers"]["P1"]
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("P1" in str(v) and "missing" in str(v) for v in result.violations)

    def test_missing_clause(self):
        sub = good_submission()
        del sub["answers"]["P1"]["6.2"]
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("6.2" in str(v) for v in result.violations)

    def test_extra_clause_not_in_template(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.3"] = {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None}
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_null_status_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["status"] = None
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_lowercase_status_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["status"] = "breach"
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_actual_zero_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["actual"] = 0
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_actual_negative_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["actual"] = -5.0
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_actual_as_string_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["actual"] = "100.50"
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_actual_not_rounded_to_2dp_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["actual"] = 100.567
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_evidence_txn_not_in_ledger_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["evidence_txn_id"] = "TXN-P1-9999"
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_evidence_txn_belongs_to_other_scenario_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["evidence_txn_id"] = "TXN-P2-0001"
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok

    def test_compliant_with_evidence_is_warning_not_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.2"]["evidence_txn_id"] = "TXN-P1-0002"
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert result.ok
        assert any("COMPLIANT" in str(w) for w in result.warnings)

    def test_infinity_actual_is_violation(self):
        sub = good_submission()
        sub["answers"]["P1"]["6.1"]["actual"] = float("inf")
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("finite" in str(v) for v in result.violations)

    def test_empty_team_is_violation(self):
        sub = good_submission()
        sub["team"] = ""
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("team" in str(v) for v in result.violations)

    def test_empty_contact_email_is_violation(self):
        sub = good_submission()
        sub["contact_email"] = "  "
        result = validate(sub, TEMPLATE, LEDGER_INDEX)
        assert not result.ok
        assert any("contact_email" in str(v) for v in result.violations)
