import json
from pathlib import Path

import pytest

from tools.score import cell_score, normalize_gt, normalize_submission, score_all, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
OPEN_DATASET = REPO_ROOT / "docs" / "agentic-bank-public"


def make_key(status="BREACH", actual=1000.0, evidence=None):
    return {"status": status, "actual": actual, "evidence_txn_id": evidence}


def make_pred(status="BREACH", actual=1000.0, evidence=None):
    return {"status": status, "actual": actual, "evidence_txn_id": evidence}


class TestCellScoreEdgeCases:
    def test_wrong_status_zeroes_cell(self):
        key = make_key(status="BREACH", actual=1000.0, evidence=None)
        pred = make_pred(status="COMPLIANT", actual=1000.0, evidence=None)
        score, e = cell_score(pred, key, "lenient")
        assert score == 0.0
        assert e is None

    def test_missing_pred_scores_zero(self):
        key = make_key()
        score, e = cell_score(None, key, "lenient")
        assert score == 0.0
        assert e is None

    @pytest.mark.parametrize(
        "error,expected_acc",
        [
            (0.0, 1.0),
            (0.025, 0.5),
            (0.05, 0.0),
            (0.10, 0.0),  # clipped at zero beyond 5%
        ],
    )
    def test_actual_error_thresholds_null_evidence_lenient(self, error, expected_acc):
        k = 1000.0
        pred_actual = k * (1 + error)
        key = make_key(status="BREACH", actual=k, evidence=None)
        pred = make_pred(status="BREACH", actual=pred_actual, evidence=None)
        score, e = cell_score(pred, key, "lenient")
        assert e == pytest.approx(error, abs=1e-9)
        # null-key evidence: lenient mode gives the full 0.50 * acc on top of 0.50 status
        assert score == pytest.approx(0.50 + 0.50 * expected_acc)

    def test_null_key_evidence_extra_txn_id_not_penalized_lenient(self):
        key = make_key(status="BREACH", actual=1000.0, evidence=None)
        pred = make_pred(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        score, _ = cell_score(pred, key, "lenient")
        assert score == pytest.approx(1.0)  # 0.50 + 0.50*1.0, no penalty for extra evidence

    def test_null_key_evidence_extra_txn_id_penalized_strict(self):
        key = make_key(status="BREACH", actual=1000.0, evidence=None)
        pred = make_pred(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        score, _ = cell_score(pred, key, "strict")
        # 0.50 status + 0.30*1.0 actual, but the 0.20 evidence slice is withheld
        assert score == pytest.approx(0.80)

    def test_null_key_evidence_null_pred_full_score_strict(self):
        key = make_key(status="BREACH", actual=1000.0, evidence=None)
        pred = make_pred(status="BREACH", actual=1000.0, evidence=None)
        score, _ = cell_score(pred, key, "strict")
        assert score == pytest.approx(1.0)

    def test_non_null_key_evidence_exact_match(self):
        key = make_key(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        pred = make_pred(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        score, _ = cell_score(pred, key, "lenient")
        assert score == pytest.approx(1.0)

    def test_non_null_key_evidence_mismatch(self):
        key = make_key(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        pred = make_pred(status="BREACH", actual=1000.0, evidence="TXN-P1-9999")
        score, _ = cell_score(pred, key, "lenient")
        # 0.50 status + 0.30*1.0 actual + 0.0 evidence
        assert score == pytest.approx(0.80)

    def test_non_null_key_evidence_null_pred(self):
        key = make_key(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        pred = make_pred(status="BREACH", actual=1000.0, evidence=None)
        score, _ = cell_score(pred, key, "lenient")
        assert score == pytest.approx(0.80)

    def test_missing_actual_still_earns_evidence_on_non_null_key(self):
        # CASE.ru.md §4: при отсутствующем actual теряются 0.30, но 0.20 за точный
        # evidence_txn_id при не-null ключе начисляются независимо.
        key = make_key(status="BREACH", actual=1000.0, evidence="TXN-P1-0001")
        pred = {"status": "BREACH", "actual": None, "evidence_txn_id": "TXN-P1-0001"}
        score, e = cell_score(pred, key, "lenient")
        assert score == pytest.approx(0.70)
        assert e is None

    def test_missing_actual_scores_status_only(self):
        key = make_key(status="COMPLIANT", actual=1000.0, evidence=None)
        pred = {"status": "COMPLIANT", "actual": None, "evidence_txn_id": None}
        score, e = cell_score(pred, key, "lenient")
        assert score == pytest.approx(0.50)
        assert e is None


class TestShapeNormalization:
    def test_normalize_gt_and_submission_align(self):
        gt = {"scenarios": {"P1": {"covenants": {"6.1": make_key()}}}}
        sub = {"answers": {"P1": {"6.1": make_pred()}}}
        gt_n = normalize_gt(gt)
        sub_n = normalize_submission(sub)
        assert gt_n["P1"]["6.1"]["status"] == "BREACH"
        assert sub_n["P1"]["6.1"]["status"] == "BREACH"

    def test_score_all_uses_normalized_shapes(self):
        gt = {"scenarios": {"P1": {"covenants": {"6.1": make_key(status="COMPLIANT", actual=5.0, evidence=None)}}}}
        sub = {"answers": {"P1": {"6.1": make_pred(status="COMPLIANT", actual=5.0, evidence=None)}}}
        results = score_all(normalize_gt(gt), normalize_submission(sub))
        assert len(results) == 1
        assert results[0].score_lenient == pytest.approx(1.0)

    def test_score_all_follows_submission_scenario_order(self):
        """Display/report order must match submission.json keys (P1…P10,B1), not lex GT order."""
        cell = make_key(status="COMPLIANT", actual=1.0, evidence=None)
        pred = make_pred(status="COMPLIANT", actual=1.0, evidence=None)
        # GT insertion order is lexicographic (B1 before P1 before P10); submission is template order.
        gt = {
            "scenarios": {
                "B1": {"covenants": {"6.1": cell}},
                "P1": {"covenants": {"6.1": cell}},
                "P10": {"covenants": {"6.1": cell}},
                "P2": {"covenants": {"6.1": cell}},
            }
        }
        sub = {
            "answers": {
                "P1": {"6.1": pred},
                "P2": {"6.1": pred},
                "P10": {"6.1": pred},
                "B1": {"6.1": pred},
            }
        }
        results = score_all(normalize_gt(gt), normalize_submission(sub))
        assert [r.scenario for r in results] == ["P1", "P2", "P10", "B1"]


class TestBaselineOnOpenSet:
    def test_all_compliant_baseline_is_about_point_26_lenient(self):
        """Known-answer sanity check: constant 'everything COMPLIANT' predictor.

        17 BREACH / 19 COMPLIANT cells in the open GT -> ~19/36 * 0.5 ~= 0.264 unweighted lenient average.
        """
        gt_path = OPEN_DATASET / "ground_truth.json"
        template_path = OPEN_DATASET / "submission_template.json"
        if not gt_path.exists() or not template_path.exists():
            pytest.skip("open dataset not present")

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        with open(template_path, encoding="utf-8") as f:
            template = json.load(f)

        baseline_sub = {"answers": {}}
        for sid, clauses in template["answers"].items():
            baseline_sub["answers"][sid] = {
                clause: {"status": "COMPLIANT", "actual": 1.0, "evidence_txn_id": None} for clause in clauses
            }

        results = score_all(normalize_gt(gt), normalize_submission(baseline_sub))
        summary = summarize(results)

        assert summary["n_cells"] == 36
        assert summary["score_lenient_avg"] == pytest.approx(19 / 36 * 0.5, abs=0.01)
