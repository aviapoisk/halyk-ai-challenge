import pandas as pd

from agent.s5_evidence import flip_test


def make_ledger(txn_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"txn_id": txn_ids, "amount": [float(i) for i in range(len(txn_ids))]})


class TestFlipTest:
    def test_skips_recompute_when_not_breach(self):
        calls = []
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            calls.append(list(subset["txn_id"]))
            return "COMPLIANT"

        evidence, diag, detail = flip_test(df, ["a", "b"], cats, recompute, current_status="COMPLIANT")
        assert evidence is None
        assert diag == []
        assert calls == []
        assert detail["flippers"] == []

    def test_single_flipper_returns_its_id(self):
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            return "COMPLIANT" if "b" not in set(subset["txn_id"]) else "BREACH"

        evidence, diag, detail = flip_test(
            df,
            ["a", "b", "c"],
            cats,
            recompute,
            current_status="BREACH",
            mechanism_txn_ids=["b"],
        )
        assert evidence == "b"
        assert diag == []
        assert detail["flippers"] == ["b"]

    def test_empty_mechanism_pool_accepts_contribution_singleton(self):
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            return "COMPLIANT" if "b" not in set(subset["txn_id"]) else "BREACH"

        evidence, diag, detail = flip_test(
            df, ["a", "b", "c"], cats, recompute, current_status="BREACH"
        )
        assert evidence == "b"
        assert detail["flippers"] == ["b"]
        assert "empty mechanism pool" in (detail.get("note") or "")

    def test_nonempty_mechanism_pool_accepts_outsider_singleton(self):
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            return "COMPLIANT" if "b" not in set(subset["txn_id"]) else "BREACH"

        evidence, diag, detail = flip_test(
            df,
            ["a", "b", "c"],
            cats,
            recompute,
            current_status="BREACH",
            mechanism_txn_ids=["a"],
        )
        assert evidence == "b"
        assert detail["flippers"] == ["b"]
        assert "outside mechanism pool" in (detail.get("note") or "")

    def test_zero_flippers_returns_none_with_diagnostic(self):
        df = make_ledger(["a", "b"])
        cats = {t: "revenue" for t in ["a", "b"]}

        def recompute(subset, categories):
            return "BREACH"

        evidence, diag, detail = flip_test(df, ["a", "b"], cats, recompute, current_status="BREACH")
        assert evidence is None
        assert len(diag) == 1
        assert "no single transaction" in diag[0]
        assert detail["flippers"] == []

    def test_multiple_flippers_returns_none_with_diagnostic(self):
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            ids = set(subset["txn_id"])
            return "COMPLIANT" if ("a" not in ids or "b" not in ids) else "BREACH"

        evidence, diag, detail = flip_test(df, ["a", "b", "c"], cats, recompute, current_status="BREACH")
        assert evidence is None
        assert len(diag) == 1
        assert "2 transactions" in diag[0]
        assert set(detail["flippers"]) == {"a", "b"}

    def test_document_channel_arbitrates_multiple_flippers(self):
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            ids = set(subset["txn_id"])
            return "COMPLIANT" if ("a" not in ids or "b" not in ids) else "BREACH"

        evidence, diag, detail = flip_test(
            df, ["a", "b", "c"], cats, recompute, current_status="BREACH", doc_named_txn_ids=["b"]
        )
        assert evidence == "b"
        note = detail.get("note") or ""
        note_ok = (
            any("document channel" in d or "mechanism" in d for d in diag)
            or "документный" in note
            or "mechanism" in note
        )
        assert note_ok
        assert detail["doc_named_candidates"] == ["b"]

    def test_reclassify_hypothesis_can_flip(self):
        df = make_ledger(["a", "b"])
        cats = {"a": "other", "b": "revenue"}

        def recompute(subset, categories):
            # BREACH unless "a" is reclassified to revenue (exclude never helps here).
            rev = [t for t in subset["txn_id"] if categories.get(t) == "revenue"]
            return "COMPLIANT" if "a" in rev else "BREACH"

        evidence, diag, detail = flip_test(
            df, ["a"], cats, recompute, current_status="BREACH", mechanism_txn_ids=["a"]
        )
        assert evidence == "a"
        assert diag == []
        assert detail["flippers"] == ["a"]

    def test_only_candidate_txn_ids_are_tried(self):
        """A transaction outside the candidate pool must never be excluded, even if
        excluding it would flip the verdict."""
        df = make_ledger(["a", "b", "c"])
        cats = {t: "revenue" for t in ["a", "b", "c"]}

        def recompute(subset, categories):
            return "COMPLIANT" if "c" not in set(subset["txn_id"]) else "BREACH"

        evidence, diag, detail = flip_test(df, ["a", "b"], cats, recompute, current_status="BREACH")
        assert evidence is None
        assert "no single transaction" in diag[0]
        assert detail["flippers"] == []
