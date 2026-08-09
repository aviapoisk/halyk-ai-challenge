"""Wave-5 unit tests — draft scrape hygiene, quote lexicon, pledge mechanism. No LLM."""

from __future__ import annotations

import pandas as pd

from agent.run import _merge_auditor_facts
from agent.s2_classify import (
    auditor_doc_is_authoritative,
    auditor_doc_is_final,
    auditor_doc_is_interim_or_draft,
    auditor_notes_deny_reclassifications,
)
from agent.s4_engine import (
    apply_pledge_based_transfer_categories,
    extract_txn_ids_from_notes,
    infer_to_category_from_quote,
    scrape_missing_txn_amounts_from_notes,
)
from agent.s5_evidence import flip_test
from agent.schemas import (
    AuditorNotesFacts,
    IngestedDocument,
    KYCFacts,
    PageText,
    Reclassification,
    RelatedParty,
)


def _page(text: str) -> PageText:
    return PageText(
        page_number=1,
        raw_text=text,
        normalized_text=text,
        char_count=len(text),
        is_scan=False,
    )


def _doc(doc_id: str, text: str) -> IngestedDocument:
    return IngestedDocument(doc_id=doc_id, path=f"/t/{doc_id}.pdf", sha256=doc_id, pages=[_page(text)])


_AUTH_NOTES = (
    "АУДИТОРСКОЕ ДЕЛО №ACC-9999/2025 ФИН. ГОД\n"
    "Примечания к финансовой отчётности\n"
    "Переклассификаций за ковенантный период не требовалось.\n"
)

_DRAFT_NOTES = (
    "ПРОЕКТ — ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ. Заменена окончательным отчётом. "
    "НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА.\n"
    "Операция TXN-ZZ-0044, первоначально учтённая как Коммунальные ($418,204.37), "
    "переклассифицирована для целей соблюдения ковенантов как Налоги.\n"
)


def test_authoritative_unstamped_notes_beat_draft():
    assert auditor_doc_is_interim_or_draft(_DRAFT_NOTES)
    assert not auditor_doc_is_final(_AUTH_NOTES)
    assert auditor_doc_is_authoritative(_AUTH_NOTES)
    assert auditor_notes_deny_reclassifications(_AUTH_NOTES)


def test_merge_skips_draft_scrape_when_authoritative_denies():
    docs = [_doc("auth", _AUTH_NOTES), _doc("draft", _DRAFT_NOTES)]
    facts = [
        AuditorNotesFacts(
            borrower_name="Borrower JSC",
            reclassifications_present=False,
            reclassification_notes=["Переклассификаций за ковенантный период не требовалось."],
            reclassifications=[],
        ),
        AuditorNotesFacts(
            borrower_name="Borrower JSC",
            reclassifications_present=True,
            reclassification_notes=[
                "TXN-ZZ-0044 переклассифицирована как Налоги ($418,204.37)"
            ],
            reclassifications=[
                Reclassification(
                    action="reclassify",
                    txn_id="TXN-ZZ-0044",
                    amount=418204.37,
                    to_category="taxes",
                    quote="как Налоги",
                )
            ],
        ),
    ]
    merged, reclass, _off, _doc_id, flags, note_named = _merge_auditor_facts(docs, facts)
    assert merged is not None
    assert not any(str(r.txn_id) == "TXN-ZZ-0044" for r in reclass)
    assert any("authoritative" in f for f in flags)
    assert any("denial" in f for f in flags)
    # Evidence-only: draft TXN still visible for flip-test naming.
    assert "TXN-ZZ-0044" in note_named


def test_denial_drops_llm_category_move_on_authoritative_doc():
    """Denial must strip LLM reclassify rows (null to_category) — P6 regression."""
    auth = (
        _AUTH_NOTES
        + "Операция TXN-P6-0044, первоначально учтённая как Коммунальные услуги "
        "($418,204.37), переклассифицирована для целей соблюдения ковенантов как Налоги.\n"
        "Основание: вопрос поставлен на промежуточном этапе.\n"
    )
    docs = [_doc("auth", auth)]
    facts = [
        AuditorNotesFacts(
            borrower_name="Taraz Cement Works JSC",
            reclassifications_present=True,
            reclassification_notes=["Переклассификаций за ковенантный период не требовалось."],
            reclassifications=[
                Reclassification(
                    action="reclassify",
                    txn_id="TXN-P6-0044",
                    amount=418204.37,
                    to_category=None,
                    quote=(
                        "Предварительные вопросы по классификации операций\n"
                        "(4.1) Операция TXN-P6-0044, первоначально учтённая как Коммунальные "
                        "услуги ($418,204.37), переклассифицирована как Налоги. "
                        "Основание: вопрос поставлен на промежуточном этапе."
                    ),
                )
            ],
        ),
    ]
    _merged, reclass, _off, _doc_id, flags, _note = _merge_auditor_facts(docs, facts)
    assert not any(str(r.txn_id) == "TXN-P6-0044" for r in reclass)
    assert any("denial" in f and "dropped" in f for f in flags) or any(
        "preliminary" in f for f in flags
    )


def test_preliminary_reclass_skipped_without_denial():
    """Draft-style preliminary reclass must not enter applied list (B1)."""
    from agent.s4_engine import apply_reclassifications, is_preliminary_reclass_quote

    quote = (
        "Предварительные вопросы по классификации операций\n"
        "(4.1) Операция TXN-B1-0023, первоначально учтённая как Операционные расходы "
        "($6,166,592.66), переклассифицирована как Коммунальные услуги. "
        "Основание: вопрос поставлен на промежуточном этапе."
    )
    assert is_preliminary_reclass_quote(quote)
    df = pd.DataFrame(
        {
            "txn_id": ["TXN-B1-0023"],
            "amount": [-6_166_592.66],
            "description": ["Plant operating and maintenance expenses"],
            "counterparty": ["Pavlodar Plant Services LLP"],
        }
    )
    cats = {"TXN-B1-0023": "operating_expense"}
    recs = [
        Reclassification(
            action="reclassify",
            txn_id="TXN-B1-0023",
            amount=6_166_592.66,
            to_category="utilities",
            quote=quote,
        )
    ]
    new_cats, _out, flags, applied, _ = apply_reclassifications(cats, df, recs)
    assert new_cats["TXN-B1-0023"] == "operating_expense"
    assert applied == []
    assert any("preliminary" in f for f in flags)


def test_infer_to_category_rashody_na_oplatu_truda():
    q = (
        "Операция TXN-ZZ-0016, первоначально учтённая как Консультационные услуги "
        "($342,118.65), переклассифицирована для целей соблюдения ковенантов "
        "как Расходы на оплату труда."
    )
    assert infer_to_category_from_quote(q) == "payroll"


def test_scrape_quote_is_window_around_match():
    header = "Sary-Arka Assurance LLP\nПРОЕКТ — ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ.\n" + ("x" * 200)
    body = (
        "Операция TXN-ZZ-0025, первоначально учтённая как Капитальные затраты "
        "($418,204.37), переклассифицирована как Расходы на оплату труда."
    )
    found = scrape_missing_txn_amounts_from_notes([header + "\n" + body])
    assert len(found) == 1
    assert found[0].txn_id == "TXN-ZZ-0025"
    assert found[0].amount == 418204.37
    assert "TXN-ZZ-0025" in (found[0].quote or "")
    assert "ПРОЕКТ" not in (found[0].quote or "")
    assert infer_to_category_from_quote(found[0].quote) == "payroll"


def test_extract_txn_ids_from_notes():
    ids = extract_txn_ids_from_notes(["see TXN-AA-0001 and TXN-AA-0002; TXN-AA-0001 again"])
    assert ids == ["TXN-AA-0001", "TXN-AA-0002"]


def test_pledge_unrestricted_enters_mechanism_pool():
    df = pd.DataFrame(
        {
            "txn_id": ["r1", "u1"],
            "amount": [-100.0, -200.0],
            "amount_usd": [-100.0, -200.0],
            "description": [
                "Transfer of funds to subsidiary Conveyor",
                "Transfer of funds to subsidiary Processing",
            ],
            "counterparty": [
                "Zhezkazgan Conveyor Assets LLP",
                "Zhezkazgan Processing Holdings LLP",
            ],
            "currency": ["USD", "USD"],
        }
    )
    cats = {
        "r1": "unrestricted_subsidiary_transfer",
        "u1": "unrestricted_subsidiary_transfer",
    }
    kyc = KYCFacts(
        borrower_name="Zhezkazgan Mining Services JSC",
        account_id="ACC-7809",
        related_party_threshold_pct=25.0,
        unrestricted_pledge_threshold_pct=50.0,
        counterparties=[
            RelatedParty(
                counterparty_name="Zhezkazgan Processing Holdings LLP",
                voting_rights_pct=100.0,
                pledged_assets_pct=11.4,
                quote="Processing 11.4%",
            ),
            RelatedParty(
                counterparty_name="Zhezkazgan Conveyor Assets LLP",
                voting_rights_pct=100.0,
                pledged_assets_pct=87.6,
                quote="Conveyor 87.6%",
            ),
        ],
    )
    new_cats, _flags, remapped = apply_pledge_based_transfer_categories(df, cats, kyc)
    assert new_cats["r1"] == "other"
    assert new_cats["u1"] == "unrestricted_subsidiary_transfer"
    assert remapped == ["u1"]


def test_flip_test_accepts_unique_mechanism_flipper():
    df = pd.DataFrame(
        {
            "txn_id": ["n1", "d1", "x1"],
            "amount": [418204.37, 1204663.28, 302118.64],
        }
    )
    cats = {t: "other" for t in ["n1", "d1", "x1"]}

    def recompute(period, categories):
        # Ratio breaches only while n1 is present.
        return "COMPLIANT" if "n1" not in set(period["txn_id"]) else "BREACH"

    chosen, _notes, meta = flip_test(
        df,
        ["n1", "d1", "x1"],
        cats,
        recompute,
        current_status="BREACH",
        mechanism_txn_ids=["n1"],
        doc_named_txn_ids=[],
    )
    assert chosen == "n1"
    assert meta.get("flippers") == ["n1"]
