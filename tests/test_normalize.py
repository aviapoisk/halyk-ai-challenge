import pytest

from agent.normalize import (
    collapse_letter_spacing,
    find_account_anchor,
    find_account_anchors,
    find_company_name,
    make_anchor_pattern,
    normalize_company_name,
    sanitize,
)


class TestSanitize:
    def test_nbsp_becomes_space(self):
        assert sanitize("100 000") == "100 000"

    def test_zero_width_space_removed(self):
        assert sanitize("AC​C-7805") == "ACC-7805"

    def test_control_chars_removed_but_newline_tab_kept(self):
        assert sanitize("a\x00b\nc\td") == "ab\nc\td"

    def test_nfc_normalizes_combining_chars(self):
        decomposed = "é"  # e + combining acute
        assert sanitize(decomposed) == "é"  # é


class TestAccountAnchor:
    def test_plain_anchor_found(self):
        assert find_account_anchor("Счёт ACC-7805 заёмщика") == "ACC-7805"

    def test_letter_spaced_anchor_found(self):
        assert find_account_anchor("A C C - 7 8 0 5") == "ACC-7805"

    def test_no_anchor_returns_none(self):
        assert find_account_anchor("нет якоря в этом тексте") is None

    def test_letter_spacing_does_not_merge_across_words(self):
        # regression: a naive whole-text space-collapse would wrongly bridge unrelated words
        text = "A C C - 1 2 3 4 к о н е ц"
        assert find_account_anchor(text) == "ACC-1234"


class TestLedgerAnchors:
    def test_no_prefix_truncation(self):
        pattern = make_anchor_pattern(["ACC-7805", "ACC-78051"])
        found = find_account_anchors("see ACC-78051 in text", pattern)
        assert found == ["ACC-78051"]

    def test_letter_spaced_ledger_id(self):
        pattern = make_anchor_pattern(["BANK-9078"])
        found = find_account_anchors("B A N K - 9 0 7 8", pattern)
        assert found == ["BANK-9078"]

    def test_multiple_anchors(self):
        pattern = make_anchor_pattern(["ACC-1001", "ACC-1002"])
        found = find_account_anchors("ACC-1001 and ACC-1002", pattern)
        assert found == ["ACC-1001", "ACC-1002"]

    def test_fallback_without_pattern(self):
        assert find_account_anchors("ACC-1234", None) == ["ACC-1234"]


class TestCollapseLetterSpacing:
    def test_letter_spaced_line_collapsed(self):
        assert collapse_letter_spacing("А У Д И Т О Р С К О Е Д Е Л О") == "АУДИТОРСКОЕДЕЛО"

    def test_normal_line_untouched(self):
        line = "Halyk Bank of Kazakhstan JSC"
        assert collapse_letter_spacing(line) == line


class TestCompanyName:
    def test_finds_name_with_jsc_suffix(self):
        text = "Some Header\nAktobe Grain Terminal JSC\nmore text"
        assert find_company_name(text) == "Aktobe Grain Terminal JSC"

    def test_finds_name_with_too_suffix(self):
        text = "Header\nАлматы Логистика ТОО\nmore"
        assert find_company_name(text) == "Алматы Логистика ТОО"

    def test_finds_name_with_ao_suffix(self):
        text = "Header\nКазМунайГаз АО\nmore"
        assert find_company_name(text) == "КазМунайГаз АО"

    def test_returns_none_when_absent(self):
        assert find_company_name("no company name here\njust text") is None

    def test_respects_window(self):
        lines = [f"filler line {i}" for i in range(20)]
        lines.append("Aktobe Grain Terminal JSC")
        text = "\n".join(lines)
        assert find_company_name(text, window_lines=12) is None
        assert find_company_name(text, window_lines=25) == "Aktobe Grain Terminal JSC"


class TestNormalizeCompanyName:
    def test_near_duplicate_names_stay_distinct(self):
        a = normalize_company_name("Shymkent Refinery Services JSC")
        b = normalize_company_name("Shymkent Refinery JSC")
        assert a != b

    def test_whitespace_and_case_are_folded(self):
        a = normalize_company_name("  Aktobe   Grain Terminal JSC ")
        b = normalize_company_name("aktobe grain terminal jsc")
        assert a == b

    def test_curly_quotes_normalized(self):
        a = normalize_company_name('“Aktobe” Grain Terminal JSC')
        b = normalize_company_name('"Aktobe" Grain Terminal JSC')
        assert a == b

    def test_quotes_and_commas_stripped_for_kyc_ledger_match(self):
        assert normalize_company_name('"Turan Capital" LLP') == normalize_company_name("Turan Capital LLP")
        assert normalize_company_name("Aral Capital Partners, LLP") == normalize_company_name(
            "Aral Capital Partners LLP"
        )

    def test_dotted_llp_matches_plain_llp(self):
        assert normalize_company_name("Aktau Holdings L.L.P.") == normalize_company_name("Aktau Holdings LLP")
        assert normalize_company_name("Atyrau Holding Group L.L.P.") == normalize_company_name(
            "Atyrau Holding Group LLP"
        )

    def test_russian_opf_aliases_to_english(self):
        assert normalize_company_name("Алматы Логистика ТОО") == normalize_company_name("Алматы Логистика LLP")
        assert normalize_company_name("КазМунайГаз АО") == normalize_company_name("КазМунайГаз JSC")
