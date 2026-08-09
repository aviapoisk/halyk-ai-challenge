"""clause_key: канонизация ключа пункта ковенанта (регресс на прогон 2026-08-08,
где qwen вернул «Пункт 6.1 …» и все вычисленные ячейки ушли в стабы)."""

from agent.normalize import clause_key

TEMPLATE = ["6.1", "6.2", "6.3"]


def test_bare_number_passes_through():
    assert clause_key("6.1", TEMPLATE) == "6.1"


def test_russian_heading_is_reduced_to_template_key():
    assert clause_key("Пункт 6.1 Минимальный коэффициент покрытия процентов", TEMPLATE) == "6.1"


def test_english_heading_is_reduced_to_template_key():
    assert clause_key("Пункт 6.2 Individual Overhead Line Ceiling", TEMPLATE) == "6.2"


def test_prefers_template_match_over_first_number():
    # Заголовок с посторонним числом до номера пункта.
    assert clause_key("Статья 2025. Пункт 6.3 Платежи", TEMPLATE) == "6.3"


def test_first_number_when_no_template_given():
    assert clause_key("Пункт 7.4 Прочее") == "7.4"


def test_no_number_returns_stripped_raw():
    assert clause_key("  Прочие условия  ", TEMPLATE) == "Прочие условия"
