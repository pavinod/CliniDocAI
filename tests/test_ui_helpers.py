from app.schemas import ScreenItem
from app.ui_helpers import (derived_label, fact_display_text,
                            format_missing_item, resolve_role)


def test_nonempty_free_text_role_wins_over_dropdown():
    assert resolve_role("GP", "Cardiologist") == "Cardiologist"


def test_selected_role_is_normalized_when_free_text_is_empty():
    assert resolve_role("GP", "  ") == "gp"


def test_other_role_without_free_text_is_empty():
    assert resolve_role("Other (type below)", "") == ""


def test_missing_item_omits_empty_reason_parentheses():
    formatted = format_missing_item("Renal function", "")
    assert formatted == "Renal function"
    assert "— ()" not in formatted
    assert format_missing_item("Renal function", "before medication review") == (
        "Renal function — needed for: before medication review")


def test_derived_formula_labels_are_human_readable():
    assert derived_label("weight_trend", 3).format(n=3) == (
        "Calculated from 3 documented weights")
    assert derived_label("bmi", 2) == (
        "Calculated from documented weight and height")
    assert derived_label("age", 1) == (
        "Calculated from the documented birth date")
    assert derived_label("unknown", 1) == "Calculated value"


def test_fact_display_text_uses_english_fact_text_in_english():
    item = ScreenItem(text="English fact", original="German source wording")

    assert fact_display_text(item, "en") == "English fact"


def test_fact_display_text_uses_normalized_original_in_german():
    item = ScreenItem(
        text="English fact", original="  German\n source   wording  ")

    assert fact_display_text(item, "de") == "German source wording"


def test_fact_display_text_uses_shortest_nonempty_quote_in_german():
    item = ScreenItem(
        text="English fact",
        sources=[
            {"quote": "  A longer\nsource wording  "},
            {"quote": " \n "},
            {"quote": "  Shorter\tsource  "},
        ],
    )

    assert fact_display_text(item, "de") == "Shorter source"


def test_fact_display_text_prefers_newest_quote_over_shorter_old_quote():
    item = ScreenItem(
        text="English fact",
        sources=[
            {"docDate": "2015-06-11", "quote": "Short old quote"},
            {"docDate": "2015-06-28",
             "quote": "Longer wording from the newest document"},
        ],
    )

    assert fact_display_text(item, "de") == (
        "Longer wording from the newest document")


def test_fact_display_text_uses_shortest_quote_when_dates_tie():
    item = ScreenItem(
        text="English fact",
        sources=[
            {"docDate": "2015-06-28", "quote": "A longer quote"},
            {"docDate": "2015-06-28", "quote": "Short quote"},
        ],
    )

    assert fact_display_text(item, "de") == "Short quote"


def test_fact_display_text_translates_generated_warning_suffixes():
    cases = [
        (
            "English medication — on board, check protocol around "
            "iodinated contrast",
            "eingenommen – Vorgehen bei jodhaltigem Kontrastmittel prüfen",
        ),
        (
            "English CT — done without contrast because of the allergy",
            "wegen der Allergie ohne Kontrastmittel durchgeführt",
        ),
    ]
    for fact_text, translated_suffix in cases:
        item = ScreenItem(text=fact_text, original="Deutscher Quelltext")
        assert fact_display_text(item, "de") == (
            f"Deutscher Quelltext — {translated_suffix}")


def test_fact_display_text_keeps_calculated_and_missing_text_in_german():
    calculated = ScreenItem(
        text="Calculated fact", original="German original", formula="age")
    missing = ScreenItem(
        text="Missing fact", original="German original", not_documented=True)

    assert fact_display_text(calculated, "de") == "Calculated fact"
    assert fact_display_text(missing, "de") == "Missing fact"


def test_fact_display_text_falls_back_without_german_source_text():
    item = ScreenItem(text="English fact")

    assert fact_display_text(item, "de") == "English fact"


def test_fact_display_text_truncates_german_source_at_word_boundary():
    item = ScreenItem(text="English fact", original=" ".join(["source"] * 60))

    result = fact_display_text(item, "de")

    assert result.endswith(" …")
    assert len(result) <= 300
    assert result[:-2].endswith("source")
