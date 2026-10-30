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


def test_fact_display_text_never_picks_an_arbitrary_source_quote():
    # A fact's sources can include a contradicting passage (e.g. the wrong
    # side). Without its own German line the fact must stay in English.
    item = ScreenItem(
        text="Flow velocity in the internal carotid artery",
        sources=[
            {"docDate": "2015-06-28", "quote": "Rechts findet sich ..."},
            {"docDate": "2015-06-11", "quote": "Links findet sich ..."},
        ],
    )

    assert fact_display_text(item, "de") == item.text


def test_fact_display_text_keeps_english_when_fact_has_a_change_note():
    item = ScreenItem(
        text="Drug A 16 mg (1-0-1) (in the Jun 2015 letter: 1-0-1/2)",
        original="Drug A 16 mg täglich morgens und abends")

    assert fact_display_text(item, "de") == item.text


def test_fact_display_text_skips_headings_and_fragments_in_german():
    heading = ScreenItem(text="Echocardiography", original="Echokardiographie:")
    fragment = ScreenItem(text="Pressure 22 mmHg", original="(RVOT 28mm), kein ...")

    assert fact_display_text(heading, "de") == "Echocardiography"
    assert fact_display_text(fragment, "de") == "Pressure 22 mmHg"


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


def test_composed_facts_stay_in_english_in_german_mode():
    joined = ScreenItem(text="Biopsy planned, later performed",
                        original="Biopsie geplant. / Biopsie erfolgte.")
    noted = ScreenItem(text="Surgery — recommended (documented 1 Jan 2020)",
                       original="Eine Operation wird empfohlen.")

    assert fact_display_text(joined, "de") == joined.text
    assert fact_display_text(noted, "de") == noted.text
