"""DE UI coverage: every section title, role label, doc-type label and the
strings rendered in app.py must have a German translation or fall back to
the English key. Clinical fact text stays untranslated by design."""

import re

from app.i18n import DE, LANGUAGES, current_lang, role_label, t
from app.needs import NEED_DEFS


def test_every_section_title_has_german():
    for section in NEED_DEFS.values():
        assert section.title in DE, section.title


def test_every_not_documented_label_has_german():
    for section in NEED_DEFS.values():
        if section.not_documented:
            assert section.not_documented in DE, section.not_documented


def test_every_role_label_has_german():
    from app.needs import ROLE_NEEDS
    for role in ROLE_NEEDS:
        pretty = role.replace("_", " ")
        pretty = pretty.upper() if len(pretty) <= 3 else pretty.title()
        assert pretty in DE, role
        assert role_label(role) == pretty  # english fallback outside streamlit


def test_unknown_key_falls_back_to_english():
    assert t("Some untranslated text") == "Some untranslated text"


def test_current_lang_defaults_to_en_outside_streamlit():
    assert current_lang() in LANGUAGES


def test_de_strings_have_no_unfilled_placeholders():
    for key, value in DE.items():
        assert re.search(r"\{\w*\}", value) is None or "{" in key, key


def test_missing_information_strings_translate_in_german(monkeypatch):
    monkeypatch.setattr("app.i18n.current_lang", lambda: "de")

    assert t("No additional laboratory values documented") == (
        "Keine weiteren Laborwerte dokumentiert")
    assert t("HbA1c: not documented") == "HbA1c: nicht dokumentiert"
    assert t("Follow-up carotid duplex ultrasound: not documented as performed") == (
        "Kontroll-Duplexsonographie der Karotiden: nicht als durchgeführt "
        "dokumentiert")


def test_dynamic_hba1c_missing_text_translates_month_in_german(monkeypatch):
    monkeypatch.setattr("app.i18n.current_lang", lambda: "de")

    assert t("HbA1c not documented after June 2015") == (
        "HbA1c nach Juni 2015 nicht dokumentiert")
