import re

from app.i18n import DE


FACT_TEXT_MAX = 300
_TRANSLATED_FACT_SUFFIXES = (
    "on board, check protocol around iodinated contrast",
    "done without contrast because of the allergy",
)


def resolve_role(choice: str, other: str) -> str:
    """Prefer free-text role input, otherwise normalize the selected label."""
    if other.strip():
        return other.strip()
    if choice == "Other (type below)":
        return ""
    return choice.casefold().replace(" ", "_")


def format_missing_item(label: str, needed_for: str | None) -> str:
    """Format a missing-item label without empty reason delimiters."""
    from app.i18n import t
    reason = (needed_for or "").strip()
    return f"{label} — {t('needed for')}: {reason}" if reason else label


def derived_label(formula: str | None, source_count: int) -> str:
    """Render calculation provenance without exposing internal formula names.

    Returns an i18n key; callers format ``n`` via app.i18n.t().
    """
    if formula == "weight_trend":
        return "Calculated from {n} documented weights"
    if formula == "bmi":
        return "Calculated from documented weight and height"
    if formula == "age":
        return "Calculated from the documented birth date"
    return "Calculated value"


# Notes the extractor appends to a fact (dose/schedule changes, comparisons).
# The German source line does not carry them, so such facts stay in English.
_FACT_NOTE = re.compile(r"\((?:in the|previously|was)\b|\bchanged\b",
                        re.IGNORECASE)


def _composed(text: str, original: str | None) -> bool:
    """A fact built from several passages, or carrying an added status note
    ("— recommended", "— later documented as performed …"), says more than
    any single German line, so it is shown in English."""
    for suffix in _TRANSLATED_FACT_SUFFIXES:  # these have a German version
        text = text.removesuffix(f" — {suffix}")
    return " — " in text or " / " in (original or "")


def _usable_original(original: str | None) -> str | None:
    text = " ".join((original or "").split())
    if len(text) < 4 or text.endswith(":") or text.startswith("("):
        return None
    return text


def fact_display_text(item, lang: str) -> str:
    """German mode shows the fact's own German source line when it has one.

    Only the extracted ``original`` of this fact is used — never an arbitrary
    quote from its sources, which may be a contradicting or partial passage.
    Facts without a usable original, or with an appended note, stay in
    English so no information is lost.
    """
    if lang != "de" or item.not_documented or item.formula:
        return item.text
    text = _usable_original(item.original)
    if (text is None or _FACT_NOTE.search(item.text)
            or _composed(item.text, item.original)):
        return item.text
    if len(text) > FACT_TEXT_MAX:
        text = text[:FACT_TEXT_MAX].rsplit(" ", 1)[0] + " …"
    for suffix in _TRANSLATED_FACT_SUFFIXES:
        if item.text.endswith(f" — {suffix}"):
            text += f" — {DE[suffix]}"
            break
    return text


def date_caption(item_text: str, date: str | None) -> str | None:
    """Return the date to show next to a fact, or None if the text has it."""
    if not date or date in item_text:
        return None
    return date


def _source_json(source) -> dict:
    if isinstance(source, dict):
        return {
            "doc": source.get("doc") or source.get("doc_id"),
            "doc_title": source.get("docTitle"),
            "doc_date": source.get("docDate"),
            "doc_hash": source.get("docHash"),
            "start": source.get("start"),
            "end": source.get("end"),
            "quote": source.get("quote", ""),
        }
    return {
        "doc": source.doc_id, "doc_title": None, "doc_date": source.doc_date,
        "doc_hash": source.doc_hash, "start": source.start, "end": source.end,
        "quote": source.quote,
    }


def _item_json(item, shown: bool) -> dict:
    return {
        "text": item.text,
        "original": item.original,
        "date": item.date,
        "status": item.status,
        "not_documented": item.not_documented,
        "derived": item.derived,
        "formula": item.formula,
        "support": item.support,
        "visible_without_clicking": shown,
        "sources": [_source_json(source) for source in item.sources or []],
    }


def screen_to_json(screen) -> dict:
    """Everything the screen shows, with full provenance, for validation."""
    return {
        "role": screen.role,
        "task": screen.task_text,
        "reference_date": screen.reference_date,
        "review_status": screen.review_status,
        "versions": screen.versions,
        "key_questions": [
            {"question": question.question,
             "answer": [_item_json(item, True) for item in question.items],
             "contradictions": question.contradiction_keys}
            for question in screen.key_questions
        ],
        "sections": [
            {"key": section.key, "title": section.title,
             "items": [_item_json(item, True)
                       for item in [*section.items, *(section.more or [])]]}
            for section in screen.sections
        ],
        "missing": [
            {"label": missing.label, "needed_for": missing.needed_for,
             "status": missing.status}
            for missing in screen.missing
        ],
        "contradictions": [
            {"key": c.key, "title": c.title, "detail": c.detail,
             "kind": c.kind,
             "passages": [_source_json(span) for span in c.evidence]}
            for c in screen.contradictions
        ],
        "errors": screen.errors,
    }
