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


def fact_display_text(item, lang: str) -> str:
    """Show source wording in German, with translated generated warnings."""
    if lang != "de" or item.not_documented or item.formula:
        return item.text
    if item.original:
        text = item.original
    else:
        quotes = []
        for source in item.sources:
            quote = " ".join(str(source.get("quote") or "").split())
            if quote:
                doc_date = str(source.get("docDate") or "")
                quotes.append((doc_date, quote))
        if not quotes:
            return item.text
        newest_date = max(doc_date for doc_date, _ in quotes)
        text = min((quote for doc_date, quote in quotes
                    if doc_date == newest_date), key=len)
    text = " ".join(text.split())
    if len(text) > FACT_TEXT_MAX:
        text = text[:FACT_TEXT_MAX].rsplit(" ", 1)[0] + " …"
    for suffix in _TRANSLATED_FACT_SUFFIXES:
        if item.text.endswith(f" — {suffix}"):
            text += f" — {DE[suffix]}"
            break
    return text
