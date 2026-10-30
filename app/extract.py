"""Deterministic fact extraction from the cleaned document texts.

Each extractor reads the documents and emits Fact objects that carry
(category, english display text, date, source spans, keywords). Nothing is
invented: a fact only exists where a span in a document supports it. Where
facts are derived (e.g. a weight change or a reformatted dose) the display
text says so and the underlying passages are attached.
"""

from __future__ import annotations

import re
from datetime import date as Date

from .labels import doc_type_label
from .lexicon import (DRUGS, DRUG_NAMES, DIAGNOSIS_EN, allergy_reactions_en,
                      diagnosis_subcategory, finding_translations_en)
from .pipeline import Document, PatientData
from .schemas import EvidenceSpan, Fact

MONTH_DE = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}
REFERENCE_DATE = "2016-04-15"
_MONTH_NAMES = {
    "januar": 1, "january": 1, "jan": 1,
    "februar": 2, "february": 2, "feb": 2,
    "märz": 3, "maerz": 3, "march": 3, "mär": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mai": 5, "may": 5,
    "juni": 6, "june": 6, "jun": 6,
    "juli": 7, "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "october": 10, "okt": 10, "oct": 10,
    "november": 11, "nov": 11,
    "dezember": 12, "december": 12, "dez": 12, "dec": 12,
}
_DATE_PATTERN = re.compile(
    r"(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{2,4})|"
    r"(?P<word_day>\d{1,2})\.?\s+(?P<word_month>[A-Za-zÄÖÜäöü]+)\s+"
    r"(?P<word_year>\d{4})|"
    r"(?P<numeric_month>\d{1,2})[./](?P<numeric_year>\d{4})|"
    r"(?P<month_only>[A-Za-zÄÖÜäöü]+)\s+(?P<month_year>\d{4})",
    re.IGNORECASE,
)
_SIDE_PATTERN = re.compile(r"\b(links?\w*|rechts?\w*|li\.|re\.)(?=\W|$)",
                           re.IGNORECASE)
_SIZE_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*x\s*(\d+(?:[.,]\d+)?)\s*cm\b",
    re.IGNORECASE,
)
_TNM_PATTERN = re.compile(
    r"\b(y?p?c?T[0-4is][a-d]?)\s*(N[0-3X][a-c]?)\s*(M[01X])"
    r"((?:\s*[VLRG][0-9X])*)",
    re.IGNORECASE,
)


def en_month(date: str) -> str:
    """Render a date at its source-supported precision."""
    parts = date.split("-")
    if len(parts) >= 2:
        return f"{MONTH_DE[parts[1]][:3]} {parts[0]}"
    return date


def en_date(date: str | None) -> str | None:
    """Render a complete or partial source-supported date."""
    if not date:
        return date
    parts = date.split("-")
    if len(parts) == 3 and parts[1] in MONTH_DE:
        return f"{int(parts[2])} {MONTH_DE[parts[1]][:3]} {parts[0]}"
    if len(parts) == 2 and parts[1] in MONTH_DE:
        return f"{MONTH_DE[parts[1]][:3]} {parts[0]}"
    return date


def fact_item(f: Fact, docs: dict[str, Document]) -> dict:
    """Serialise a fact for the screen: display fields + evidence with
    doc title, date, quote and hash for full provenance."""
    return {
        "key": f.key,
        "category": f.category,
        "text": f.text,
        "original": f.original,
        "date": en_date(f.date),
        "event_date": (f.date if f.category == "history.event" else None),
        "event_month": (
            f.date[:7] if f.category == "history.event" and f.date else None),
        "derived": f.derived,
        "status": f.status,
        "formula": f.formula,
        "sources": [
            {
                "doc": s.doc_id,
                "start": s.start,
                "end": s.end,
                "docTitle": docs[s.doc_id].title,
                "docDate": docs[s.doc_id].date,
                "docHash": s.doc_hash,
                "quote": docs[s.doc_id].quote(s),
            }
            for s in f.evidence
        ],
    }


# ---------------------------------------------------------------- helpers

def _sentences(doc: Document):
    """Yield (start, end, text) sentence-ish chunks (line-based split)."""
    for i, ln in enumerate(doc.lines):
        if ln.strip():
            yield doc.line_offsets[i], doc.line_offsets[i] + len(ln), ln


def _span(doc: Document, start: int, end: int) -> EvidenceSpan:
    return doc.sentence_span(start, end)


def _regex_spans(doc: Document, pattern: str, flags=re.IGNORECASE):
    return [_span(doc, m.start(), m.end()) for m in re.finditer(pattern, doc.text, flags)]


def _multiline_sentence_span(doc: Document, start: int,
                             end: int) -> EvidenceSpan:
    """Span a sentence across soft line breaks, ending at sentence punctuation."""
    text = doc.text
    blank_line = text.rfind("\n\n", 0, start)
    paragraph_start = blank_line + 2 if blank_line >= 0 else 0
    section_headers = list(re.finditer(
        r"(?im)^[ \t]*([^\n:]{1,60})[ \t]*:[ \t]*",
        text[paragraph_start:start]))
    if (section_headers
            and section_headers[-1].group(1).strip().casefold() == "diagnosen"):
        line_start = text.rfind("\n", 0, start) + 1
        header_end = paragraph_start + section_headers[-1].end()
        span_start = max(line_start, header_end)
        line_end = text.find("\n", end)
        return doc.span(span_start, line_end if line_end >= 0 else len(text))
    start_bound = paragraph_start
    for match in re.finditer(
            r"[.!?](?=\s+[A-ZÄÖÜ]|\s*$)",
            text[paragraph_start:start]):
        point = paragraph_start + match.start()
        if text[point] == "." and point and point + 1 < len(text):
            if text[point - 1].isdigit() and text[point + 1].isdigit():
                continue
            if re.search(
                    r"(?:\bA|St\.p|z\.B|Dr|li|re|Vd\.a|ca|bzw|ggf|prof|etc|usw)\.$",
                    text[max(0, point - 8):point + 1], re.IGNORECASE):
                continue
        start_bound = paragraph_start + match.end()
    paragraph_end = text.find("\n\n", end)
    limit = paragraph_end if paragraph_end >= 0 else len(text)
    if paragraph_end >= 0:
        continuation = paragraph_end + 2
        while continuation < len(text) and text[continuation] == "\n":
            continuation += 1
        if continuation < len(text) and text[continuation].islower():
            next_paragraph_end = text.find("\n\n", continuation)
            limit = (next_paragraph_end
                     if next_paragraph_end >= 0 else len(text))
    for match in re.finditer(r"[.!?]", text[end:limit]):
        point = end + match.start()
        if text[point] == "." and point and point + 1 < len(text):
            if text[point - 1].isdigit() and text[point + 1].isdigit():
                continue
            if re.search(
                    r"(?:\bA|St\.p|z\.B|Dr|li|re|Vd\.a|ca|bzw|ggf|prof|etc|usw)\.$",
                         text[max(0, point - 8):point + 1], re.IGNORECASE):
                continue
        following = text[point + 1:]
        if following and not re.match(r"\s+(?:[A-ZÄÖÜ]|$)", following):
            continue
        return doc.span(start_bound, point + 1)
    return doc.span(start_bound, limit)


def _multiline_spans(doc: Document, pattern: str) -> list[EvidenceSpan]:
    spans = {}
    for match in re.finditer(pattern, doc.text, re.IGNORECASE):
        span = _multiline_sentence_span(doc, match.start(), match.end())
        spans.setdefault((span.start, span.end), span)
    return list(spans.values())


def _document_lines(doc: Document) -> tuple[list[str], list[int]]:
    if doc.lines and len(doc.lines) == len(doc.line_offsets):
        return doc.lines, doc.line_offsets
    lines = doc.text.split("\n")
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1
    return lines, offsets


def _line_for_offset(doc: Document, offset: int) -> str:
    lines, offsets = _document_lines(doc)
    line_index = max(
        (index for index, start in enumerate(offsets) if start <= offset),
        default=0)
    return lines[line_index]


def _continued_header_span(doc: Document, offset: int,
                           label: str) -> EvidenceSpan:
    lines, offsets = _document_lines(doc)
    line_index = max(
        (index for index, start in enumerate(offsets) if start <= offset),
        default=0)
    header = lines[line_index]
    start = offsets[line_index]
    end = start + len(header)
    is_ultrasound_header = (
        label == "Cerebral artery ultrasound"
        and re.match(r"^\s*us hirnversorgende arterien\b", header,
                     re.IGNORECASE))
    if not header.rstrip().endswith(":") and not is_ultrasound_header:
        return doc.span(start, end)
    included = False
    for index in range(line_index + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            if included:
                break
            continue
        is_header = bool(re.match(r"^\s*[^:]{1,80}:\s*.*$", line))
        is_ultrasound_subheading = (
            label == "Cerebral artery ultrasound"
            and re.match(
                r"^\s*(?:doppler/)?duplex sonographie der "
                r"(?:hirnversorgenden )?halsarterien\s*:\s*$",
                           line, re.IGNORECASE))
        if is_header and not is_ultrasound_subheading:
            break
        end = offsets[index] + len(line)
        included = True
    return doc.span(start, end)


def _procedure_fact_text(label: str, quote: str,
                         fallback_date: str | None) -> str:
    values = _parsed_source_values(quote, fallback_date)
    dates = _dates_in(quote)
    date_labels = {_display_date(value[2]) for value in dates}
    if fallback_date:
        date_labels.add(_display_date(fallback_date))
    values = [value for value in values if value not in date_labels]
    translations = finding_translations_en(quote)
    values.extend(value for value in translations if value not in values)
    return f"{label}, {', '.join(dict.fromkeys(values))}" if values else label


def _schedule_en(row: str) -> str | None:
    """Render a German 4-column medication schedule (mo mi ab na) in English."""
    m = re.search(r"(\d+(?:/\d+)?)\s+(\d+(?:/\d+)?)\s+(\d+(?:/\d+)?)\s+(\d+(?:/\d+)?)", row)
    if m:
        mo, mi, ab, na = m.groups()
        sig = f"{mo}-{mi}-{ab}" + (f"-{na}" if na not in ("0", "0/0") else "")
        parts = []
        if mo != "0":
            parts.append(f"{mo} in the morning")
        if mi != "0":
            parts.append(f"{mi} at midday")
        if ab != "0":
            parts.append(f"{ab} in the evening")
        if na != "0":
            parts.append(f"{na} at night")
        return f"{', '.join(parts)} ({sig})"
    if re.search(r"1x\s*/\s*Woche|1x/Woche|wöchentlich", row):
        return "once weekly"
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _parse_date_match(match: re.Match) -> str | None:
    groups = match.groupdict()
    if groups.get("day"):
        day, month, year = (int(groups["day"]), int(groups["month"]),
                            groups["year"])
        month = int(month)
    elif groups.get("word_day"):
        day = int(groups["word_day"])
        month = _MONTH_NAMES.get(groups["word_month"].casefold().rstrip("."))
        year = groups["word_year"]
    elif groups.get("numeric_month"):
        day = None
        month = int(groups["numeric_month"])
        year = groups["numeric_year"]
    else:
        day = None
        month = _MONTH_NAMES.get(
            (groups.get("month_only") or "").casefold().rstrip("."))
        year = groups.get("month_year")
    if not month or not year:
        return None
    year = int(year)
    if year < 100:
        year += 2000
    try:
        if day is None:
            return f"{year:04d}-{month:02d}"
        return Date(year, month, day).isoformat()
    except ValueError:
        return None


def _dates_in(text: str) -> list[tuple[int, int, str]]:
    dates = []
    for match in _DATE_PATTERN.finditer(text):
        parsed = _parse_date_match(match)
        if parsed:
            dates.append((match.start(), match.end(), parsed))
    return dates


def _display_date(value: str | None) -> str | None:
    if not value:
        return None
    return en_date(value)


def _side(text: str) -> str | None:
    match = _SIDE_PATTERN.search(text)
    if not match:
        return None
    return "left" if match.group(1).casefold().startswith(("l", "li")) else "right"


def _number_text(value: str) -> str:
    return value.replace(",", ".")


def _parsed_source_values(text: str, fallback_date: str | None = None) -> list[str]:
    values = []
    dates = _dates_in(text)
    parsed_date = dates[0][2] if dates else fallback_date
    if parsed_date:
        values.append(_display_date(parsed_date) or parsed_date)

    size = _SIZE_PATTERN.search(text)
    if size:
        values.append(
            f"{_number_text(size.group(1))} x {_number_text(size.group(2))} cm")
    else:
        single_size = re.search(r"(\d+(?:[.,]\d+)?)\s*cm\b", text, re.IGNORECASE)
        if single_size:
            values.append(f"{_number_text(single_size.group(1))} cm")

    birads = re.search(r"bi-?rads?\s*([IV0-6]+)", text, re.IGNORECASE)
    if birads:
        values.append(f"BI-RADS {birads.group(1).upper()}")

    for percent in dict.fromkeys(re.findall(r"(\d+)\s*%", text)):
        values.append(f"{percent}%")

    side = _side(text)
    if side:
        values.append(side)

    pressure = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg",
                         text, re.IGNORECASE)
    if pressure:
        values.append(f"blood pressure {pressure.group(1)}/{pressure.group(2)} mmHg")

    months = re.search(r"\bin\s+(\d+)\s+monaten\b", text, re.IGNORECASE)
    if months:
        values.append(f"in {months.group(1)} months")

    return list(dict.fromkeys(values))


def _source_fact_text(label: str, quote: str,
                      fallback_date: str | None = None) -> str:
    values = _parsed_source_values(quote, fallback_date)
    return f"{label}, {', '.join(values)}" if values else label


def _event_dates(doc: Document, event: str) -> list[tuple[str, EvidenceSpan]]:
    """Return dates nearest to admission or discharge wording in each line."""
    markers = re.compile(r"aufnahme|aufgenommen|entlassen|entlassung",
                         re.IGNORECASE)
    out = []
    for line_index, line in enumerate(doc.lines):
        line_dates = _dates_in(line)
        line_markers = list(markers.finditer(line))
        for start, end, parsed in line_dates:
            relevant = [
                marker for marker in line_markers
                if (("aufnahme" in marker.group().casefold()
                     or "aufgenommen" in marker.group().casefold())
                    if event == "admission" else
                    ("entlass" in marker.group().casefold()))
            ]
            if not relevant:
                continue
            nearest = min(relevant, key=lambda marker: abs(marker.start() - start))
            other_event = (
                "entlass" if event == "admission" else "aufnahme")
            competing = [
                marker for marker in line_markers
                if other_event in marker.group().casefold()
            ]
            if competing and min(
                    abs(marker.start() - start) for marker in competing
            ) < abs(nearest.start() - start):
                continue
            offset = doc.line_offsets[line_index]
            out.append((parsed, doc.sentence_span(offset + start, offset + end)))
    return out


def _age_in_years(birth_date: str, event_date: str) -> int:
    birth = Date.fromisoformat(birth_date)
    event = Date.fromisoformat(event_date)
    return event.year - birth.year - (
        (event.month, event.day) < (birth.month, birth.day))


def _plan_status(category: str, event_date: str | None) -> str:
    if category == "plan.surgery":
        return "planned"
    precise_date = bool(event_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date))
    if event_date and ((precise_date and event_date > REFERENCE_DATE)
                       or (len(event_date) == 7
                           and event_date > REFERENCE_DATE[:7])):
        return "planned"
    if category in {"plan.followup", "plan.treatment"}:
        return "recommended"
    if category == "plan.done" and precise_date and event_date <= REFERENCE_DATE:
        return "completed"
    return "positive"


# ---------------------------------------------------------------- extractors

def extract_demographics(data: PatientData) -> list[Fact]:
    birth_pattern = re.compile(
        r"geb\.?\s*(?:am\s*)?(\d{1,2})\.(\d{1,2})\.(\d{4})",
        re.IGNORECASE,
    )
    birth: tuple[str, EvidenceSpan, str] | None = None
    sex_info: tuple[str, EvidenceSpan, str] | None = None
    for doc in sorted(data.documents.values(), key=lambda item: item.date):
        if sex_info is None:
            match = re.search(r"\b(Frau|Herr|weiblich|männlich)\b",
                              doc.text, re.IGNORECASE)
            if match:
                sex_info = (
                    "Female" if match.group(1).casefold() in
                    {"frau", "weiblich"} else "Male",
                    doc.sentence_span(match.start(), match.end()),
                    match.group(1),
                )
        if birth is None:
            match = birth_pattern.search(doc.text)
            if match:
                day, month, year = map(int, match.groups())
                try:
                    birth_date = Date(year, month, day).isoformat()
                except ValueError:
                    continue
                birth = (
                    birth_date,
                    doc.sentence_span(match.start(), match.end()),
                    doc.text[match.start():match.end()],
                )

    if birth is None:
        if not sex_info:
            return []
        return [Fact(
            key="patient.demographics", category="patient.demographics",
            text=sex_info[0], original=sex_info[2], evidence=[sex_info[1]],
            kw={"sex", sex_info[0].casefold(), "demographics", "patient"},
        )]

    birth_date, birth_span, original = birth
    sex = sex_info[0] if sex_info else None
    description = ", ".join(
        part for part in (sex, f"born {birth_date}" if birth_date else None)
        if part)
    demographic_evidence = [birth_span]
    if sex_info:
        demographic_evidence.append(sex_info[1])
    facts = [Fact(
        key="patient.demographics", category="patient.demographics",
        text=description,
        original=" / ".join(value for value in (
            sex_info[2] if sex_info else None, original) if value),
        evidence=_unique_spans(demographic_evidence),
        kw={"age", "born", "birth", "sex", "female", "male",
            "demographics", "patient"},
    )]

    reference_age = _age_in_years(birth_date, REFERENCE_DATE)
    facts.append(Fact(
        key="patient.age.reference", category="patient.age",
        text=f"Age {reference_age} years on {REFERENCE_DATE} (born {birth_date})",
        original=original, date=REFERENCE_DATE, evidence=[birth_span],
        kw={"age", "born", "birth", "demographics", "patient"},
        derived=True, formula="age",
        inputs=[
            f"birth:{birth_span.doc_id}:{birth_span.start}:{birth_span.end}",
            f"reference_date:{REFERENCE_DATE}",
        ],
    ))

    admission_dates: dict[str, list[EvidenceSpan]] = {}
    for doc in data.documents.values():
        for event_date, span in _event_dates(doc, "admission"):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
                admission_dates.setdefault(event_date, []).extend(
                    [birth_span, span])
    for event_date, evidence in sorted(admission_dates.items()):
        age = _age_in_years(birth_date, event_date)
        facts.append(Fact(
            key=f"patient.age.admission.{event_date}",
            category="patient.age",
            text=f"Age {age} years at admission {event_date} (born {birth_date})",
                original=original, date=event_date, evidence=_unique_spans(evidence),
                kw={"age", "born", "birth", "admission", "demographics", "patient"},
                derived=True, formula="age",
                inputs=[
                    f"birth:{birth_span.doc_id}:{birth_span.start}:{birth_span.end}",
                    f"admission:{span.doc_id}:{span.start}:{span.end}",
                ],
            ))
    return facts


def _section_lines(doc: Document, start_pat: str, stop_pats: list[str]):
    """Line indexes of a labelled block ('Diagnosen:', 'Allergien:', ...)."""
    out = []
    in_block = False
    for i, ln in enumerate(doc.lines):
        if not in_block:
            if re.match(start_pat, ln.strip(), re.IGNORECASE):
                in_block = True
            continue
        s = ln.strip()
        if not s:
            break
        if any(re.match(p, s, re.IGNORECASE) for p in stop_pats):
            break
        out.append((i, s))
    return out


def _status_for_dx_line(ln: str) -> str:
    low = ln.lower()
    if re.search(r"vd\.a\.|verdacht|suspected", low):
        return "suspected"
    if re.search(r"st\.?\s*p\.|status post|z\.n\.", low):
        return "historical"
    return "positive"


def extract_diagnoses(data: PatientData) -> list[Fact]:
    facts: dict[str, Fact] = {}
    stop = [r"allergien", r"sehr geehrte", r"vitalparameter", r"befunde", r"anamnese",
            r"wir berichten", r"verlauf", r"therapie", r"procedere", r"erhobene"]
    for doc in data.documents.values():
        for i, ln in _section_lines(doc, r"^diagnosen\s*:?$", stop):
            sub = diagnosis_subcategory(ln)
            en = None
            for pat, label in DIAGNOSIS_EN:
                if re.search(pat, ln, re.IGNORECASE):
                    en = label
                    break
            breast_diagnosis = bool(re.match(
                r"^(?:vd\.a\.\s*)?mamma[- ]?ca\b|^(?:vd\.a\.\s*)?mammakarzinom\b",
                ln, re.IGNORECASE))
            if breast_diagnosis:
                side = ("left" if re.search(r"links|\bli\.", ln, re.IGNORECASE)
                        else "right" if re.search(r"rechts|\bre\.", ln, re.IGNORECASE)
                        else None)
                text = f"Breast cancer, {side}" if side else "Breast cancer"
                sub = "active"
                key = "dx.breast_cancer"
            else:
                text = en or "Diagnosis recorded"
                carotid_diagnosis = re.search(
                    r"carotis\w*[^.\n]*(?:\bli\.|\bre\.|\blinks|\brechts)",
                    ln, re.IGNORECASE)
                if carotid_diagnosis:
                    pct = re.search(r"(\d+(?:[.,]\d+)?)\s*%", ln)
                    side = ("left" if re.search(r"links|\bli\.", ln, re.IGNORECASE)
                            else "right")
                    if pct:
                        value = pct.group(1).replace(",", ".")
                        text = (f"Carotid stenosis: up to {value}% stenosis of "
                                f"the internal carotid artery, {side} side")
            if not breast_diagnosis:
                key = "dx." + _norm(text if text.startswith("Carotid stenosis:")
                                    else en or ln)
            if text.startswith("Carotid stenosis:"):
                key = "dx." + _norm(text)
            if key in facts:
                if len(facts[key].evidence) < 3:
                    facts[key].evidence.append(doc.line_span(i))
                if breast_diagnosis and _status_for_dx_line(ln) == "positive":
                    facts[key].status = "positive"
                    facts[key].date = max(facts[key].date or "", doc.date[:7])
                    facts[key].original = ln
            else:
                kw = set(re.findall(r"[a-zäöüß]+", ln.lower()))
                facts[key] = Fact(key=key, category=f"diagnosis.{sub}", text=text,
                                  original=ln,
                                  status=_status_for_dx_line(ln),
                                  date=doc.date[:7], evidence=[doc.line_span(i)], kw=kw)
    return list(facts.values())


def extract_allergies(data: PatientData) -> list[Fact]:
    facts = []
    stop = [r"sehr geehrte", r"anamnese", r"wir berichten", r"vitalparameter", r"befunde"]
    for doc in data.documents.values():
        for i, ln in _section_lines(doc, r"^allergien\s*:?$", stop):
            low = ln.casefold()
            if "km" in ln.lower() or "kontrast" in ln.lower():
                reactions = allergy_reactions_en(ln)
                text = "Contrast media allergy"
                if reactions:
                    text += f" ({', '.join(reactions)})"
            elif re.search(r"\bkeine?\b|keinerlei|no known", low):
                text = "No known allergies"
            else:
                text = "Allergy recorded"
            facts.append(Fact(
                key=f"allergy.{doc.doc_id}.{i}", category="allergy", text=text,
                original=ln,
                date=doc.date[:7], evidence=[doc.line_span(i)],
                kw={"allergy", "allergie", "contrast", "kontrast", "km", "intolerance"}))
    # dedupe identical lines (both letters list the same allergy)
    seen: dict[str, Fact] = {}
    for f in facts:
        k = _norm(f.text)
        if k in seen and len(seen[k].evidence) < 3:
            seen[k].evidence += f.evidence
        else:
            seen[k] = f
    return list(seen.values())


_VITAL_PATTERNS = [
    (r"größe\s*:\s*(\d+)\s*cm", "measurement.height", "Height {v} cm"),
    (r"gewicht\s*:\s*(\d+(?:[,.]\d+)?)\s*kg", "measurement.weight", "Weight {v} kg"),
    (r"bmi\s*:\s*(\d+(?:[,.]\d+)?)", "measurement.bmi", "BMI {v}"),
    (r"körpertemperatur\s*:\s*(\d+(?:[,.]\d+)?)", "measurement.temperature", "Body temperature {v} °C"),
    (r"mmse\s*:\s*(\d+)\s*/\s*(\d+)", "measurement.mmse", "MMSE {v}"),
]


def extract_vitals(data: PatientData) -> list[Fact]:
    facts = []
    for doc in data.documents.values():
        for i, ln in enumerate(doc.lines):
            for pat, cat, tmpl in _VITAL_PATTERNS:
                m = re.search(pat, ln, re.IGNORECASE)
                if m:
                    val = "/".join(g.replace(",", ".") for g in m.groups())
                    facts.append(Fact(
                        key=f"{cat}.{doc.doc_id}.{i}", category=cat,
                        text=tmpl.format(v=val), date=doc.date,
                        evidence=[doc.line_span(i)],
                        kw=set(cat.split(".")[1].split("_")) | {ln.split(":")[0].lower()}))
            m = re.match(r"rr\b[^:]*:\s*(\d{2,3})\s*/\s*(\d{2,3})", ln.strip(), re.IGNORECASE)
            if m:
                facts.append(Fact(
                    key=f"measurement.blood_pressure.{doc.doc_id}.{i}", category="measurement.blood_pressure",
                    text=f"Blood pressure {m.group(1)}/{m.group(2)} mmHg", date=doc.date,
                    evidence=[doc.line_span(i)], kw={"rr", "blood", "pressure", "blutdruck"}))
            m = re.match(r"hf\s*:?\s*(\d+)\s*/?\s*min", ln.strip(), re.IGNORECASE)
            if m:
                facts.append(Fact(
                    key=f"measurement.heart_rate.{doc.doc_id}.{i}", category="measurement.heart_rate",
                    text=f"Heart rate {m.group(1)}/min", date=doc.date,
                    evidence=[doc.line_span(i)], kw={"hf", "heart", "rate", "puls", "herzfrequenz"}))
    return facts


def extract_measurements_text(data: PatientData) -> list[Fact]:
    facts: list[Fact] = []
    for doc in data.documents.values():
        for i, ln in enumerate(doc.lines):
            for pat, cat, tmpl, kws in [
                (r"hba1c[^%]{0,20}?(\d+,\d+)\s*%", "measurement.hba1c",
                 "HbA1c {v}%", {"hba1c", "blood sugar", "blutzucker", "long-term"}),
                (r"\bef\s*(\d+)\s*%", "measurement.ef",
                 "Left-ventricular ejection fraction {v}%", {"ef", "ejection", "fraction", "ejektionsfraktion"}),
                (r"(\d{2,3})\s*/\s*(\d{2,3})\s*mmhg", "measurement.blood_pressure",
                 "Blood pressure {v} mmHg", {"rr", "blood", "pressure", "blutdruck"}),
                (r"tapse\s*(\d+)\s*mm", "measurement.cardiac",
                 "TAPSE {v} mm (normal right-ventricular function)", {"tapse"}),
                (r"systol\.?\s*pap\s*(\d+)\s*mm", "measurement.cardiac",
                 "Systolic pulmonary-artery pressure {v} mmHg (no pulmonary hypertension)", {"pap"}),
                (r"flussgeschwindigkeit von\s*(\d+)\s*cm", "measurement.vascular",
                 "Flow velocity {v} cm/s in the internal carotid artery", {"flow", "velocity", "flussgeschwindigkeit"}),
                (r"mittelwert\s*:\s*(\d+/\d+)", "measurement.blood_pressure",
                 "24h mean blood pressure {v} mmHg", {"24h", "rr", "ambulatory", "blood", "pressure"}),
            ]:
                for m in re.finditer(pat, ln, re.IGNORECASE):
                    v = "/".join(g.replace(",", ".") for g in m.groups())
                    facts.append(Fact(
                        key=f"{cat}.{doc.doc_id}.{i}.{m.start()}", category=cat,
                        text=tmpl.format(v=v), date=doc.date,
                        evidence=[_span(doc, doc.line_offsets[i] + m.start(), doc.line_offsets[i] + m.end())],
                        kw=kws))
        for m in re.finditer(
            r"\bEKG:\s*(?:\n\s*)?HF\s*:?\s*(\d+)\s*/\s*min\s*,?\s*Normaltyp",
            doc.text, re.IGNORECASE):
            facts.append(Fact(
                key=f"measurement.cardiac.ecg.{doc.doc_id}.{m.start()}",
                category="measurement.cardiac",
                text=f"ECG: heart rate {m.group(1)}/min, normal axis (Normaltyp)",
                date=doc.date, evidence=[_span(doc, m.start(), m.end())],
                kw={"ecg", "ekg", "heart", "rate", "cardiac"}))
        for m in re.finditer(
            r"(?:durchgängiger|kontinuierlicher)\s+sinusrhythmus"
            r"(?:\s+mit\s+hf\s+von\s+(\d+)\s*-\s*(\d+)\s*\((\d+)\)\s*/min)?",
            doc.text, re.IGNORECASE):
            rates = m.groups()
            rate_text = (f", heart rate {rates[0]}–{rates[1]} ({rates[2]})/min"
                         if all(rates) else "")
            facts.append(Fact(
                key=f"measurement.cardiac.sinus.{doc.doc_id}.{m.start()}",
                category="measurement.cardiac",
                text=f"Long-term ECG: continuous sinus rhythm{rate_text}",
                date=doc.date, evidence=[_span(doc, m.start(), m.end())],
                kw={"ecg", "ekg", "sinus", "rhythm", "cardiac"}))
        # qualitative echo result
        m = re.search(
            r"normale systolische lv-funktion \(ef\s*(\d+)%\)",
            doc.text, re.IGNORECASE)
        if m:
            facts.append(Fact(
                key=f"measurement.ef.qual.{doc.doc_id}", category="measurement.ef",
                text=f"Left-ventricular ejection fraction {m.group(1)}%",
                date=doc.date, evidence=[_span(doc, m.start(), m.end())],
                kw={"ef", "ejection", "fraction", "echo", "cardiac", "lv", "function", "systolic"}))
    return facts


_BRAND = {
    "thrombo ass": "Thrombo ASS", "ass": "ASS", "acetylsalicyl": "ASS",
    "aspirin": "Aspirin", "glucophage": "Glucophage", "metformin": "Glucophage",
    "actos": "Actos", "pioglitazon": "Actos", "pioglitazone": "Actos",
    "byetta": "Byetta", "exenatid": "Byetta", "exenatide": "Byetta",
    "blopress": "Blopress", "candesartan": "Blopress", "zocor": "Zocor",
    "simvastatin": "Zocor", "oleovit": "Oleovit D3", "vitamin d": "Oleovit D3",
    "alendronat": "Alendronat", "alendronate": "Alendronat",
    "nitro": "Nitroglycerin", "nitroglycerin": "Nitroglycerin",
}


def _med_row_text(drug: str, row: str) -> str:
    inn, cls, _ = DRUGS[drug]
    sched = _schedule_en(row) or ""
    strength = ""
    m = re.search(r"(\d+(?:[,.]\d+)?)\s*(mg|mcg|µg)\b", row.lower())
    if m:
        strength = f" {m.group(1)} {m.group(2).replace('mcg', 'µg')}"
    elif "tropfen" in row.lower():
        strength = " drops"
    brand = _BRAND.get(drug, drug.title())
    text = f"{brand} ({inn}){strength}"
    if sched:
        text += f" — {sched}"
    return text


def extract_medication(data: PatientData) -> list[Fact]:
    """Medication-table rows plus in-text drug mentions; deduped by drug.

    For drugs on both letters the newest row is the headline text and an
    earlier different regimen is appended as a note.
    """
    rows: dict[str, dict] = {}      # drug -> {date: (text, [spans]), 'kw': set}
    mentions: dict[str, dict] = {}  # drug -> {'spans': [], 'label': str, 'date': str}

    for doc in data.documents.values():
        in_table = False
        for i, ln in enumerate(doc.lines):
            if re.match(r"^\s*entlassungsmedikation", ln, re.IGNORECASE):
                in_table = True
                continue
            if in_table:
                s = ln.strip()
                if not s:
                    in_table = False
                    continue
                if re.match(r"(applikation|mit freundlichen|laborbefund)", s, re.IGNORECASE):
                    if re.match(r"applikation", s, re.IGNORECASE):
                        continue
                    in_table = False
                    continue
                low = s.lower()
                drug = next((d for d in DRUG_NAMES if d in low), None)
                if not drug:
                    continue
                inn, cls, cls_en = DRUGS[drug]
                rec = rows.setdefault(drug, {"dates": {}, "kw": {
                    drug, inn, cls, cls_en, "medication", "medikation", "drug",
                    *inn.split()}})
                if cls == "antidiabetic":
                    rec["kw"] |= {"glucose", "diabetes", "blutzucker", "sugar"}
                if cls == "antihypertensive":
                    rec["kw"] |= {"blood", "pressure", "blutdruck", "hypertonie"}
                if cls == "antithrombotic":
                    rec["kw"] |= {"blood thinner", "anticoag", "antiplatelet",
                                  "gerinnung", "thinner", "platelet"}
                rec["dates"].setdefault(doc.date[:7], []).append(
                    (_med_row_text(drug, s), doc.line_span(i)))
        # in-text mentions (therapy decisions, acute medication)
        for m in re.finditer(
            r"(thrombo ass\s*\d+\s*mg|blopress\s*\d+\s*mg|\d+\s*hübe\s*nitro)",
            doc.text, re.IGNORECASE,
        ):
            frag = m.group(0)
            drug = next((d for d in DRUG_NAMES if d in frag.lower()), None)
            if not drug:
                continue
            rec = mentions.setdefault(drug, {"spans": [], "date": doc.date[:7], "frag": frag})
            rec["spans"].append(_span(doc, m.start(), m.end()))
            if doc.date[:7] > rec["date"]:
                rec["date"] = doc.date[:7]
                rec["frag"] = frag

    facts: list[Fact] = []
    for drug, rec in rows.items():
        inn, cls, _ = DRUGS[drug]
        dates = sorted(rec["dates"])
        latest = dates[-1]
        text = rec["dates"][latest][0][0]
        spans = [sp for d in dates for _, sp in rec["dates"][d]]
        earlier = [t for d in dates[:-1] for t, _ in rec["dates"][d] if t != text]
        if earlier:
            sig = re.search(r"\d+(?:/\d+)?\s*-\s*\d+(?:/\d+)?\s*-\s*\d+(?:/\d+)?(?:\s*-\s*\d+(?:/\d+)?)?",
                            earlier[-1])
            note = sig.group(0).replace(" ", "") if sig else earlier[-1]
            text += f" (in the {en_month(dates[0])} letter: {note})"
        if drug in mentions:
            spans += mentions[drug]["spans"]
        facts.append(Fact(
            key=f"med.{drug}", category=f"medication.{cls}", text=text,
            date=latest, evidence=spans[:4], kw=rec["kw"], derived=True))
    for drug, rec in mentions.items():
        if drug in rows:
            continue  # spans already merged into the table fact
        inn, cls, _ = DRUGS[drug]
        frag = rec["frag"]
        puffs = re.search(r"(\d+)\s*hübe", frag, re.IGNORECASE)
        label = "Nitroglycerin spray"
        if puffs:
            label += f", {puffs.group(1)} puffs"
        if "nitro" in drug:
            label = label or frag[0].upper() + frag[1:]
        facts.append(Fact(
            key=f"med.{drug}", category=f"medication.{cls}", text=label,
            date=rec["date"], original=frag, evidence=rec["spans"],
            kw={drug, inn, "medication"}, derived=True))
    return facts


def extract_carotid(data: PatientData) -> list[Fact]:
    facts = []
    for doc in data.documents.values():
        for m in re.finditer(
            r"(maximal\s*)?(\d+)\s*%?\s*ige?\s*(abgangs)?stenose[^.\n]*carotis[^.\n]*"
            r"|carotis[^.\n]*stenose[^.\n]*"
            r"|(\d+)\s*%\s*karotisstenose[^.\n]*",
            doc.text, re.IGNORECASE,
        ):
            frag = m.group(0)
            low = frag.lower()
            if frag.rstrip().endswith("?") or len(frag) < 25:
                continue  # referral question, not a finding
            pct = re.search(r"(\d+)\s*%", frag)
            side = _side(frag)
            values = []
            if pct:
                values.append(f"{pct.group(1)}%")
            if side:
                values.append(side)
            facts.append(Fact(
                key=f"vascular.carotid.{doc.doc_id}.{m.start()}", category="diagnosis.vascular",
                text=("Carotid artery stenosis" +
                      (f" — {', '.join(values)}" if values else "")),
                date=doc.date[:7], original=doc.quote(_span(doc, m.start(), m.end())),
                evidence=[_span(doc, m.start(), m.end())],
                kw={"carotid", "karotis", "stenosis", "stenose", "artery",
                    "duplex", *(values or [])}))
    return facts


def extract_histology(data: PatientData) -> list[Fact]:
    facts = []
    receptor_pattern = re.compile(r"(östrogen|progesteron)rezeptor", re.IGNORECASE)
    component_pattern = re.compile(
        r"(?:östrogen|progesteron)rezeptor|her-?2\w*|ki-?67",
        re.IGNORECASE,
    )
    for doc in data.documents.values():
        is_report = (
            re.search(r"histolog|patholog", doc.title, re.IGNORECASE)
            or re.search(
                r"immunhistologisches profil|histopathologischer befund|"
                r"entnahmedatum",
                doc.text, re.IGNORECASE)
        )
        if not is_report:
            continue

        specimen_match = re.search(
            r"entnahmedatum\s*:?\s*(\d{1,2}\.\d{1,2}\.\d{4})",
            doc.text, re.IGNORECASE)
        specimen_date = (
            _dates_in(specimen_match.group(1))[0][2]
            if specimen_match and _dates_in(specimen_match.group(1))
            else doc.date)

        type_spans = _regex_spans(doc, r"invasiv duktal\w*[^.\n]*")
        span = next((item for item in type_spans
                     if _side(doc.quote(item))),
                    type_spans[0] if type_spans else None)
        if span:
            quote = doc.quote(span)
            histology_type = next(
                (label for pattern, label in DIAGNOSIS_EN
                 if re.search(pattern, quote, re.IGNORECASE)),
                None,
            )
            if not histology_type:
                continue
            side = _side(quote)
            text = f"Histology: {histology_type}"
            if side:
                text += f" of the {side} breast"
            facts.append(Fact(
                key=f"onco.type.{doc.doc_id}.{span.start}",
                category="diagnosis.oncology", text=text, date=specimen_date,
                original=quote, evidence=[span],
                kw={"histology", "histologie", "invasive", "ductal",
                    "carcinoma", "breast", "karzinom", "mamma", "tumor"},
            ))

        for match in _TNM_PATTERN.finditer(doc.text):
            qualifiers = re.findall(
                r"([VLRG])\s*([0-9X])", match.group(4), re.IGNORECASE)
            components = [match.group(1), match.group(2), match.group(3)]
            components.extend(f"{letter.upper()}{value.upper()}"
                              for letter, value in qualifiers)
            tnm = " ".join(components)
            span = doc.sentence_span(match.start(), match.end())
            facts.append(Fact(
                key=f"onco.tnm.{doc.doc_id}.{match.start()}",
                category="diagnosis.oncology", text=f"TNM stage: {tnm}",
                date=specimen_date, original=doc.quote(span), evidence=[span],
                kw={"tnm", "stage", "staging", *(part.casefold()
                     for part in components), "tumor", "lymph", "spread"},
            ))

        for match in receptor_pattern.finditer(doc.text):
            following = component_pattern.search(doc.text, match.end())
            line_end = doc.text.find("\n", match.end())
            end = min(
                position for position in (
                    following.start() if following else len(doc.text),
                    line_end if line_end >= 0 else len(doc.text),
                    len(doc.text),
                )
            )
            segment = doc.text[match.start():end]
            receptor = ("Oestrogen" if match.group(1).casefold() == "östrogen"
                        else "Progesterone")
            status_match = re.search(r"\b(positiv|negativ)\b", segment,
                                     re.IGNORECASE)
            status = ({"positiv": "positive", "negativ": "negative"}.get(
                status_match.group(1).casefold()) if status_match else None)
            details = []
            percentage = re.search(r"(\d+)\s*%", segment)
            allred = re.search(r"allred[^\d]{0,15}(\d+)", segment,
                               re.IGNORECASE)
            if percentage:
                details.append(f"{percentage.group(1)}% of tumour cells")
            if allred:
                details.append(f"Allred score {allred.group(1)}")
            text = f"{receptor} receptor"
            if status:
                text += f" {status}"
            if details:
                text += f" ({', '.join(details)})"
            span = doc.sentence_span(match.start(), end)
            facts.append(Fact(
                key=f"onco.{match.group(1).casefold()}.{doc.doc_id}.{match.start()}",
                category="diagnosis.oncology", text=text, date=specimen_date,
                original=doc.quote(span), evidence=[span], status=status,
                kw={"estrogen", "oestrogen", "progesterone", "receptor",
                    "hormone", "er", "pr", "allred"},
            ))

        for match in re.finditer(r"her-?2\w*", doc.text, re.IGNORECASE):
            next_component = re.search(r"ki-?67", doc.text[match.end():],
                                       re.IGNORECASE)
            line_end = doc.text.find("\n", match.end())
            end = min(
                position for position in (
                    match.end() + next_component.start()
                    if next_component else len(doc.text),
                    line_end if line_end >= 0 else len(doc.text),
                    len(doc.text),
                )
            )
            segment = doc.text[match.start():end]
            status_match = re.search(r"\b(positiv|negativ)\b", segment,
                                     re.IGNORECASE)
            status = ({"positiv": "positive", "negativ": "negative"}.get(
                status_match.group(1).casefold()) if status_match else None)
            score = re.search(r"([0-3])\s*\+", segment)
            text = "HER2"
            if status:
                text += f" {status}"
            if score:
                text += f" ({score.group(1)}+)"
            span = doc.sentence_span(match.start(), end)
            facts.append(Fact(
                key=f"onco.her2.{doc.doc_id}.{match.start()}",
                category="diagnosis.oncology", text=text, date=specimen_date,
                original=doc.quote(span), evidence=[span], status=status,
                kw={"her2", "her2neu", "receptor"},
            ))

        for match in re.finditer(
                r"ki-?67[^\d]{0,30}(\d+)\s*%", doc.text, re.IGNORECASE):
            span = doc.sentence_span(match.start(), match.end())
            facts.append(Fact(
                key=f"onco.ki67.{doc.doc_id}.{match.start()}",
                category="diagnosis.oncology",
                text=f"Ki-67 proliferation rate {match.group(1)}%",
                date=specimen_date, original=doc.quote(span), evidence=[span],
                kw={"ki67", "ki-67", "proliferation", "rate", "divide"},
            ))

        if specimen_match:
            span = doc.sentence_span(specimen_match.start(), specimen_match.end())
            facts.append(Fact(
                key=f"onco.material.{doc.doc_id}.{specimen_match.start()}",
                category="procedure.biopsy",
                text=f"Biopsy specimen collected {_display_date(specimen_date)}",
                date=specimen_date, original=doc.quote(span), evidence=[span],
                kw={"biopsy", "stanzbiopsie", "core", "specimen", "entnahme"},
            ))
    return facts


_EXAM_PATTERNS = [
    (r"schädel-?ct[^.\n]*", "procedure.imaging", "Head CT",
     {"ct", "schädel", "head", "brain", "native", "kontrast", "contrast"}),
    (r"us hirnversorgende arterien[^.\n]*", "procedure.imaging", "Cerebral artery ultrasound",
     {"duplex", "doppler", "ultrasound", "sonographie", "carotid", "karotis"}),
    (r"langzeit-ekg[^.\n]*", "procedure.cardiac", "Long-term ECG",
     {"ekg", "ecg", "long-term", "langzeit", "arrhythmia"}),
    (r"24h-rr[^.\n]*", "procedure.cardiac", "24-hour blood-pressure monitoring",
     {"24h", "rr", "blood pressure", "ambulatory"}),
    (r"transthorakale[sn]?\s+echokardiographie[^.\n]*", "procedure.cardiac", "Transthoracic echocardiography",
     {"echo", "echocardiography", "herzecho", "cardiac"}),
    (r"konsil für augenheilkunde[^.\n]*|augenkonsil[^.\n]*", "procedure.ophthalmology", "Ophthalmology consultation",
     {"eye", "auge", "ophthalmolog", "konsil", "fundus", "retina"}),
    (r"mammograph\w*[^.\n]*|vorsorge-mammographie[^.\n]*", "procedure.imaging",
     "Mammography",
     {"mammography", "mammographie", "screening", "breast", "mamma"}),
    (r"ultraschalluntersuchung der [^\n]*brust[^.\n]*", "procedure.imaging",
     "Breast ultrasound",
     {"ultrasound", "sonographie", "breast", "birads", "axilla", "lymph"}),
    (r"bi-?rads?\s*[IV0-6]+", "diagnosis.oncology",
     "BI-RADS",
     {"birads", "bi-rads", "breast", "malign"}),
]


def extract_procedures(data: PatientData) -> list[Fact]:
    facts = []
    for doc in data.documents.values():
        for pat, cat, text, kws in _EXAM_PATTERNS:
            matches = list(re.finditer(pat, doc.text, re.IGNORECASE))[:2]
            header_matches = [
                match for match in matches
                if text in {"Head CT", "Cerebral artery ultrasound"}
                and _line_for_offset(doc, match.start()).rstrip().endswith(":")
            ]
            for match in matches:
                line = _line_for_offset(doc, match.start())
                is_header = line.rstrip().endswith(":")
                is_us_header = (
                    text == "Cerebral artery ultrasound"
                    and re.match(r"^\s*us hirnversorgende arterien\b", line,
                                 re.IGNORECASE))
                if text == "Head CT" and header_matches and not is_header:
                    continue
                sp = (_continued_header_span(doc, match.start(), text)
                      if is_header or is_us_header
                      else _multiline_sentence_span(
                          doc, match.start(), match.end()))
                key = f"{cat}.{pat[:18]}.{doc.doc_id}"
                quote = doc.quote(sp)
                dates = _dates_in(quote)
                fact_date = dates[0][2] if dates else doc.date
                evidence = [sp]
                original = quote
                fact_text = _procedure_fact_text(text, quote, fact_date)
                if text == "Head CT" and is_header:
                    result = re.search(
                        r"\bkeine\s+(?:anzeichen\s+für\s+(?:eine\s+)?)?blutung\b",
                        doc.text, re.IGNORECASE)
                    if result:
                        result_span = _multiline_sentence_span(
                            doc, result.start(), result.end())
                        if (result_span.start, result_span.end) != (
                                sp.start, sp.end):
                            evidence.append(result_span)
                            result_quote = doc.quote(result_span)
                            original += " / " + result_quote
                            for finding in finding_translations_en(result_quote):
                                if finding.casefold() not in fact_text.casefold():
                                    fact_text += f", {finding}"
                status = ("negative" if finding_translations_en(original)
                          else "positive")
                facts.append(Fact(
                    key=f"{key}.{sp.start}", category=cat, text=fact_text,
                    date=fact_date, original=original, status=status,
                    evidence=evidence, kw=set(kws)))
    seen = {}
    for f in facts:
        k = (f.category, f.text, f.date)
        if k in seen:
            seen[k].evidence += f.evidence
            if (f.date or "") > (seen[k].date or ""):
                seen[k].date = f.date
                seen[k].status = f.status
        else:
            seen[k] = f
    return list(seen.values())


def extract_plans(data: PatientData) -> list[Fact]:
    """Extract recommendations and link planned procedures to later reports."""
    facts = []
    rules = [
        (r"nächste geplant\w* duplexsonographie[^.\n]*|"
         r"duplexsonographie der karotisgefäße[^.\n]*",
         "plan.followup.carotid_duplex", "Carotid duplex ultrasound follow-up",
         {"carotid", "duplex", "karotis", "follow-up", "kontrolle", "ultrasound"}),
        (r"regelmäßige augenkontrollen[^.\n]*|regelmäßige augenärztliche kontrollen[^.\n]*",
         "plan.followup.eye", "Regular ophthalmology check-ups",
         {"eye", "augen", "ophthalmolog", "check", "kontrolle"}),
        (r"regelmäßige blutdruckkontrollen[^.\n]*",
         "plan.followup.bp", "Regular blood-pressure checks",
         {"blood pressure", "blutdruck", "gp", "hausarzt", "kontrolle", "medication"}),
        (r"sonographie der brust\+axilla[^.\n]*|ultraschallgezielte stanzbiopsie[^.\n]*geplant[^.\n]*|stanzbiopsie geplant",
         "plan.done.biopsy", "Breast ultrasound and core biopsy",
         {"biopsy", "stanzbiopsie", "ultrasound", "sonographie", "planned"}),
        (r"stationäre aufnahme[^.\n]*terminisiert[^.\n]*",
         "admin.admission.planned", "Inpatient admission scheduled",
         {"admission", "aufnahme", "scheduled", "terminisiert"}),
        (r"nachfolgende procedere[^.\n]*tumorboard[^.\n]*|"
         r"tumorboard[^.\n]*besprochen[^.\n]*|tumorboard besprochen[^.\n]*",
         "plan.done.tumorboard", "Tumour board discussion documented",
         {"tumorboard", "tumour board"}),
        (r"quadrantenresektion[^.\n]*",
         "plan.surgery", "Quadrantectomy with intraoperative radiotherapy",
         {"quadrantectomy", "quadrantenresektion", "iort", "surgery",
          "operation", "resection"}),
    ]
    for doc in data.documents.values():
        for pat, key, label, kws in rules:
            for sp in _regex_spans(doc, pat)[:2]:
                quote = doc.quote(sp)
                dates = _dates_in(quote)
                event_date = dates[0][2] if dates else doc.date
                category = ".".join(key.split(".")[:2])
                text = _source_fact_text(label, quote, event_date)
                evidence = [sp]
                status = _plan_status(category, event_date)
                if category in {"plan.surgery", "admin.admission"}:
                    status = "planned"
                if category == "plan.done" and "biopsy" in key:
                    performed = _later_procedure_report(
                        data, doc, r"biops\w*|stanz\w*")
                    if performed:
                        performed_date, performed_span = performed
                        evidence.append(performed_span)
                        text += f" — later documented as performed {_display_date(performed_date)}"
                        event_date = performed_date
                        status = "completed"
                    else:
                        text += " — planned"
                        status = "planned"
                if category == "plan.done" and "tumorboard" in key:
                    status = "completed"
                facts.append(Fact(
                    key=f"{key}.{doc.doc_id}.{sp.start}", category=category,
                    text=text, date=event_date, status=status,
                    original=" / ".join(data.documents[item.doc_id].quote(item)
                                        for item in evidence),
                    evidence=evidence, kw=set(kws)))
    seen = {}
    for f in facts:
        k = (f.category, f.text)
        if k in seen:
            seen[k].evidence += f.evidence
            if (f.date or "") > (seen[k].date or ""):
                seen[k].date = f.date
                seen[k].status = f.status
        else:
            seen[k] = f
    return list(seen.values())


def _later_procedure_report(
        data: PatientData, source_doc: Document, procedure_pattern: str
        ) -> tuple[str, EvidenceSpan] | None:
    completion = re.compile(
        r"erfolgte|durchgeführt|vorgenommen|wurde .* biops",
        re.IGNORECASE,
    )
    for doc in sorted(data.documents.values(), key=lambda item: item.date):
        if doc.date <= source_doc.date:
            continue
        for match in re.finditer(procedure_pattern, doc.text, re.IGNORECASE):
            span = doc.sentence_span(match.start(), match.end())
            quote = doc.quote(span)
            if not completion.search(quote):
                continue
            dates = _dates_in(doc.text)
            if dates:
                date = min(dates, key=lambda entry: abs(entry[0] - match.start()))[2]
            else:
                date = doc.date
            return date, span
    return None


def extract_history(data: PatientData) -> list[Fact]:
    """Translate source-backed historical events without fixed patient values."""
    facts = []
    rules = [
        (r"hypertensive krise|blutdruckentgleisung",
         "Hypertensive crisis",
         {"hypertensive", "crisis", "dizziness", "schwindel", "vision",
          "sehstörung", "episode", "previous"}),
        (r"amaurosis fugax", "Amaurosis fugax",
         {"amaurosis", "fugax", "vision", "eye", "transient", "blindness"}),
        (r"zurückgeführt", "Symptoms attributed to carotid stenosis",
         {"cause", "ursache", "carotid", "karotis", "stenosis", "attributed"}),
        (r"\bnitro\w*|\bnitroglycerin\w*",
         "Emergency treatment with nitroglycerin",
         {"emergency", "treatment", "nitro", "medication"}),
    ]
    for doc in data.documents.values():
        for pattern, label, keywords in rules:
            for span in _multiline_spans(doc, pattern)[:2]:
                quote = doc.quote(span)
                if label == "Emergency treatment with nitroglycerin":
                    if not re.search(
                            r"verabreicht|bekam|verabreichung|notarzt|\bna\b",
                            quote, re.IGNORECASE):
                        continue
                source_dates = _dates_in(quote)
                if source_dates:
                    fact_date = source_dates[0][2]
                elif doc_type_label(doc.title) == "Discharge letter":
                    admissions = _event_dates(doc, "admission")
                    fact_date = admissions[0][0] if admissions else doc.date
                else:
                    fact_date = doc.date
                date_label = _display_date(fact_date)
                values = [value for value in _parsed_source_values(
                    quote, fact_date) if value != date_label]
                if label == "Hypertensive crisis":
                    pressures = re.findall(
                        r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg",
                        quote, re.IGNORECASE)
                    values = []
                    if pressures:
                        systolic, diastolic = max(
                            ((int(sys), int(dia)) for sys, dia in pressures))
                        values.append(
                            f"peak BP {systolic}/{diastolic} mmHg")
                    symptoms = []
                    if re.search(r"schwindel", quote, re.IGNORECASE):
                        symptoms.append("dizziness")
                    if re.search(r"seh\w*|amaurosis", quote, re.IGNORECASE):
                        symptoms.append("vision symptoms")
                    if symptoms:
                        values.append(", ".join(symptoms))
                if label == "Emergency treatment with nitroglycerin":
                    dose = re.search(
                        r"\b(\d+)\s*hüb(?:e|en)?\b", quote, re.IGNORECASE)
                    text = (f"{label} — {dose.group(1)} puffs of "
                            "nitroglycerin spray" if dose else label)
                else:
                    text = label + (f" — {'; '.join(dict.fromkeys(values))}"
                                    if values else "")
                facts.append(Fact(
                    key=f"hx.{doc.doc_id}.{span.start}",
                    category="history.event", status="historical",
                    text=text, date=fact_date,
                    original=quote, evidence=[span], kw=set(keywords),
                ))
    return facts


def extract_admin(data: PatientData) -> list[Fact]:
    """Administrative passages (dates, case ids) — feed contradiction checks."""
    facts = []
    for doc in data.documents.values():
        for pat, key, label in [
            (r"fall\s*:\s*\d{6,}", "admin.case_id", "Case id"),
            (r"aufnahme am\s*\d{1,2}/\d{1,2}/\d{2,4}", "admin.admission", "Admission date (letter header)"),
            (r"aufgenommen am\s*\d{1,2}\.\d{1,2}\.\d{4}", "admin.admission", "Admission date (text)"),
            (r"entlassen am\s*[\d/]+", "admin.discharge", "Discharge date (letter header)"),
            (r"entlassen[^.\n]*am\s*\d{1,2}\.\d{1,2}\.\d{4}", "admin.discharge", "Discharge date (text)"),
            (r"entnahmedatum\s*:\s*\d{1,2}\.\d{1,2}\.\d{4}", "admin.biopsy_date", "Specimen collection date"),
            (r"stationäre aufnahme[^.\n]*\d{1,2}\.\d{1,2}\.\d{4}[^.\n]*", "admin.admission", "Planned admission date"),
        ]:
            for sp in _regex_spans(doc, pat):
                frag = doc.quote(sp)
                facts.append(Fact(
                    key=f"{key}.{doc.doc_id}.{sp.start}", category=key,
                    text=f"{label}: {frag}", date=doc.date[:7], evidence=[sp],
                    kw={"admin", "date", "aufnahme", "entlassen", "fall"}))
    return facts


def extract_breast_findings(data: PatientData) -> list[Fact]:
    facts = []
    for doc in data.documents.values():
        for pat, label, kws in [
            (r"vergrößerte lymphknoten[^.\n]*",
             "Enlarged axillary lymph nodes",
             {"lymph", "lymphknoten", "axilla", "node"}),
            (r"keine vergrößerten lymphknoten[^.\n]*",
             "Enlarged axillary lymph nodes",
             {"lymph", "lymphknoten", "axilla", "node"}),
            (r"ausdehnung des tumors[^.\n]*|maximale ausdehnung[^.\n]*",
             "Tumour extent",
             {"tumor", "extent", "ausdehnung", "size"}),
            (r"knoten im äußeren oberen quadranten[^.\n]*",
             "Palpable breast lump",
             {"lump", "knoten", "quadrant", "palpable"}),
        ]:
            for sp in _multiline_spans(doc, pat)[:2]:
                quote = doc.quote(sp)
                text = _source_fact_text(label, quote, doc.date)
                st = ("negative" if re.search(
                    r"\b(kein|keine|ohne|nicht)\b", quote, re.IGNORECASE)
                      else "positive")
                source_dates = _dates_in(quote)
                facts.append(Fact(
                    key=f"breast.{doc.doc_id}.{sp.start}", category="finding.breast",
                    text=text, status=st,
                    date=source_dates[0][2] if source_dates else doc.date[:7],
                    original=quote, evidence=[sp], kw=set(kws)))
    return facts


def build_fact_store(data: PatientData) -> list[Fact]:
    facts: list[Fact] = []
    facts += extract_demographics(data)
    facts += extract_diagnoses(data)
    facts += extract_allergies(data)
    facts += extract_vitals(data)
    facts += extract_measurements_text(data)
    facts += extract_medication(data)
    facts += extract_carotid(data)
    facts += extract_histology(data)
    facts += extract_procedures(data)
    facts += extract_plans(data)
    facts += extract_history(data)
    facts += extract_admin(data)
    facts += extract_breast_findings(data)
    # global dedupe on identical text+category keeping first (spans merged already)
    seen = {}
    for f in facts:
        f.evidence = _unique_spans(f.evidence)
        date_key = (
            f.date if f.category.startswith("procedure.")
            or f.category == "history.event" else None)
        k = (f.category, _norm(f.text), date_key)
        if k in seen:
            seen[k].evidence = _unique_spans(seen[k].evidence + f.evidence)
            if (f.date or "") > (seen[k].date or ""):
                seen[k].date = f.date
                seen[k].status = f.status
        else:
            seen[k] = f
    return list(seen.values())


def _unique_spans(spans: list[EvidenceSpan]) -> list[EvidenceSpan]:
    unique = {}
    for span in spans:
        unique.setdefault((span.doc_id, span.start, span.end), span)
    return list(unique.values())
