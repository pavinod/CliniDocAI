"""Rule-based contradiction detection — dynamically over the documents.

Each detector collects the passages that talk about the same thing and checks
whether they can all be true, then links every passage involved and explains
the incompatibility. Deliberately conservative: pure measurement drift over
time and differences between document text and the JSON index are not flagged.
No detector encodes patient-specific answers — they only match document
patterns (laterality words, dates, IDs, negated findings) and compare them.
"""

from __future__ import annotations

import re

from .labels import doc_type_label
from .pipeline import Document, PatientData
from .schemas import Contradiction, EvidenceSpan


def _spans(doc: Document, pattern: str) -> list[EvidenceSpan]:
    return [doc.span(m.start(), m.end()) for m in re.finditer(pattern, doc.text, re.IGNORECASE)]


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _contra(key, title, detail, kind, spans) -> list[Contradiction]:
    if len(spans) < 2:
        return []
    return [Contradiction(key=key, title=title, detail=detail, kind=kind,
                          evidence=spans)]


# --------------------------------------------------------------- detectors

def check_bmi(data: PatientData) -> list[Contradiction]:
    """Documented BMI vs the height and weight stated in the same letter."""
    out: list[Contradiction] = []
    for doc in data.documents.values():
        h = re.search(r"größe\s*:\s*(\d+)\s*cm", doc.text, re.IGNORECASE)
        w = re.search(r"gewicht\s*:\s*(\d+(?:[,.]\d+)?)\s*kg", doc.text, re.IGNORECASE)
        b = re.search(r"bmi\s*:\s*(\d+(?:[,.]\d+)?)", doc.text, re.IGNORECASE)
        if not (h and w and b):
            continue
        calc = round(_num(w.group(1)) / ((_num(h.group(1)) / 100.0) ** 2), 1)
        if abs(calc - _num(b.group(1))) >= 0.5:
            out += _contra(
                f"bmi.{doc.doc_id}",
                f"BMI does not match height and weight "
                f"({doc_type_label(doc.title)}, {doc.date[:7]})",
                f"The letter states {_num(w.group(1)):g} kg and {h.group(1)} cm, which gives a BMI "
                f"of {calc} kg/m² — but documents {_num(b.group(1)):g}.",
                "numeric",
                [doc.span(m.start(), m.end()) for m in (h, w, b)])
    return out


_LATERALITY_PAT = re.compile(
    r"(?:links|rechts|li\.|re\.)\s+findet sich.{0,60}?carotis interna[^\n]*|"
    r"arteria carotis interna\s*(?:links|rechts|li\.|re\.)\b[^\n]*",
    re.IGNORECASE)


def check_laterality(data: PatientData) -> list[Contradiction]:
    """Conflicting left/right statements about the same finding.

    Collects every laterality-bearing vascular-finding passage; conflicting
    sides for the same structure (carotid) are flagged. Same-side repeats are
    not flagged, and unrelated laterality mentions are ignored.
    """
    claims: list[tuple[EvidenceSpan, str]] = []
    for doc in data.documents.values():
        for m in _LATERALITY_PAT.finditer(doc.text):
            frag = m.group(0).lower()
            side_word = re.search(r"links|rechts|li\.|re\.", frag)
            side = "left" if side_word.group(0).startswith(("l", "li")) else "right"
            claims.append((doc.span(m.start(), m.end()), side))
    sides = {c[1] for c in claims}
    if len(sides) < 2:
        return []
    side_text = " vs ".join(sorted(sides))
    return _contra(
        "vascular.laterality",
        f"Side of the carotid stenosis: {side_text}",
        f"Conflicting sides are documented for the same carotid finding: {side_text}.",
        "laterality",
        [c[0] for c in claims])


def check_conflicting_dates(data: PatientData) -> list[Contradiction]:
    """Flag distinct admission days reported for the same month-long stay."""
    claims: list[tuple[str, EvidenceSpan]] = []
    pat = re.compile(
        r"aufnahme am\s*(\d{1,2})[./](\d{1,2})[./](\d{2,4})|"
        r"am\s*(\d{1,2})\.(\d{1,2})\.(\d{4})[^.\n]*?aufgenommen|"
        r"aufgenommen[^.\n]*?am\s*(\d{1,2})\.(\d{1,2})\.(\d{4})|"
        r"stationäre aufnahme[^.\n]*?(\d{1,2})\.(\d{1,2})\.(\d{4})",
        re.IGNORECASE)
    for doc in data.documents.values():
        for m in pat.finditer(doc.text):
            groups = m.groups()
            day, month, year = next(
                (groups[i], groups[i + 1], groups[i + 2])
                for i in range(0, len(groups), 3) if groups[i])
            year = f"20{year}" if len(year) == 2 else year
            date = f"{int(day):02d}.{int(month):02d}.{year}"
            claims.append((date, doc.span(m.start(), m.end())))

    by_stay: dict[str, list[tuple[str, EvidenceSpan]]] = {}
    for claim in claims:
        by_stay.setdefault(claim[0][-4:] + "-" + claim[0][3:5], []).append(claim)

    out = []
    for stay, group in by_stay.items():
        dates = sorted({claim[0] for claim in group})
        if len({date[:2] for date in dates}) < 2:
            continue
        out += _contra(
            f"admin.admission_date.{stay}",
            "Conflicting admission dates for the same stay",
            f"The stay is documented with different admission dates: {' / '.join(dates)}.",
            "date",
            [claim[1] for claim in group if claim[0] in dates])
    return out


def check_case_ids(data: PatientData) -> list[Contradiction]:
    """Documents of the same hospital stay carrying different case numbers.

    Groups 'Fall: <id>' identifiers by the stay they belong to (same
    admission month); different numbers inside one stay are flagged.
    """
    claims: list[tuple[str, EvidenceSpan, str, str]] = []
    for doc in data.documents.values():
        for m in re.finditer(r"fall\s*:\s*(\d{6,})", doc.text, re.IGNORECASE):
            label = f"{doc_type_label(doc.title)}, {doc.date[:7]}"
            claims.append((m.group(1), doc.span(m.start(), m.end()), label,
                           doc.date[:7]))
    # group by the month of the stay a doc reports on (doc date month)
    by_month: dict[str, list] = {}
    for c in claims:
        by_month.setdefault(c[3], []).append(c)
    out: list[Contradiction] = []
    for month, group in by_month.items():
        if len({g[0] for g in group}) >= 2:
            desc = " vs ".join(f"'{g[0]}' ({g[2]})" for g in group)
            out += _contra(
                f"admin.case_id.{month}",
                "Conflicting case numbers for the same hospital stay",
                f"Documents of the same stay are filed under different case numbers: {desc}.",
                "identity",
                [g[1] for g in group])
    return out


def check_event_dates(data: PatientData) -> list[Contradiction]:
    """A recorded procedure/collection whose own dates disagree.

    Compares the specimen-collection date (Entnahmedatum) with the date the
    same biopsy/exam is recorded on in other letters.
    """
    claims: list[tuple[str, EvidenceSpan, str]] = []
    pats = [
        (r"entnahmedatum\s*:\s*(\d{1,2})\.(\d{1,2})\.(\d{4})", "histology report"),
        (r"(?:ultraschall\w*|biopsie|stanz\w*)[^\n]{0,80}?vom\s*"
         r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
         "discharge letter"),
    ]
    biopsy_seen = False
    for doc in data.documents.values():
        if re.search(r"ultraschallgezielte biopsie|stanzbiopsie", doc.text, re.IGNORECASE):
            biopsy_seen = True
        for pat, src in pats:
            for m in re.finditer(pat, doc.text, re.IGNORECASE):
                claims.append((f"{m.group(1)}.{m.group(2)}.{m.group(3)}",
                               doc.span(m.start(), m.end()), src))
    if not biopsy_seen or len({c[0] for c in claims}) < 2:
        return []
    desc = " vs ".join(f"{c[0]} ({c[2]})" for c in claims)
    return _contra(
        "event.biopsy_date",
        "Conflicting dates for the same biopsy",
        "The breast core biopsy is dated differently in the documents: " + desc + ".",
        "date",
        [c[1] for c in claims])


_NEGATION_PAT = re.compile(r"(keine|kein|ohne|nicht)\s+([a-zäöüß]+)", re.IGNORECASE)


def check_negation_conflicts(data: PatientData) -> list[Contradiction]:
    """A finding affirmed in one document and negated in another.

    Matches positive/negative statements about the same finding class
    (e.g. enlarged lymph nodes). Only same-period documents are compared, so
    genuine later resolution does not fire.
    """
    pos: list[EvidenceSpan] = []
    neg: list[EvidenceSpan] = []
    for doc in data.documents.values():
        pos += _spans(doc, r"vergrößerte lymphknoten[^.\n]*")
        neg += _spans(doc, r"keine vergrößerten lymphknoten[^.\n]*")
    if not (pos and neg):
        return []
    return _contra(
        "finding.lymph_nodes",
        "Axillary lymph nodes: enlarged on palpation vs none on ultrasound",
        "One document records enlarged axillary lymph nodes on palpation while another "
        "records no enlarged axillary lymph nodes on ultrasound — the two findings "
        "cannot both hold at the same time.",
        "negation",
        pos + neg)


def detect_contradictions(data: PatientData) -> list[Contradiction]:
    out: list[Contradiction] = []
    out += check_laterality(data)
    out += check_bmi(data)
    out += check_conflicting_dates(data)
    out += check_case_ids(data)
    out += check_event_dates(data)
    out += check_negation_conflicts(data)
    return out
