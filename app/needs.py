"""Situation -> information needs -> screen sections.

The mapping from a clinical role/task to *which kinds of facts* it needs is
explicitly allowed to be hard-coded by the task rules. Finding the facts
themselves, their passages and the contradictions is done by the pipeline.

Each need becomes a section on the screen. Sections are one of:
  facts     - list every matching fact (deduped)
  latest    - only the most recently dated fact(s) per sub-key
  timeline  - facts sorted by date
  checklist - plan/recommendation items with a documented/not-documented status
  computed  - assembled by a dedicated builder (contrast safety, weight trend…)
  search    - run a keyword search over the documents; 'not documented' if empty
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import (MONTH_DE, REFERENCE_DATE, Fact, _dates_in, _event_dates,
                      en_date, en_month, fact_item)
from .lexicon import ROLES, TASK_NEEDS
from .pipeline import Document, PatientData
from .schemas import EvidenceSpan


@dataclass
class Section:
    key: str
    title: str
    kind: str                       # facts | latest | timeline | checklist | computed | search
    categories: tuple = ()
    search_terms: tuple = ()
    not_documented: str = ""        # label used when the section is empty
    builder: str = ""               # name of computed builder
    exclude: tuple = ()             # category prefixes to drop
    kw_filter: tuple = ()           # keep only facts whose kw intersects this set
    search_labels: tuple = ()       # (regex, english label) applied to raw search hits


# ------------------------------------------------------------------ need list

NEED_DEFS: dict[str, Section] = {
    "allergy": Section("allergy", "Allergies & warnings", "facts",
                       ("allergy",), not_documented="No allergy documented"),
    "meds.current": Section("meds.current", "Current medication (latest list)",
                            "facts", ("medication.",), exclude=("medication.nitrate",)),
    "meds.antithrombotic": Section("meds.antithrombotic", "Antithrombotic medication",
                                   "facts", ("medication.antithrombotic",),
                                   not_documented="No antithrombotic medication documented"),
    "meds.antidiabetic": Section("meds.antidiabetic", "Glucose-lowering medication",
                                 "facts", ("medication.antidiabetic",),
                                 not_documented="No glucose-lowering medication documented"),
    "meds.antihypertensive": Section("meds.antihypertensive", "Blood-pressure medication",
                                     "facts", ("medication.antihypertensive",),
                                     not_documented="No antihypertensive medication documented"),
    "contrast": Section("contrast", "Before iodinated contrast", "computed",
                        builder="contrast"),
    "cardiac": Section("cardiac", "Cardiac function", "timeline",
                       ("measurement.ef", "measurement.cardiac", "procedure.cardiac",
                        "measurement.heart_rate")),
    "vitals": Section("vitals", "Vitals & measurements (latest each)", "latest",
                      ("measurement.",),
                      # cardiac, vascular and lab values have their own sections
                      exclude=("measurement.ef", "measurement.cardiac",
                               "measurement.vascular", "measurement.hba1c")),
    "weight": Section("weight", "Weight course", "computed", builder="weight"),
    "hba1c": Section("hba1c", "HbA1c", "computed", builder="hba1c"),
    "labs": Section("labs", "Laboratory results", "computed", builder="labs"),
    "dx.active": Section(
        "dx.active", "Diagnoses", "facts", ("diagnosis.",),
        exclude=("diagnosis.oncology",)),
    "dx.oncology": Section("dx.oncology", "Tumour diagnosis & histology", "facts",
                           ("diagnosis.oncology", "finding.breast")),
    "dx.vascular": Section("dx.vascular", "Carotid stenosis", "facts",
                           ("diagnosis.vascular",)),
    "history": Section("history", "Previous episode", "facts", ("history.event",)),
    "procedures": Section("procedures", "Examinations & procedures", "timeline",
                          ("procedure.",)),
    "imaging.breast": Section("imaging.breast", "Breast imaging & biopsy", "timeline",
                              ("procedure.imaging", "procedure.biopsy"),
                              kw_filter=("breast", "mamma", "birads", "axilla",
                                         "biopsy", "stanzbiopsie", "mammography")),
    "followup": Section("followup", "Recommended follow-ups", "computed",
                        builder="followup"),
    "plan.treatment": Section("plan.treatment", "Treatment plan", "facts",
                              ("plan.treatment", "plan.surgery", "plan.done")),
    "admin.stays": Section("admin.stays", "Hospital stays & documents", "computed",
                           builder="stays"),
    "renal": Section("renal", "Renal function", "search",
                     search_terms=("kreatinin", "creatin", "gfr", "niere", "renal",
                                   "nierenfunktion", "harnstoff"),
                     not_documented="Renal function (creatinine/eGFR): not documented"),
    "anesthesia": Section("anesthesia", "Anaesthesia history", "search",
                          search_terms=("anästhes", "narkose", "intubation", "anesthesia"),
                          not_documented="Previous anaesthesia: not documented"),
    "airway": Section("airway", "Airway / aspiration risk notes", "search",
                      search_terms=("atemweg", "airway", "intubation", "aspiration",
                                    "schlafapnoe", "reflux", "hernie", "gerd"),
                      search_labels=(
                          (r"schlafapnoe", "Sleep apnoea"),
                          (r"reflux|gerd", "Gastro-oesophageal reflux disease (GERD), grade II"),
                          (r"hiatus", "Hiatus hernia"),
                          (r"atemweg|airway|intubation", "Airway assessment"),
                          (r"aspiration", "Aspiration"),
                      ),
                      not_documented="No airway assessment documented"),
    "lifestyle": Section("lifestyle", "Lifestyle (smoking, alcohol)", "search",
                         search_terms=("rauch", "nikotin", "smok", "alkohol", "alcohol",
                                       "tabak", "zigarett"),
                         not_documented="Smoking/alcohol: not documented"),
    "infection": Section("infection", "Infections & antibiotic therapy", "search",
                         search_terms=("infekt", "antibiot", "sepsis", "fieber",
                                       "entzünd"),
                         not_documented="No infection or antibiotic therapy documented"),
    "pain": Section("pain", "Pain", "search",
                    search_terms=("schmerz", "pain", "analges"),
                    not_documented="Pain assessment: not documented"),
    "pregnancy": Section("pregnancy", "Pregnancy", "search",
                         search_terms=("schwanger", "pregnan", "gravid"),
                         not_documented="Pregnancy status: not documented"),
    "functional": Section("functional", "Functional status", "search",
                          search_terms=("mmse", "mobil", "sturz", "frailty",
                                        "adl", "selbständig"),
                          not_documented="Functional assessment: not documented"),
    "anticoag": Section("anticoag", "Oral anticoagulation (Marcumar/DOAK/INR)", "search",
                        search_terms=("marcumar", "phenprocoumon", "doak", "noak",
                                      "warfarin", "rivaroxaban", "apixaban", "inr", "quick"),
                        not_documented="No oral anticoagulation documented"),
    "blood": Section("blood", "Blood group / antibodies", "search",
                     search_terms=("blutgrupp", "blood group", "antikörpersuchtest",
                                   "antibody"),
                     not_documented="Blood group: not documented"),
}


# --------------------------------------------------------------- role matrix

BASELINE = ["allergy", "meds.current", "dx.active", "vitals"]

ROLE_NEEDS: dict[str, list[str]] = {
    "anaesthetist": ["allergy", "meds.antithrombotic", "cardiac", "meds.current",
                     "airway", "dx.active", "vitals", "labs", "renal",
                     "anesthesia", "lifestyle", "anticoag"],
    "surgeon": ["dx.oncology", "imaging.breast", "plan.treatment", "meds.antithrombotic",
                "allergy", "dx.active", "meds.current", "vitals", "anticoag", "renal"],
    "gp": ["dx.active", "meds.current", "allergy", "vitals", "hba1c", "followup"],
    "emergency": ["history", "dx.vascular", "allergy", "contrast",
                  "meds.antihypertensive", "meds.current", "vitals", "cardiac",
                  "anticoag"],
    "oncologist": ["dx.oncology", "imaging.breast", "plan.treatment", "procedures",
                   "dx.active", "meds.current", "allergy"],
    "cardiologist": ["cardiac", "dx.vascular", "history", "meds.antihypertensive",
                     "meds.current", "vitals", "hba1c", "dx.active", "allergy"],
    "radiologist": ["procedures", "imaging.breast", "dx.oncology", "dx.vascular",
                    "contrast", "allergy"],
    "diabetologist": ["meds.antidiabetic", "hba1c", "weight", "meds.current",
                      "dx.active", "vitals", "renal", "followup"],
    "nurse": ["vitals", "meds.current", "allergy", "dx.active", "followup"],
    "pharmacist": ["meds.current", "allergy", "renal", "hba1c", "anticoag"],
    "ophthalmologist": ["history", "dx.vascular", "followup", "hba1c",
                        "meds.antihypertensive", "meds.antidiabetic", "allergy"],
    "neurologist": ["history", "dx.vascular", "cardiac", "vitals", "meds.current",
                    "contrast", "allergy"],
    "pathologist": ["dx.oncology", "imaging.breast", "procedures"],
    "internist": BASELINE + ["hba1c", "cardiac", "followup"],
    "gynaecologist": ["dx.oncology", "imaging.breast", "dx.active", "meds.current",
                      "allergy", "vitals"],
    "urologist": ["renal", "dx.active", "meds.current", "allergy", "vitals"],
    "orthopaedist": ["dx.active", "meds.current", "allergy", "vitals", "functional"],
    "physiotherapist": ["functional", "dx.active", "procedures", "plan.treatment",
                        "meds.current", "allergy"],
    "dietitian": ["weight", "hba1c", "meds.antidiabetic", "dx.active", "vitals",
                  "meds.current"],
    "psychiatrist": ["history", "dx.active", "meds.current", "allergy",
                     "functional", "followup"],
    "dentist": ["allergy", "meds.current", "dx.active", "procedures",
                "plan.treatment"],
    "clinician": BASELINE + ["followup"],
}

TASK_EXTRA = {
    "anaesthesia": ["allergy", "meds.antithrombotic", "cardiac", "airway", "renal",
                    "anesthesia", "anticoag"],
    "surgery_planning": ["dx.oncology", "imaging.breast", "plan.treatment",
                         "meds.antithrombotic", "anticoag"],
    "surgery_date": ["plan.treatment"],
    "admission": ["allergy", "meds.current", "vitals", "plan.treatment"],
    "diabetes": ["meds.antidiabetic", "hba1c", "weight"],
    "contrast": ["contrast", "allergy", "renal"],
    "neuro_emergency": ["history", "dx.vascular", "cardiac"],
    "followup": ["followup", "procedures", "admin.stays"],
    "antithrombotic": ["meds.antithrombotic", "anticoag"],
    "weight": ["weight", "hba1c"],
    "cardiac": ["cardiac"],
    "renal": ["renal"],
    "allergy": ["allergy"],
    "blood_pressure": ["meds.antihypertensive", "vitals"],
    "infection": ["infection"],
    "pain": ["pain"],
    "pregnancy": ["pregnancy"],
    "lifestyle": ["lifestyle"],
    "functional": ["functional"],
}


# -------------------------------------------------------------- engine logic

def detect_role(text: str) -> str:
    low = text.lower()
    # weight explicit role mentions ("you are the X") higher
    m = re.search(r"(?:you are|as|role)\s+(?:the|an|a|her)\s*([a-z\- ]+?)[\.,\n]", low)
    hay = m.group(1) if m else low
    for pat, role in ROLES:
        if re.search(pat, hay):
            return role
    for pat, role in ROLES:
        if re.search(pat, low):
            return role
    return "clinician"


def select_needs(role: str, text: str) -> list[Section]:
    low = text.lower()
    task_keys: list[str] = []
    generic_keys: list[str] = []
    for task_rule in TASK_NEEDS:
        pat, extra_groups = task_rule[:2]
        allowed_roles = task_rule[2] if len(task_rule) > 2 else None
        if allowed_roles and role not in allowed_roles:
            continue
        if re.search(pat, low):
            for g in extra_groups:
                target = generic_keys if g in GENERIC_TASK_GROUPS else task_keys
                for k in TASK_EXTRA[g]:
                    if k not in target:
                        target.append(k)
    role_keys = ROLE_NEEDS.get(role, ROLE_NEEDS["clinician"])

    def role_rank(key: str) -> tuple[int, int]:
        in_role = key in role_keys
        return (0 if in_role else 1,
                role_keys.index(key) if in_role else task_keys.index(key))

    # Safety first for every role, then the sections the task specifically
    # asks for (in the role's usual order), then the rest of the role's
    # sections, then sections only implied by generic task words.
    keys = list(SAFETY_FIRST)
    for key in sorted(task_keys, key=role_rank) + role_keys + generic_keys:
        if key not in keys:
            keys.append(key)
    return [NEED_DEFS[k] for k in keys]


SAFETY_FIRST = ("allergy",)
# Task groups triggered by words that say *when* rather than *what*
# ("before surgery", "on admission"); they add sections but never reorder.
GENERIC_TASK_GROUPS = {"surgery_date", "admission"}


# -------------------------------------------------------------- assembling

def _facts_of(facts: list[Fact], prefixes: tuple, exclude: tuple = (),
              kw_filter: tuple = ()) -> list[Fact]:
    out = []
    for f in facts:
        if not any(f.category.startswith(p) for p in prefixes):
            continue
        if any(f.category.startswith(p) for p in exclude):
            continue
        if kw_filter and not (f.kw & set(kw_filter)):
            continue
        out.append(f)
    return out


def _latest_per_key(facts: list[Fact]) -> list[Fact]:
    """One fact per category: the one with the most recent date."""
    best: dict[str, Fact] = {}
    for f in facts:
        k = f.category
        if k not in best or (f.date or "") > (best[k].date or ""):
            best[k] = f
    return list(best.values())


def _search_docs(data: PatientData, terms: tuple) -> list[EvidenceSpan]:
    spans = []
    for doc in data.documents.values():
        for term in terms:
            for m in re.finditer(re.escape(term), doc.text, re.IGNORECASE):
                start, end = doc.sentence_bounds(m.start(), m.end())
                spans.append(doc.span(start, end))
    return spans


def _srcs(docs: dict[str, Document], spans: list[EvidenceSpan]) -> list[dict]:
    return [
        {"doc": s.doc_id, "start": s.start, "end": s.end,
         "docTitle": docs[s.doc_id].title, "docDate": docs[s.doc_id].date,
         "docHash": s.doc_hash, "quote": docs[s.doc_id].quote(s)}
        for s in spans
    ]


# --------------------------------------------------------- computed sections

def _b_contrast(data, facts, docs):
    items = []
    for f in _facts_of(facts, ("allergy",)):
        items.append({**fact_item(f, docs), "derived": False})
    for f in facts:
        if f.key.startswith("med.glucophage") or f.key.startswith("med.metformin"):
            items.append({**fact_item(f, docs),
                          "text": f.text + " — on board, check protocol around iodinated contrast",
                          "derived": True})
    for f in _facts_of(facts, ("procedure.imaging",)):
        if "ct" in f.kw or "schädel" in f.kw:
            items.append({**fact_item(f, docs),
                          "text": f.text + " — done without contrast because of the allergy",
                          "derived": True})
    renal = _search_docs(data, NEED_DEFS["renal"].search_terms)
    if renal:
        items.append({"text": "Renal function documented", "date": None,
                      "derived": True, "not_documented": False,
                      "sources": _srcs(docs, renal[:3])})
    else:
        items.append({"text": "Renal function (creatinine/eGFR): not documented",
                      "date": None, "derived": False, "not_documented": True, "sources": []})
    return items


def _b_weight(data, facts, docs):
    ws = sorted(_facts_of(facts, ("measurement.weight",)), key=lambda f: f.date or "")
    if not ws:
        return []
    vals = []
    spans = []
    for f in ws:
        m = re.search(r"(\d+(?:[,.]\d+)?)", f.text)
        if m:
            vals.append((f.date, float(m.group(1).replace(",", ".")), f))
            spans += f.evidence[:1]
    parts = [f"{v:g} kg ({d})" for d, v, _ in vals]
    delta = vals[0][1] - vals[-1][1]
    trend = "loss" if delta > 0 else "gain"
    txt = " → ".join(parts) + f" — {trend} of about {abs(delta):g} kg"
    return [{"text": txt, "date": vals[-1][0], "derived": True,
             "formula": "weight_trend", "not_documented": False,
             "sources": _srcs(docs, spans)}]


def _b_hba1c(data, facts, docs):
    hs = sorted(_facts_of(facts, ("measurement.hba1c",)), key=lambda f: f.date or "")
    items = [{**fact_item(f, docs)} for f in hs]
    if hs and (hs[-1].date or "") < REFERENCE_DATE[:7]:
        year, month = (hs[-1].date or "").split("-")[:2]
        last_date = f"{MONTH_DE[month]} {year}"
        items.append({"text": f"HbA1c not documented after {last_date}",
                      "date": None, "derived": True, "not_documented": True, "sources": []})
    if not hs:
        items.append({"text": "HbA1c: not documented", "date": None,
                      "derived": False, "not_documented": True, "sources": []})
    return items


def _b_labs(data, facts, docs):
    items = _b_hba1c(data, facts, docs)
    items.append({"text": "No additional laboratory values documented",
                  "date": None, "derived": False, "not_documented": True, "sources": []})
    return items


def _b_followup(data, facts, docs):
    items = []
    fu = _facts_of(facts, ("plan.followup",))
    done = _search_docs(data, ("duplex", "doppler"))
    for f in fu:
        d = {**fact_item(f, docs)}
        if "duplex" in f.key or "carotid" in f.key:
            source_docs = {span.doc_id for span in f.evidence}
            source_dates = [docs[doc_id].date for doc_id in source_docs]
            cutoff = max(source_dates, default=f.date or "")
            later = [
                span for span in done
                if docs[span.doc_id].date > cutoff
                and span.doc_id not in source_docs
            ]
            if later:
                d["text"] += " — performed (recorded)"
            else:
                d["text"] += " — no record that it took place"
                d["not_documented"] = True
        items.append(d)
    if fu and not any(
            not item.get("not_documented") for item in items
            if "duplex" in item["text"].casefold()
            or "carotid" in item["text"].casefold()):
        items.append({
            "key": "plan.followup.carotid_duplex",
            "text": "Follow-up carotid duplex ultrasound: not documented as performed",
            "date": None, "derived": True, "not_documented": True, "sources": []})
    return dedupe_missing_items(items)


def missing_item_key(item: dict) -> str:
    key = item.get("key") or item.get("missing_key")
    if key:
        parts = str(key).split(".")
        if parts[:2] == ["plan", "followup"] and len(parts) >= 3:
            return ".".join(parts[:3])
        return str(key)
    return re.sub(r"\s+", " ", item.get("text", "").casefold()).strip()


def dedupe_missing_items(items: list[dict]) -> list[dict]:
    """Keep the most informative missing item for each underlying fact key."""
    output = []
    positions = {}
    for item in items:
        if not item.get("not_documented"):
            output.append(item)
            continue
        key = missing_item_key(item)
        if key not in positions:
            positions[key] = len(output)
            output.append(item)
            continue
        previous_index = positions[key]
        previous = output[previous_index]
        detail = lambda candidate: (
            len(candidate.get("text", "")), len(candidate.get("sources", [])))
        if detail(item) > detail(previous):
            output[previous_index] = item
    return output


def _b_stays(data, facts, docs):
    from .screen import doc_type_label

    items = []
    for doc in sorted(docs.values(), key=lambda item: item.date):
        admissions = _event_dates(doc, "admission")
        discharges = _event_dates(doc, "discharge")
        event_spans = [span for _, span in admissions + discharges]
        visit_dates = []
        if "ambulant" in doc.title.casefold():
            for index, line in enumerate(doc.lines):
                visit = re.search(
                    r"ambulanten besuch|ambulant\w*[^.\n]*behandlung",
                    line, re.IGNORECASE)
                if not visit:
                    continue
                line_dates = _dates_in(line)
                if line_dates:
                    offset = doc.line_offsets[index]
                    span = doc.sentence_span(offset, offset + len(line))
                    visit_dates.extend(
                        value for start, _, value in line_dates
                        if not re.search(
                            r"\bgeb\.?\s*(?:am\s*)?$",
                            line[max(0, start - 18):start], re.IGNORECASE)
                    )
                    event_spans.append(span)
        if not event_spans:
            continue

        details = []
        if admissions:
            dates = sorted({en_date(value) for value, _ in admissions})
            details.append(f"admission date{'s' if len(dates) != 1 else ''}: "
                           + " / ".join(dates))
        if discharges:
            dates = sorted({en_date(value) for value, _ in discharges})
            details.append(f"discharge date{'s' if len(dates) != 1 else ''}: "
                           + " / ".join(dates))
        if visit_dates:
            dates = sorted({en_date(value) for value in visit_dates})
            details.append(f"visit: {' / '.join(dates)}")

        title = doc_type_label(doc.title)
        text = f"{title} · {doc.date}"
        if details:
            text += " — " + "; ".join(details)
        unique_spans = []
        seen = set()
        for span in event_spans:
            marker = (span.doc_id, span.start, span.end)
            if marker not in seen:
                unique_spans.append(span)
                seen.add(marker)
        items.append({
            "text": text, "date": doc.date, "derived": True,
            "not_documented": False,
            "original": " / ".join(doc.quote(span) for span in unique_spans),
            "sources": _srcs(docs, unique_spans),
        })
    return items


BUILDERS = {
    "contrast": _b_contrast,
    "weight": _b_weight,
    "hba1c": _b_hba1c,
    "labs": _b_labs,
    "followup": _b_followup,
    "stays": _b_stays,
}


def _normalised_item_text(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _merge_item_sources(first: dict, duplicate: dict) -> None:
    sources = first.setdefault("sources", [])
    seen = {(s.get("doc", s.get("doc_id")), s.get("start"), s.get("end"))
            for s in sources}
    for source in duplicate.get("sources", []):
        key = (source.get("doc", source.get("doc_id")),
               source.get("start"), source.get("end"))
        if key not in seen:
            sources.append(source)
            seen.add(key)
    if not first.get("original") and duplicate.get("original"):
        first["original"] = duplicate["original"]
    elif duplicate.get("original") and duplicate["original"] not in first.get("original", ""):
        first["original"] = f'{first["original"]} / {duplicate["original"]}'


def _screen_item_key(item: dict) -> str:
    text = item["text"]
    low = text.casefold()
    category = item.get("category", "")
    if category == "history.event":
        return (f"history:{_normalised_item_text(text)}:"
                f"{item.get('event_date') or item.get('date') or ''}")
    if category == "measurement.ef" or re.search(r"ejection fraction|\bef\b", low):
        value = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
        if value:
            return f"ef:{value.group(1).replace(',', '.')}"
    if "carotid" in low:
        value = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
        side = ("left" if re.search(r"\bleft\b|\blinks\b|\bli\.", low)
                else "right" if re.search(r"\bright\b|\brechts\b|\bre\.", low)
                else "?")
        if value:
            return f"carotid:{value.group(1).replace(',', '.')}:{side}"
    return _normalised_item_text(text)


def _dedupe_screen_items(sections: list[dict]) -> set[str]:
    first: dict[str, dict] = {}
    shown = set()
    for section in sections:
        for bucket in ("items", "more"):
            unique = []
            for item in section.get(bucket, []):
                key = _screen_item_key(item)
                shown.add(key)
                shown.add(_normalised_item_text(item["text"]))
                if key in first:
                    _merge_item_sources(first[key], item)
                else:
                    first[key] = item
                    unique.append(item)
            if bucket in section or unique:
                section[bucket] = unique
    return shown


def _situation_score(item: dict, tokens: set[str]) -> int:
    text_tokens = set(re.findall(r"[a-zäöüß]{3,}", item["text"].casefold()))
    score = sum(1 for token in tokens if token in text_tokens)
    surgical_context = {"surgeon", "surgery", "operation", "quadrantectomy",
                        "tumour", "tumor", "oncologist"}
    oncology_markers = ("histology", "carcinoma", "tnm", "stage", "receptor",
                        "allred", "her2", "ki-67", "proliferation")
    if tokens.intersection(surgical_context) and any(
            marker in item["text"].casefold() for marker in oncology_markers):
        score += 3
    return score


def _limit_overloaded_sections(sections: list[dict], role: str,
                               situation: str) -> None:
    tokens = {token for token in re.findall(r"[a-zäöüß]{3,}", situation.casefold())
              if token not in _STOP}
    for section in sections:
        items = section["items"]
        if section["key"] == "dx.active":
            historical = [item for item in items if item.get("status") == "historical"]
            current = [item for item in items if item.get("status") != "historical"]
            general_role = role in {"gp", "internist", "clinician"}

            def current_rank(pair):
                index, item = pair
                text = item["text"].casefold()
                chronic = any(marker in text for marker in (
                    "diabetes", "hypertension", "carotid", "hyperlip",
                    "hyperchol", "lupus", "sle", "reflux", "osteoporosis",
                    "vitamin-d deficiency", "sleep apnoea"))
                active = item.get("status") in {"positive", "planned"}
                if general_role:
                    vascular = any(marker in text for marker in (
                        "carotid", "stenosis", "vascular"))
                    return (-int(chronic), -int(active),
                            -_situation_score(item, tokens), -int(vascular),
                            index)
                return (-_situation_score(item, tokens), -int(active), index)

            ranked = sorted(enumerate(current), key=current_rank)
            historical_ranked = sorted(
                enumerate(historical),
                key=lambda pair: (-_situation_score(pair[1], tokens), pair[0]))
            candidates = ranked + historical_ranked if general_role else ranked
            section["items"] = [item for _, item in candidates[:6]]
            section["more"] = ([item for _, item in candidates[6:]]
                               + ([] if general_role else
                                  [item for _, item in historical_ranked]))
        elif (len(items) > 8
              and role not in {"gp", "internist", "clinician"}):
            ranked = sorted(enumerate(items),
                            key=lambda pair: (
                                0 if section["key"] == "cardiac"
                                and role in {"anaesthetist", "cardiologist"}
                                and pair[1].get("category") == "measurement.ef" else 1,
                                -_situation_score(pair[1], tokens), pair[0]))
            section["items"] = [item for _, item in ranked[:6]]
            section["more"] = [item for _, item in ranked[6:]]
        else:
            section.setdefault("more", [])
    _keep_conflicting_values_visible(sections)


GENERAL_ROLES = {"gp", "internist", "clinician"}
# Sections whose purpose is to list examinations, so bare exam names stay.
_EXAM_LIST_SECTIONS = {"procedures", "imaging.breast"}
_DISPLAY_DATE = re.compile(r"\b\d{1,2} [A-Z][a-z]{2} \d{4}\b")


def _bare_exam(item: dict) -> bool:
    """An examination named without any finding ('Long-term ECG')."""
    return (item.get("category", "").startswith("procedure.")
            and "," not in item["text"] and not re.search(r"\d", item["text"]))


def _measure_kind(item: dict) -> str | None:
    """'heart rate' for 'Heart rate 83/min'; None if not a measured value."""
    if not item.get("category", "").startswith("measurement."):
        return None
    text = _DISPLAY_DATE.sub("", item["text"])
    match = re.search(r"\d", text)
    if not match:
        return None
    kind = _normalised_item_text(text[:match.start()])
    return kind or None


def _focus_sections(sections: list[dict], role: str) -> None:
    """Show what this role needs, once, without a hidden 'more' list.

    * bare examination names are dropped where findings are listed;
    * a repeated measurement keeps only its most recent value (different
      values on that same date are all kept — they may be a contradiction);
    * the diagnosis list keeps only its most relevant entries for specialist
      roles; general roles, and every other section, show everything.
    """
    for section in sections:
        items = section.get("items", [])
        # Only the long diagnosis list is cut to the most relevant entries
        # for specialist roles; every other section is shown in full.
        if section["key"] != "dx.active" or role in GENERAL_ROLES:
            items = items + section.get("more", [])
        if section["key"] in _EXAM_LIST_SECTIONS:
            named = [item["text"] for item in items if not _bare_exam(item)]
            items = [item for item in items if not _bare_exam(item)
                     or not any(text.startswith(item["text"] + ",")
                                for text in named)]
        else:
            items = [item for item in items if not _bare_exam(item)] or items
        newest: dict[str, str] = {}
        for item in items:
            kind = _measure_kind(item)
            if kind:
                newest[kind] = max(newest.get(kind, ""), item.get("iso_date") or "")
        section["items"] = [
            item for item in items
            if not _measure_kind(item)
            or (item.get("iso_date") or "") == newest[_measure_kind(item)]]
        section["more"] = []



def _measure_key(text: str) -> tuple[str, str] | None:
    """('marker x rate', '45') for 'Marker X rate 45%'."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    if not match:
        return None
    label = re.sub(r"\b(positive|negative|positiv|negativ)\b|\(", " ",
                   text[:match.start()].casefold())
    label = _normalised_item_text(label)
    return (label, match.group(1).replace(",", ".")) if label else None


def _keep_conflicting_values_visible(sections: list[dict]) -> None:
    """Never hide one value of a measurement behind 'more' while another,
    different value of the same measurement is shown."""
    for section in sections:
        visible = {}
        for item in section.get("items", []):
            key = _measure_key(item["text"])
            if key:
                visible.setdefault(key[0], set()).add(key[1])
        kept = []
        for item in section.get("more", []):
            key = _measure_key(item["text"])
            if key and key[0] in visible and key[1] not in visible[key[0]]:
                section["items"].append(item)
            else:
                kept.append(item)
        if "more" in section:
            section["more"] = kept


# ------------------------------------------------------------- public entry

def _section_items(sec: Section, data: PatientData, facts: list[Fact],
                   docs: dict, used_fact_keys: set | None = None) -> list[dict]:
    """Items of one section, built from the documents (no screen logic)."""
    items = []
    if sec.kind == "computed":
        items = BUILDERS[sec.builder](data, facts, docs)
    elif sec.kind == "search":
        spans = _search_docs(data, sec.search_terms)
        for s in spans[:12]:
            raw = docs[s.doc_id].quote(s)
            label = None
            for pat, lab in sec.search_labels:
                if re.search(pat, raw, re.IGNORECASE):
                    label = lab
                    break
            if label is None:
                label = f"{sec.title} finding"
            item = {"text": label, "date": docs[s.doc_id].date,
                    "derived": False, "not_documented": False,
                    "original": raw,
                    "sources": _srcs(docs, [s])}
            items.append(item)
    else:
        fs = _facts_of(facts, sec.categories, sec.exclude, sec.kw_filter)
        if sec.kind == "latest":
            fs = _latest_per_key(fs)
        elif sec.kind == "timeline":
            fs = sorted(fs, key=lambda f: (f.date or "", f.text), reverse=True)
        else:
            fs = sorted(fs, key=lambda f: (f.category, f.text))
        for f in fs:
            if used_fact_keys is not None:
                used_fact_keys.add(f.key)
            items.append({**fact_item(f, docs), "not_documented": False})
    if not items and sec.not_documented:
        items = [{"text": sec.not_documented, "date": None, "derived": False,
                  "not_documented": True, "sources": []}]
    for it in items:
        it.setdefault("not_documented", False)
        it.setdefault("sources", [])
        it.setdefault("derived", False)
        it.setdefault("date", None)
    return items


def build_screen(situation: str, data: PatientData, facts: list[Fact]) -> dict:
    role = detect_role(situation)
    sections = select_needs(role, situation)
    docs = data.documents

    out_sections = []
    used_fact_keys = set()
    for sec in sections:
        items = _section_items(sec, data, facts, docs, used_fact_keys)
        if items:
            out_sections.append({"key": sec.key, "title": sec.title, "items": items})

    # Merge repeated labels before limiting overload and filling the fallback.
    shown_texts = _dedupe_screen_items(out_sections)
    _limit_overloaded_sections(out_sections, role, situation)
    _focus_sections(out_sections, role)

    # generic relevance fallback: surface any high-scoring facts not shown yet
    extra = relevance_fallback(situation, facts, used_fact_keys, docs, shown_texts)
    if extra:
        out_sections.append({"key": "also", "title": "Also relevant", "items": extra})
        _dedupe_screen_items(out_sections)

    return {"role": role, "sections": out_sections}


_STOP = set("""the a an and or of in on at to for with without she her his he patient
is are was were be been you your we our as by from that this it its their has have had
do does will would can could should may might not no yes before after when which what
who whom how where why any all each per vs than then so such into over under again once
every also only just about between during scheduled plan planning check review""".split())


def relevance_fallback(situation: str, facts: list[Fact], used: set,
                       docs: dict[str, Document],
                       exclude_texts: set[str] | None = None) -> list[dict]:
    exclude_texts = exclude_texts or set()
    tokens = {t for t in re.findall(r"[a-zäöüß]{3,}", situation.lower()) if t not in _STOP}
    scored = []
    for f in facts:
        if f.key in used:
            continue
        if f.category.startswith(("diagnosis.oncology", "finding.breast")):
            continue
        if _normalised_item_text(f.text) in exclude_texts:
            continue
        if _screen_item_key({"text": f.text, "category": f.category}) in exclude_texts:
            continue
        score = 0
        for t in tokens:
            if t in f.kw:
                score += 2
            elif t in f.text.lower():
                score += 1
        if score >= 3:
            scored.append((score, f))
    scored.sort(key=lambda x: -x[0])
    return [{**fact_item(f, docs), "not_documented": False} for _, f in scored[:8]]


# -------------------------------------------------------------- profiles

def load_profiles(path: str = "config/profiles.yaml") -> None:
    """Merge role/task relevance rules from config/profiles.yaml.

    Configured lists replace the built-in defaults entry-wise; anything not
    present in the file keeps the code defaults.
    """
    import os
    if not os.path.exists(path):
        return
    import yaml
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    global ROLES, TASK_NEEDS, TASK_EXTRA, ROLE_NEEDS
    if cfg.get("roles"):
        ROLES = [(r["pattern"], r["role"]) for r in cfg["roles"]]
    if cfg.get("task_needs"):
        TASK_NEEDS = [(t["pattern"], t["groups"], t.get("roles"))
                      for t in cfg["task_needs"]]
    if cfg.get("task_extra"):
        TASK_EXTRA.update({k: list(v) for k, v in cfg["task_extra"].items()})
    if cfg.get("role_needs"):
        ROLE_NEEDS.update({k: list(v) for k, v in cfg["role_needs"].items()})


# ------------------------------------------------------- key questions
#
# The three questions a role (optionally narrowed by task words) needs
# answered first. These are templates for *kinds* of facts per role/task —
# the answers are always looked up in the documents. Entries are tried in
# order; the first whose roles and task pattern match wins.
#
# question: shown above the answer
# sections: NEED_DEFS keys whose items may answer it
# match:    optional regex an item's text must contain
# latest:   keep only the most recent value of each measurement

KEY_QUESTIONS: list[dict] = [
    {"roles": ["anaesthetist"], "questions": [
        {"question": "Which antithrombotic drug (blood thinner) does she take "
                     "and at what dose?", "sections": ["meds.antithrombotic"]},
        {"question": "Which allergy is documented?", "sections": ["allergy"]},
        {"question": "What was the last documented left-ventricular ejection "
                     "fraction?", "sections": ["cardiac"],
         "match": r"ejection fraction", "latest": True},
    ]},
    {"roles": ["surgeon", "oncologist", "gynaecologist", "pathologist"],
     "questions": [
        {"question": "What is the histology (tissue type) and the TNM stage "
                     "(tumour size, lymph nodes, spread)?",
         "sections": ["dx.oncology"], "match": r"histolog|carcinoma|\bTNM\b"},
        {"question": "What is the receptor status and the Ki-67 rate?",
         "sections": ["dx.oncology"], "match": r"receptor|HER2|Ki-67"},
        {"question": "What did the tumour board recommend and when is it "
                     "scheduled?", "sections": ["plan.treatment"],
         "match": r"tumour board discussion documented|quadrantectomy|"
                  r"mastectomy|resection|surgery"},
    ]},
    {"roles": ["gp", "internist", "diabetologist", "dietitian", "clinician"],
     "task": r"diabet|glucose|blood sugar|hba1c",
     "questions": [
        {"question": "Which glucose-lowering drugs does she take and at what "
                     "doses?", "sections": ["meds.antidiabetic"]},
        {"question": "What is her latest HbA1c and from when?",
         "sections": ["hba1c"]},
        {"question": "How did her weight change?", "sections": ["weight"]},
    ]},
    {"roles": ["emergency", "neurologist"], "questions": [
        {"question": "Has this happened before and what was found to be the "
                     "cause?", "sections": ["history"]},
        {"question": "Is anything documented that matters before giving "
                     "iodinated contrast?", "sections": ["contrast"]},
        {"question": "Which antihypertensive (blood pressure drug) does she "
                     "take and at what dose?",
         "sections": ["meds.antihypertensive"]},
    ]},
    {"roles": ["gp", "internist", "clinician", "ophthalmologist"],
     "task": r"follow-?up|nachsorge|kontroll|open item|discharge",
     "questions": [
        {"question": "Which follow-ups were recommended at discharge?",
         "sections": ["followup"]},
        {"question": "Is there a record that the follow-up carotid "
                     "ultrasound took place?", "sections": ["followup"],
         "match": r"carotid|duplex"},
        {"question": "On which side is the carotid stenosis?",
         "sections": ["dx.vascular"], "match": r"carotid"},
    ]},
    {"roles": ["pharmacist"], "questions": [
        {"question": "Which medicines does she take and at what doses?",
         "sections": ["meds.current"]},
        {"question": "Which allergy is documented?", "sections": ["allergy"]},
        {"question": "Is her renal function documented?", "sections": ["renal"]},
    ]},
    {"roles": ["cardiologist"], "questions": [
        {"question": "What was the last documented left-ventricular ejection "
                     "fraction?", "sections": ["cardiac"],
         "match": r"ejection fraction", "latest": True},
        {"question": "What vascular findings are documented?",
         "sections": ["dx.vascular"]},
        {"question": "Which antihypertensive does she take and at what dose?",
         "sections": ["meds.antihypertensive"]},
    ]},
    {"roles": ["nurse"], "questions": [
        {"question": "Which allergy is documented?", "sections": ["allergy"]},
        {"question": "Which medicines does she take and at what doses?",
         "sections": ["meds.current"]},
        {"question": "Which procedure is planned and when?",
         "sections": ["plan.treatment"], "match": r"planned|quadrantectomy"},
    ]},
    # any other role or task
    {"roles": [], "questions": [
        {"question": "Which allergy is documented?", "sections": ["allergy"]},
        {"question": "Which medicines does she take and at what doses?",
         "sections": ["meds.current"]},
        {"question": "Which diagnoses are documented?", "sections": ["dx.active"]},
    ]},
]


def select_key_questions(role: str, situation: str) -> list[dict]:
    low = situation.casefold()
    for entry in KEY_QUESTIONS:
        if entry["roles"] and role not in entry["roles"]:
            continue
        if entry.get("task") and not re.search(entry["task"], low):
            continue
        return entry["questions"]
    return KEY_QUESTIONS[-1]["questions"]


def build_key_questions(situation: str, data: PatientData,
                        facts: list[Fact]) -> list[dict]:
    """Answer the role's three key questions from document facts."""
    role = detect_role(situation)
    docs = data.documents
    answers = []
    for spec in select_key_questions(role, situation):
        items: list[dict] = []
        for key in spec["sections"]:
            for item in _section_items(NEED_DEFS[key], data, facts, docs):
                if spec.get("match") and not item.get("not_documented") and \
                        not re.search(spec["match"], item["text"], re.IGNORECASE):
                    continue
                items.append(item)
        box = [{"key": "answer", "items": items, "more": []}]
        _dedupe_screen_items(box)
        if spec.get("latest"):
            _focus_sections(box, role)
        items = box[0]["items"]
        if not items:
            items = [{"text": "Not documented", "date": None, "derived": False,
                      "not_documented": True, "sources": []}]
        answers.append({"question": spec["question"], "items": items})
    return answers
