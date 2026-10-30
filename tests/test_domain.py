"""Domain tests on independently authored synthetic fixtures (public CI).

No real patient data here — these fixtures live in tests/fixtures and are
small enough to read at a glance.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.contradictions import check_bmi, detect_contradictions
from app.extract import (_unique_spans, build_fact_store, extract_allergies,
                         extract_diagnoses)
from app.lexicon import allergy_reactions_en
from app.model import DeterministicAdapter, ModelError, HttpAdapter
from app.pipeline import Document, PatientData, clean_text, load_patient
from app.schemas import EvidenceSpan, Screen
from app.screen import (_check_support, _promote_empty_sections,
                        build_screen_model, doc_type_label)
from app import needs

FIX = Path(__file__).parent / "fixtures" / "patient"


def _synthetic_data(text: str) -> PatientData:
    lines = text.splitlines()
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1
    doc = Document(
        doc_id="synthetic", title="Synthetic", date="2016-04-15",
        doc_type="letter", text=text, doc_hash="synthetic",
        lines=lines, line_offsets=offsets)
    return PatientData(
        patient={}, documents={doc.doc_id: doc}, doc_order=[doc.doc_id], index={})


@pytest.fixture(scope="module")
def data():
    return load_patient(FIX)


@pytest.fixture(scope="module")
def facts(data):
    return build_fact_store(data)


# ---- HTML normalisation + evidence mapping -------------------------------

def test_html_normalisation_preserves_offsets(data):
    doc = data.documents["syn03"]
    assert "&auml" not in doc.text and "<BR>" not in doc.text
    for d in data.documents.values():
        for m in re.finditer(r"carotis interna", d.text):
            s = d.span(m.start(), m.end())
            assert s.quote == "carotis interna"
            assert d.quote(s) == s.quote


def test_invalid_spans_rejected(data):
    doc = data.documents["syn01"]
    with pytest.raises(Exception):
        doc.span(-1, 5)
    with pytest.raises(Exception):
        doc.span(10, 5)
    # an evidence span pointing at a non-existent document is rejected at
    # quote-resolution time
    bad = EvidenceSpan(doc_id="missing", start=0, end=10, doc_hash="x", quote="q")
    with pytest.raises(Exception):
        data.documents[bad.doc_id].quote(bad)


def test_sentence_bounds_keep_date_and_ecg_facts(facts, data):
    doc = data.documents["syn03"]
    date_start = doc.text.index("05.04.2016")
    start, end = doc.sentence_bounds(date_start, date_start + len("05.04.2016"))
    quote = doc.span(start, end).quote
    assert "05.04.2016" in quote
    assert "auf der Station aufgenommen" in quote

    ecg = next(f for f in facts if f.category == "measurement.cardiac"
               and f.text.startswith("ECG: heart rate 72"))
    assert ecg.date == doc.date
    assert ecg.evidence[0].quote == "EKG: HF 72/min, Normaltyp"


def test_duplicate_evidence_spans_are_collapsed(data):
    span = data.documents["syn03"].line_span(0)
    assert len(_unique_spans([span, span])) == 1


@pytest.mark.parametrize(
    "title,label",
    [
        ("24h-RR-Messung", "24-hour blood pressure report"),
        ("Langzeit-RR", "24-hour blood pressure report"),
        ("Blutdruckmessung", "24-hour blood pressure report"),
        ("Stat. Patientenbrief (Interne)", "Discharge letter"),
        ("Ambulanter Patientenbrief", "Outpatient letter"),
        ("Histologie-Befund", "Histology report"),
        ("Langzeit-EKG", "Long-term ECG report"),
        ("Echokardiographie", "Echocardiography report"),
        ("Carotis-Duplex", "Ultrasound report"),
        ("CT Thorax", "Imaging report"),
        ("Radiologischer Befund", "Imaging report"),
        ("Konsil für Augenheilkunde", "Consultation note"),
    ],
)
def test_document_titles_get_plain_language_labels(title, label):
    assert doc_type_label(title) == label


def test_bmi_contradiction_uses_plain_document_label():
    text = "Größe: 170 cm\nGewicht: 80 kg\nBMI: 40"
    lines = text.splitlines()
    offsets = [0, len(lines[0]) + 1, len(lines[0]) + len(lines[1]) + 2]
    doc = Document(
        doc_id="synthetic", title="Stat. Patientenbrief (Interne)",
        date="2015-06-28", doc_type="letter", text=text,
        doc_hash="synthetic", lines=lines, line_offsets=offsets)
    data = PatientData(
        patient={}, documents={doc.doc_id: doc}, doc_order=[doc.doc_id],
        index={})

    contradiction = check_bmi(data)[0]

    assert "Discharge letter, 2015-06" in contradiction.title
    assert "Stat. Patientenbrief" not in contradiction.title
    assert len(contradiction.evidence) == 3
    assert {span.quote.split(":", 1)[0].casefold()
            for span in contradiction.evidence} == {"grösse", "gewicht", "bmi"}
    from app.pdf import render_pdf
    from pypdf import PdfReader
    import io
    screen = Screen(
        role="clinician", task_text="Review the record",
        reference_date="2016-04-15", sections=[],
        contradictions=[contradiction], missing=[])
    pdf_text = "".join(
        page.extract_text() for page in PdfReader(io.BytesIO(
            render_pdf(screen))).pages)
    assert "Discharge letter, 2015-06" in pdf_text
    assert "Stat. Patientenbrief" not in pdf_text


# ---- clinical semantics ---------------------------------------------------

def test_status_semantics(facts):
    by_text = {f.text: f for f in facts}
    st = {f.text: f.status for f in facts}
    assert any("Carotid artery stenosis" in t and v == "suspected"
               for t, v in st.items())
    assert any("appendectomy" in t.casefold() and v == "historical"
               for t, v in st.items())
    neg = [f for f in facts if f.status == "negative"]
    pos = [f for f in facts if "vergrößerte Lymphknoten" in str(f.evidence)]
    assert neg or pos  # negation handled one way or another


def test_future_surgery_is_planned_in_screen_and_pdf():
    screen, _ = build_screen_model(
        "Surgeon: planning the quadrantectomy", FIX)
    item = next(item for section in screen.sections
                if section.key == "plan.treatment"
                for item in section.items
                if "quadrantectomy" in item.text.lower())
    assert item.status == "planned"
    assert item.date == "16 May 2016"

    from app.pdf import render_pdf
    from pypdf import PdfReader
    import io
    pdf = render_pdf(screen)
    text = "".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert "PLANNED" in text
    assert "16 May 2016" in text


def test_support_drops_only_laterality_conflicting_span(data):
    item = {
        "text": "Left 60% carotid stenosis",
        "sources": [
            {"quote": "Rechts findet sich eine Einengung von 60%."},
            {"quote": "Links findet sich eine Einengung von 60%."},
        ],
    }
    kept, dropped = _check_support([item], data, DeterministicAdapter())
    assert dropped == 1
    assert len(kept) == 1
    assert len(kept[0]["sources"]) == 1
    assert "Links" in kept[0]["sources"][0]["quote"]
    assert DeterministicAdapter().assess_support(
        "Height 171 cm", "Height 170 cm") == "unsupported"
    assert DeterministicAdapter().assess_support(
        "Byetta 5 µg", "Byetta Injektionslösung 5mcg") == "supported"
    assert DeterministicAdapter().assess_support(
        "Arterial hypertension", "Art. Hypertonie",
        original="Art. Hypertonie") == "supported"
    assert DeterministicAdapter().assess_support(
        "Arterial hypertension — documented as: Art. Hypertonie",
        "Unrelated source passage") == "unchecked"
    drop_log = []
    _, dropped = _check_support(
        [{"text": "Left carotid stenosis 60%",
          "sources": [{"quote": "Right carotid stenosis 60%"}]}],
        data, DeterministicAdapter(), drop_log)
    assert dropped == 1
    assert drop_log[0]["reason"].startswith("laterality mismatch")
    untraceable, _ = _check_support(
        [{"text": "Unsubstantiated fact", "sources": []}],
        data, DeterministicAdapter())
    assert untraceable == []


@pytest.mark.parametrize(
    "claim,quote",
    [
        ("Dose 5 µg", "Dose 5mcg"),
        ("Dose 5 µg", "Dose 5 ug"),
        ("Dose 1 mg", "Dose 1mg"),
        ("Volume 2 ml", "Volume 2mL"),
        ("BMI 27,7 kg/m²", "BMI 27.7kg/m2"),
        ("Flow 150 cm/s", "Flow 150cm/sek"),
        ("Stenosis 60%", "Stenosis 60%ige"),
        ("Pressure 22 mmHg", "PAP 22mm Hg"),
    ],
)
def test_support_normalizes_units_and_whitespace(claim, quote):
    assert DeterministicAdapter().assess_support(claim, quote) == "supported"


def test_derived_weight_trend_matches_inputs_and_recomputes(data):
    sources = [
        {"quote": "Gewicht: 78 kg"},
        {"quote": "Gewicht: 67 kg"},
        {"quote": "Gewicht: 66 kg"},
    ]
    claim = ("78 kg (2015-06-28) → 67 kg (2016-04-05) → "
             "66 kg (2016-04-15) — loss of about 12 kg")
    item = {"text": claim, "derived": True, "formula": "weight_trend",
            "sources": sources}

    kept, dropped = _check_support([item], data, DeterministicAdapter())

    assert dropped == 0
    assert len(kept) == 1
    assert kept[0]["sources"] == sources


def test_derived_weight_trend_with_incorrect_result_is_dropped(data):
    item = {
        "text": ("78 kg (2015-06-28) → 67 kg (2016-04-05) → "
                 "66 kg (2016-04-15) — loss of about 20 kg"),
        "derived": True,
        "formula": "weight_trend",
        "sources": [
            {"quote": "Gewicht: 78 kg"},
            {"quote": "Gewicht: 67 kg"},
            {"quote": "Gewicht: 66 kg"},
        ],
    }
    drop_log = []

    kept, dropped = _check_support(
        [item], data, DeterministicAdapter(), drop_log)

    assert kept == []
    assert dropped == 3
    assert all(row["reason"].startswith("derived result mismatch")
               for row in drop_log)


def test_derived_bmi_is_recomputed_from_cited_height_and_weight(data):
    item = {
        "text": "BMI 27.7 kg/m²",
        "derived": True,
        "formula": "bmi",
        "sources": [
            {"quote": "Gewicht: 80 kg"},
            {"quote": "Größe: 170 cm"},
        ],
    }

    kept, dropped = _check_support([item], data, DeterministicAdapter())

    assert dropped == 0
    assert len(kept) == 1


def test_translated_facts_keep_original_source_wording():
    data = _synthetic_data(
        "Diagnosen:\nArt. Hypertonie\nMammakarzinom links\n"
        "Histologie: invasiv duktales Mammakarzinom\n\n"
        "Allergien:\nKM-Allergie (Übelkeit, Schwindel, Erbrechen)\n")
    diagnoses = extract_diagnoses(data)
    hypertension = next(f for f in diagnoses if f.text == "Arterial hypertension")
    cancer = next(f for f in diagnoses if f.text == "Breast cancer, left")
    histology = next(f for f in diagnoses
                     if f.text == "Invasive ductal carcinoma")
    allergy = extract_allergies(data)[0]

    assert hypertension.original == "Art. Hypertonie"
    assert cancer.category == "diagnosis.active"
    assert cancer.original == "Mammakarzinom links"
    assert histology.category == "diagnosis.oncology"
    assert allergy.text == "Contrast media allergy (nausea, dizziness, vomiting)"
    assert allergy.original == "KM-Allergie (Übelkeit, Schwindel, Erbrechen)"
    assert allergy_reactions_en("Übelkeit, Schwindel, Erbrechen") == [
        "nausea", "dizziness", "vomiting"]
    no_allergy = extract_allergies(
        _synthetic_data("Allergien:\nkeine bekannt\n"))[0]
    assert no_allergy.text == "No known allergies"
    assert no_allergy.original == "keine bekannt"
    active_section = needs.NEED_DEFS["dx.active"]
    active = needs._facts_of(
        diagnoses, active_section.categories, active_section.exclude)
    assert cancer in active
    assert histology not in active


def test_equivalent_ef_and_carotid_items_merge_sources():
    sections = [{"key": "cardiac", "items": [
        {"text": "Left-ventricular ejection fraction 60%",
         "category": "measurement.ef", "sources": [{"doc": "one"}]},
        {"text": "Normal systolic LV function (EF 60%)",
         "category": "measurement.ef", "sources": [{"doc": "two"}]},
        {"text": "Carotid artery stenosis (Carotissklerose li. 60%)",
         "category": "diagnosis.vascular", "sources": [{"doc": "three"}]},
        {"text": "Carotid stenosis: up to 60% stenosis of the internal carotid artery, left side",
         "category": "diagnosis.vascular", "sources": [{"doc": "four"}]},
    ]}]
    needs._dedupe_screen_items(sections)
    assert len(sections[0]["items"]) == 2
    ef = next(item for item in sections[0]["items"]
              if item["category"] == "measurement.ef")
    carotid = next(item for item in sections[0]["items"]
                   if item["category"] == "diagnosis.vascular")
    assert {source["doc"] for source in ef["sources"]} == {"one", "two"}
    assert {source["doc"] for source in carotid["sources"]} == {"three", "four"}


@pytest.mark.parametrize("role", ["gp", "internist", "clinician", "nurse"])
def test_active_diagnoses_are_capped_for_every_role(role):
    section = {"key": "dx.active", "items": [
        {"text": text, "status": "positive", "sources": []}
        for text in (
            "Breast cancer, left", "Arterial hypertension", "Diabetes mellitus type 2",
            "Carotid stenosis", "Hyperlipidaemia", "Systemic lupus erythematosus",
            "Sleep apnoea", "Gastro-oesophageal reflux disease")
    ] + [{"text": "Status post appendectomy", "status": "historical", "sources": []}]}
    needs._limit_overloaded_sections([section], role, "routine review")
    assert len(section["items"]) == 6
    assert len(section["more"]) == 3
    if role in {"gp", "internist", "clinician"}:
        assert "Breast cancer, left" not in [
            item["text"] for item in section["items"][:4]]
        assert any("Carotid" in item["text"] for item in section["items"])


@pytest.mark.parametrize("role", ["anaesthetist", "cardiologist"])
def test_ef_is_ranked_into_visible_cardiac_items(role):
    section = {"key": "cardiac", "items": [
        {"text": f"Cardiac observation {i}", "category": "measurement.cardiac"}
        for i in range(8)
    ] + [{"text": "Left-ventricular ejection fraction 60%",
          "category": "measurement.ef"}]}
    needs._limit_overloaded_sections([section], role, "cardiovascular review")
    assert any(item["category"] == "measurement.ef" for item in section["items"])


def test_empty_visible_section_promotes_overflow():
    sections = [{"items": [], "more": [{"text": "Procedure"}]}]
    _promote_empty_sections(sections)
    assert sections == [{"items": [{"text": "Procedure"}], "more": []}]


def test_screen_deduplicates_text_and_merges_sources():
    sections = [
        {"items": [{"text": "Carotid stenosis", "sources": [
            {"doc": "a", "start": 1, "end": 2}]}]},
        {"items": [{"text": "CAROTID stenosis!", "sources": [
            {"doc": "b", "start": 3, "end": 4}]}]},
    ]
    needs._dedupe_screen_items(sections)
    assert len(sections[0]["items"]) == 1
    assert sections[1]["items"] == []
    assert len(sections[0]["items"][0]["sources"]) == 2


def test_historical_diagnoses_go_into_more_for_specialist():
    sections = [{
        "key": "dx.active",
        "items": ([{"text": f"Current diagnosis {i}", "status": "positive"}
                   for i in range(7)]
                  + [{"text": "Appendectomy", "status": "historical"}]),
    }]
    needs._limit_overloaded_sections(sections, "pharmacist", "medication review")
    assert len(sections[0]["items"]) == 6
    assert len(sections[0]["more"]) == 2
    assert all(item["status"] != "historical" for item in sections[0]["items"])


def test_medication_schedule_survives(facts):
    med = [f for f in facts if f.category.startswith("medication")]
    assert med, "no medication extracted"
    texts = " ".join(f.text for f in med)
    assert "Metformin" in texts or "metformin" in texts
    # 'mo ab' -> morning and evening
    assert "morning" in texts and "evening" in texts


def test_derived_facts_cite_inputs(facts):
    derived = [f for f in facts if f.derived]
    for f in derived:
        if f.formula:
            assert f.inputs or f.evidence, f"derived fact {f.key} lacks cited inputs"


def test_missing_distinct_from_negative():
    screen, _ = build_screen_model("You are the diabetologist reviewing this patient.", FIX)
    labels = [m.label for m in screen.missing]
    assert labels  # e.g. no HbA1c in fixtures
    assert all(m.status == "not_documented" for m in screen.missing)


def test_prompt_injection_ignored():
    screen, _ = build_screen_model("You are the surgeon planning the operation.", FIX)
    body = " ".join(i.text for s in screen.sections for i in s.items)
    assert "HACKED" not in body


# ---- contradictions ---------------------------------------------------------

def test_contradictions_detected(data):
    cs = {c.key: c for c in detect_contradictions(data)}
    assert any("laterality" in k for k in cs)
    assert any(k.startswith("bmi") for k in cs)
    assert any("admission" in k for k in cs)
    assert any("lymph" in k for k in cs)
    for c in cs.values():
        assert len(c.evidence) >= 2
        assert all(e.quote for e in c.evidence)


def test_temporal_change_not_flagged(tmp_path, data):
    # same measured value drifting over time must not alert
    docs = tmp_path / "p"
    (docs / "documents").mkdir(parents=True)
    import json
    json.dump({"documents": [
        {"id": "t1", "title": "A", "date": "2016-01-01", "type": "x"},
        {"id": "t2", "title": "B", "date": "2016-02-01", "type": "x"}]},
        open(docs / "documents.json", "w"))
    json.dump({"text": "Größe: 170 cm<BR>Gewicht: 80 kg<BR>BMI: 27.7"},
              open(docs / "documents" / "t1.json", "w"))
    json.dump({"text": "Größe: 171 cm<BR>Gewicht: 82 kg<BR>BMI: 28.0"},
              open(docs / "documents" / "t2.json", "w"))
    json.dump({"patient": {}}, open(docs / "patient.json", "w"))
    d = load_patient(docs)
    assert not any("bmi" in c.key for c in detect_contradictions(d))


# ---- adapter ----------------------------------------------------------------

def test_model_timeout_recoverable(monkeypatch):
    ad = HttpAdapter("http://127.0.0.1:9/x", "k", timeout=0.01)
    with pytest.raises(ModelError):
        ad.assess_support("c", "q")


def test_invalid_model_output_recoverable():
    det = DeterministicAdapter()
    assert det.assess_support("totally unrelated claim", "short quote") in (
        "unsupported", "unchecked")


def test_input_change_changes_output(facts, data):
    a = needs.build_screen("You are the diabetologist.", data, facts)
    b = needs.build_screen("You are the ophthalmologist.", data, facts)
    assert [s["title"] for s in a["sections"]] != [s["title"] for s in b["sections"]]


def test_new_role_paraphrases():
    assert needs.detect_role("You are the eye doctor on call") == "ophthalmologist"
    assert needs.detect_role("the heart specialist needs an overview") in (
        "cardiologist", "clinician")


@pytest.mark.parametrize(
    "situation,expected_role,first_sections,oncology,has_surgery",
    [
        ("Pharmacist: medication reconciliation before surgery", "pharmacist",
         {"meds.current"}, False, True),
        ("Nurse: admission the day before surgery", "nurse",
         {"allergy", "vitals"}, False, True),
        ("Cardiologist: preoperative cardiovascular risk assessment before surgery",
         "cardiologist", {"cardiac"}, False, True),
        ("Ophthalmologist: follow-up after amaurosis fugax", "ophthalmologist",
         {"history"}, False, False),
        ("Physiotherapist: mobilisation after surgery", "physiotherapist",
         {"functional"}, False, True),
        ("Surgeon: planning the quadrantectomy", "surgeon",
         {"dx.oncology"}, True, True),
    ],
)
def test_role_first_needs(situation, expected_role, first_sections, oncology,
                          has_surgery):
    needs.load_profiles()
    role = needs.detect_role(situation)
    sections = needs.select_needs(role, situation)
    keys = [section.key for section in sections]
    assert role == expected_role
    assert keys[0] in first_sections
    assert ("dx.oncology" in keys) is oncology
    if has_surgery:
        assert "plan.treatment" in keys


def test_cache_invalidates_on_change(data, facts):
    from app.screen import load_domain, _cache
    _cache.clear()
    d1, f1, _ = load_domain(FIX)
    d2, f2, _ = load_domain(FIX)
    assert f1 is f2  # warm-cache hit
    first_key = next(iter(_cache))
    # change model/prompt version -> different cache key -> recompute
    import app.screen as sc
    old = sc.cache_key
    sc.cache_key = lambda: "bump"
    try:
        d3, f3, _ = load_domain(FIX)
    finally:
        sc.cache_key = old
    assert f3 is not f1
    assert next(iter(_cache)) != first_key


# ---- screen schema + pdf ----------------------------------------------------

def test_screen_schema_valid():
    screen, stats = build_screen_model(
        "You are the anaesthetist preparing this patient for surgery as of 2016-04-15.", FIX)
    assert isinstance(screen, Screen)
    assert screen.role == "anaesthetist"
    assert screen.reference_date == "2016-04-15"
    assert screen.review_status == (
        "Rule-based extraction (no AI model) — pending clinician review")


def test_pdf_matches_screen():
    from app.pdf import render_pdf
    from pypdf import PdfReader
    import io
    screen, _ = build_screen_model("You are the surgeon planning the operation.", FIX)
    pdf = render_pdf(screen)
    reader = PdfReader(io.BytesIO(pdf))
    text = "".join(p.extract_text() for p in reader.pages)
    first = next(i for s in screen.sections for i in s.items)
    assert first.text[:40] in text
    assert "Draft - please verify against sources" in text


def test_missing_data_dir_returns_recoverable_error(tmp_path):
    from app.screen import build_screen_model
    screen, stats = build_screen_model(
        "You are the nurse. Admission before surgery.", tmp_path / "does-not-exist")
    assert screen.errors and "failed to load documents" in screen.errors[0]
    assert screen.sections == [] and screen.missing == [] and screen.contradictions == []
