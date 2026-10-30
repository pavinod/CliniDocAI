from pathlib import Path

from app.extract import (build_fact_store, extract_demographics, extract_history,
                         fact_item)
from app.model import DeterministicAdapter
from app.needs import _b_stays, dedupe_missing_items
from app.pipeline import Document, PatientData, clean_text
from app.screen import _check_support


FIXTURES = Path(__file__).parent / "fixtures" / "extraction"


def _fixture_data(name: str, title: str, date: str = "2024-02-03") -> PatientData:
    text = clean_text((FIXTURES / name).read_text(encoding="utf-8"))
    lines = text.splitlines()
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1
    doc = Document(
        doc_id="synthetic", title=title, date=date,
        doc_type="letter", text=text, doc_hash="synthetic",
        lines=lines, line_offsets=offsets)
    return PatientData(
        patient={}, documents={doc.doc_id: doc}, doc_order=[doc.doc_id], index={})


def test_synthetic_histology_values_are_extracted_from_source():
    data = _fixture_data("histology_report.txt", "Histology report")
    facts = build_fact_store(data)
    texts = {fact.text for fact in facts}

    assert "Oestrogen receptor positive (90% of tumour cells, Allred score 6)" in texts
    assert "Progesterone receptor negative" in texts
    assert "Ki-67 proliferation rate 35%" in texts
    assert "TNM stage: pT1c N1a M0 G3" in texts


def test_alternate_birth_dates_produce_different_derived_ages():
    first = extract_demographics(
        _fixture_data("demographics_a.txt", "Discharge letter"))
    second = extract_demographics(
        _fixture_data("demographics_b.txt", "Discharge letter"))
    first_age = next(f for f in first if f.formula == "age"
                     and "on " in f.text)
    second_age = next(f for f in second if f.formula == "age"
                      and "on " in f.text)

    assert first_age.text != second_age.text
    assert first_age.text.startswith("Age 46 years")
    assert second_age.text.startswith("Age 45 years")
    assert any(f.formula == "age" and "at admission" in f.text for f in first)


def test_age_derivation_recomputes_from_birth_date():
    source = {"quote": "geb. am 01.01.1970"}
    correct = {
        "text": "Age 46 years on 2016-04-15 (born 1970-01-01)",
        "derived": True, "formula": "age", "sources": [source],
    }
    wrong = {
        "text": "Age 47 years on 2016-04-15 (born 1970-01-01)",
        "derived": True, "formula": "age", "sources": [source],
    }

    kept, dropped = _check_support(
        [correct], _fixture_data("demographics_a.txt", "Discharge letter"),
        DeterministicAdapter())
    rejected, wrong_dropped = _check_support(
        [wrong], _fixture_data("demographics_a.txt", "Discharge letter"),
        DeterministicAdapter())

    assert dropped == 0 and len(kept) == 1
    assert wrong_dropped == 1 and rejected == []


def test_exam_text_contains_source_size_and_birads_values():
    data = _fixture_data("breast_exam.txt", "Ultrasound report")
    facts = build_fact_store(data)
    exam = next(f for f in facts
                if f.category == "procedure.imaging"
                and "Breast ultrasound" in f.text)

    assert "4 x 2 cm" in exam.text
    assert "BI-RADS IV" in exam.text
    assert exam.original


def test_multiline_episode_and_exam_findings_are_source_derived():
    facts = build_fact_store(
        _fixture_data("multiline_episode.txt", "Discharge letter"))

    episode = next(f for f in facts if f.category == "history.event"
                   and "peak BP" in f.text)
    nitro = next(f for f in facts if f.category == "history.event"
                 and "nitroglycerin spray" in f.text)
    head_ct = next(f for f in facts if f.category == "procedure.imaging"
                   and f.text.startswith("Head CT"))
    ultrasound = next(f for f in facts if f.category == "procedure.imaging"
                      and f.text.startswith("Cerebral artery ultrasound"))

    assert "185/110 mmHg" in episode.text
    assert "\n185/110 mmHg" in episode.original
    assert "2 puffs of nitroglycerin spray" in nitro.text
    assert "Keine Blutung" in head_ct.original
    assert "no haemorrhage" in head_ct.text
    assert head_ct.date == "2015-06-11"
    assert "11 Jun 2015" not in head_ct.text
    assert "150 cm/sek" in ultrasound.original
    assert ultrasound.date == "2015-06-11"
    assert "150 cm" in ultrasound.text


def test_history_uses_date_in_evidence_before_admission_date():
    facts = extract_history(_fixture_data(
        "history_date_explicit.txt", "Stat. Patientenbrief (Interne)"))
    crisis = next(f for f in facts if f.text.startswith("Hypertensive crisis"))

    assert crisis.date == "2024-02-09"


def test_history_keeps_month_precision_from_numeric_month_date():
    facts = extract_history(_fixture_data(
        "history_month_date.txt", "Outpatient letter"))
    crisis = next(f for f in facts if f.text.startswith("Hypertensive crisis"))

    assert crisis.date == "2015-06"


def test_history_uses_discharge_admission_date_as_fallback():
    facts = extract_history(_fixture_data(
        "history_date_admission.txt", "Stat. Patientenbrief (Interne)"))
    crisis = next(f for f in facts if f.text.startswith("Hypertensive crisis"))

    assert crisis.date == "2024-02-11"


def test_history_event_items_merge_details_and_keep_treatment_separate():
    from app.screen import _merge_history_event_items

    data = _fixture_data("history_merge.txt", "Stat. Patientenbrief (Interne)")
    facts = build_fact_store(data)
    section = {
        "items": [
            {**fact_item(fact, data.documents), "not_documented": False}
            for fact in facts if fact.category == "history.event"
        ],
        "more": [],
    }
    _merge_history_event_items([section])

    crises = [item for item in section["items"]
              if item["text"].startswith("Hypertensive crisis")]
    treatments = [item for item in section["items"]
                  if item["text"].startswith("Emergency treatment")]
    assert len(crises) == 1
    assert crises[0]["text"] == (
        "Hypertensive crisis — dizziness, vision symptoms; peak BP 185/110 mmHg")
    assert len(crises[0]["sources"]) == 2
    assert crises[0]["date"] == "11 Feb 2024"
    assert len(treatments) == 1
    assert "2 puffs of nitroglycerin spray" in treatments[0]["text"]


def test_history_diagnosis_list_does_not_span_adjacent_findings():
    facts = build_fact_store(
        _fixture_data("diagnosis_list.txt", "Discharge letter"))
    crisis = next(f for f in facts
                  if f.category == "history.event"
                  and f.text.startswith("Hypertensive crisis"))

    assert crisis.original == "Hypertensive Krise mit Schwindel und Sehstörung"
    assert "55 Prozent" not in crisis.original


def test_missing_followups_dedupe_by_underlying_plan_key():
    items = [
        {"key": "plan.followup.carotid_duplex.synthetic.23",
         "text": "Carotid duplex ultrasound follow-up, 28 Jun 2015, in 2 months "
                 "— no record that it took place",
         "not_documented": True, "sources": [{"doc": "synthetic"}]},
        {"key": "plan.followup.carotid_duplex",
         "text": "Follow-up carotid duplex ultrasound: not documented as performed",
         "not_documented": True, "sources": []},
    ]

    deduped = dedupe_missing_items(items)

    assert len(deduped) == 1
    assert deduped[0]["text"] == (
        "Carotid duplex ultrasound follow-up, 28 Jun 2015, in 2 months "
        "— no record that it took place")
    assert "in 2 months" in deduped[0]["text"]


def test_visit_timeline_excludes_birth_date_from_visit_date():
    data = _fixture_data("outpatient_visit.txt", "Ambulanter Patientenbrief")
    items = _b_stays(data, [], data.documents)

    assert len(items) == 1
    assert "visit: 3 Apr 2024" in items[0]["text"]
    assert "1 Jan 1970" not in items[0]["text"]
