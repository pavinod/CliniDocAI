"""Screen orchestration: situation -> typed Screen model.

Wires the domain pieces together: document store, fact extraction, needs
selection, contradiction detection, claim-support assessment, caching.
Document text is treated strictly as untrusted data — it is never executed
or interpreted as instructions.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from datetime import date as Date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from . import needs
from .contradictions import detect_contradictions
from .extract import _dates_in, build_fact_store
from .labels import doc_type_label
from .model import (DeterministicAdapter, ModelError, get_adapter,
                    support_mismatch_reason)
from .pipeline import Document, PatientData, load_patient
from .schemas import (
    Contradiction, EvidenceSpan, Fact, KeyQuestion, MissingItem, Screen,
    ScreenItem, ScreenSection, sha256_text,
    EXTRACTION_VERSION, MODEL_VERSION, PROMPT_VERSION,
)

REFERENCE_DATE = "2016-04-15"
_data_dir = Path(os.environ.get(
    "DATA_ROOT",
    os.environ.get("CLINIDOC_DATA_DIR",
                   Path(__file__).resolve().parent.parent / "data" / "patient-0000-0001")))

_cache: dict[str, tuple[PatientData, list[Fact], list[Contradiction], float]] = {}


_WEIGHT_VALUE = re.compile(r"(\d+(?:[.,]\d+)?)\s*kg\b", re.IGNORECASE)
_HEIGHT_VALUE = re.compile(r"(\d+(?:[.,]\d+)?)\s*cm\b", re.IGNORECASE)
_WEIGHT_TREND_RESULT = re.compile(
    r"\b(loss|gain)\s+of\s+about\s+(\d+(?:[.,]\d+)?)\s*kg\b",
    re.IGNORECASE)
_BMI_RESULT = re.compile(r"\bBMI\s*[:=]?\s*(\d+(?:[.,]\d+)?)",
                         re.IGNORECASE)
_AGE_RESULT = re.compile(
    r"\bAge\s+(\d+)\s+years\s+(?:on|at admission)\s+"
    r"(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_AGE_BIRTH = re.compile(r"\bborn\s+(\d{4}-\d{2}-\d{2})\b",
                        re.IGNORECASE)


def _decimal_value(value: str) -> Decimal:
    return Decimal(value.replace(",", "."))


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _matches_stated_rounding(actual: Decimal, stated: Decimal) -> bool:
    decimal_places = max(0, -stated.as_tuple().exponent)
    quantum = Decimal(1).scaleb(-decimal_places)
    return actual.quantize(quantum, rounding=ROUND_HALF_UP) == stated


def _age_source_mismatch(item: dict, quote: str) -> str | None:
    result = _AGE_RESULT.search(item["text"])
    birth = _AGE_BIRTH.search(item["text"])
    if not result or not birth:
        return "derived input mismatch: age requires a birth date and target date"
    cited_dates = {value for _, _, value in _dates_in(quote)}
    if birth.group(1) not in cited_dates:
        return "derived input mismatch: quote contains no matching birth date"
    target_date = result.group(2)
    if target_date != REFERENCE_DATE and target_date not in cited_dates:
        return "derived input mismatch: quote contains no matching admission date"
    return None


def _derived_source_claim(item: dict, quote: str) -> tuple[str | None, str | None]:
    formula = item.get("formula")
    if formula == "age":
        return "Age input", None

    if formula == "weight_trend":
        result = _WEIGHT_TREND_RESULT.search(item["text"])
        input_text = item["text"][:result.start()] if result else item["text"]
        inputs = {_decimal_value(match.group(1))
                  for match in _WEIGHT_VALUE.finditer(input_text)}
        supported = {_decimal_value(match.group(1))
                     for match in _WEIGHT_VALUE.finditer(quote)}
        matches = sorted(inputs & supported)
        if not matches:
            return None, "derived input mismatch: quote contains no weight from the trend"
        return " ".join(f"Weight {_format_decimal(value)} kg" for value in matches), None

    if formula == "bmi":
        inputs = [
            *(f"Weight {_format_decimal(_decimal_value(match.group(1)))} kg"
              for match in _WEIGHT_VALUE.finditer(quote)),
            *(f"Height {_format_decimal(_decimal_value(match.group(1)))} cm"
              for match in _HEIGHT_VALUE.finditer(quote)),
        ]
        if not inputs:
            return None, "derived input mismatch: quote contains no BMI input value"
        return " ".join(inputs), None

    return item["text"], None


def _derived_result_mismatch(item: dict, sources: list[dict]) -> str | None:
    formula = item.get("formula")
    if formula == "age":
        result = _AGE_RESULT.search(item["text"])
        birth = _AGE_BIRTH.search(item["text"])
        if not result or not birth:
            return "derived result mismatch: age inputs or result are incomplete"
        target_date = result.group(2)
        cited_dates = {
            value
            for source in sources
            for _, _, value in _dates_in(source.get("quote", ""))
        }
        if birth.group(1) not in cited_dates:
            return "derived result mismatch: cited passages omit the birth date"
        if target_date != REFERENCE_DATE and target_date not in cited_dates:
            return "derived result mismatch: cited passages omit the admission date"
        try:
            birth_date = Date.fromisoformat(birth.group(1))
            event_date = Date.fromisoformat(target_date)
        except ValueError:
            return "derived result mismatch: age contains an invalid date"
        calculated = event_date.year - birth_date.year - (
            (event_date.month, event_date.day)
            < (birth_date.month, birth_date.day)
        )
        if calculated != int(result.group(1)):
            return (
                f"derived result mismatch: dates give age {calculated}, "
                f"but claim says {result.group(1)}"
            )
        return None

    if formula == "weight_trend":
        result = _WEIGHT_TREND_RESULT.search(item["text"])
        input_text = item["text"][:result.start()] if result else item["text"]
        values = [_decimal_value(match.group(1))
                  for match in _WEIGHT_VALUE.finditer(input_text)]
        source_values = [
            _decimal_value(match.group(1))
            for source in sources
            for match in _WEIGHT_VALUE.finditer(source.get("quote", ""))
        ]
        if not result or len(values) < 2:
            return "derived result mismatch: weight trend inputs or result are incomplete"
        if not Counter(values) <= Counter(source_values):
            return "derived result mismatch: cited inputs do not support all trend values"
        change = values[0] - values[-1]
        direction, stated = result.groups()
        expected_direction = (
            "loss" if change > 0 else "gain" if change < 0 else direction.casefold())
        if direction.casefold() != expected_direction:
            return (f"derived result mismatch: expected {expected_direction}, "
                    f"but claim says {direction.casefold()}")
        if not _matches_stated_rounding(abs(change), _decimal_value(stated)):
            return (f"derived result mismatch: cited weights give {abs(change):g} kg, "
                    f"but claim says {stated} kg")
        return None

    if formula == "bmi":
        result = _BMI_RESULT.search(item["text"])
        weights = {
            _decimal_value(match.group(1))
            for source in sources
            for match in _WEIGHT_VALUE.finditer(source.get("quote", ""))
        }
        heights = {
            _decimal_value(match.group(1))
            for source in sources
            for match in _HEIGHT_VALUE.finditer(source.get("quote", ""))
        }
        if not result or len(weights) != 1 or len(heights) != 1:
            return "derived result mismatch: BMI requires one cited weight and height"
        weight, height = next(iter(weights)), next(iter(heights))
        calculated = weight / ((height / Decimal(100)) ** 2)
        stated = _decimal_value(result.group(1))
        if not _matches_stated_rounding(calculated, stated):
            return (f"derived result mismatch: cited weight and height give "
                    f"BMI {calculated:.1f}, but claim says {stated}")
    return None


def cache_key() -> str:
    """Invalidates whenever any document, model, or prompt version changes."""
    return sha256_text("|".join([
        EXTRACTION_VERSION, MODEL_VERSION, PROMPT_VERSION]))


def load_domain(data_dir: Path | None = None):
    """Load documents + fact store + contradictions, with content-hash cache."""
    data_dir = data_dir or _data_dir
    data = load_patient(data_dir)
    key = sha256_text(
        "|".join(sorted(d.doc_hash for d in data.documents.values()))
        + cache_key())
    if key in _cache:
        return _cache[key][0], _cache[key][1], _cache[key][2]
    facts = build_fact_store(data)
    contra = detect_contradictions(data)
    _cache.clear()
    _cache[key] = (data, facts, contra, time.time())
    return data, facts, contra


def parse_situation(situation: str) -> tuple[str, str, str]:
    """(role, task_text, reference_date) from free text.

    The reference date is taken from an 'as of YYYY-MM-DD' clause when present
    and defaults to the scenario reference date; never from scenario IDs.
    """
    ref = REFERENCE_DATE
    m = re.search(r"as of\s+(\d{4}-\d{2}-\d{2})", situation)
    if m:
        ref = m.group(1)
    needs.load_profiles()
    role = needs.detect_role(situation)
    task = re.sub(r"(?i)as of\s+\d{4}-\d{2}-\d{2}", "", situation).strip(" .\n")
    return role, task or situation.strip(), ref


_MISSING_LABELS = [
    ("Renal function / eGFR", ("renal",)),
    ("Anesthesia / airway history", ("anesthesia", "airway")),
    ("HbA1c result", ("hba1c",)),
    ("Planned follow-up status", ("followup",)),
]


def _missing_items(section_dicts: list[dict]) -> list[MissingItem]:
    """Items a role needs that remain 'not documented'.

    Distinct from negated findings: a missing item means *no record*, not a
    normal or absent finding. A missing follow-up record does not prove the
    follow-up never occurred — it is labelled accordingly.
    """
    out: list[MissingItem] = []
    seen = set()
    for sec in section_dicts:
        for it in sec.get("items", []):
            if it.get("not_documented"):
                key = needs.missing_item_key(it)
                if key in seen:
                    continue
                seen.add(key)
                out.append(MissingItem(
                    label=it["text"],
                    needed_for=sec["title"],
                    status="not_documented"))
    return out


def _promote_empty_sections(section_dicts: list[dict]) -> None:
    for section in section_dicts:
        if not section.get("items") and section.get("more"):
            section["items"] = section["more"]
            section["more"] = []


def _merge_history_event_items(section_dicts: list[dict]) -> None:
    groups = {}
    for section in section_dicts:
        for bucket in ("items", "more"):
            retained = []
            for item in section.get(bucket, []):
                month = item.get("event_month")
                if item.get("category") != "history.event" or not month:
                    retained.append(item)
                    continue
                concept = item["text"].split(" — ", 1)[0]
                key = (concept.casefold(), month)
                group = groups.get(key)
                if group is None:
                    group = {
                        "concept": concept,
                        "items": [],
                        "details": [],
                        "dates": set(),
                        "sources": [],
                        "originals": [],
                    }
                    groups[key] = group
                    retained.append(item)
                group["items"].append(item)
                event_date = item.get("event_date")
                if event_date:
                    group["dates"].add(event_date)
                _, separator, detail_text = item["text"].partition(" — ")
                if separator:
                    group["details"].extend(
                        detail for detail in detail_text.split("; ")
                        if detail)
                group["sources"].extend(item.get("sources", []))
                original = item.get("original")
                if original and original not in group["originals"]:
                    group["originals"].append(original)
            section[bucket] = retained

    for (concept_key, month), group in groups.items():
        target = group["items"][0]
        details = list(dict.fromkeys(group["details"]))
        if concept_key == "hypertensive crisis":
            peaks = []
            symptoms = set()
            for detail in details:
                match = re.fullmatch(
                    r"peak BP (\d{2,3})/(\d{2,3}) mmHg", detail,
                    re.IGNORECASE)
                if match:
                    peaks.append((int(match.group(1)), int(match.group(2)),
                                  detail))
                elif detail.casefold() in {
                        "dizziness", "vision symptoms",
                        "dizziness, vision symptoms"}:
                    symptoms.update(
                        part.strip().casefold()
                        for part in detail.split(", "))
                else:
                    symptoms.update(
                        part for part in ("dizziness", "vision symptoms")
                        if part in detail.casefold())
            details = []
            if symptoms:
                details.append(", ".join(
                    part for part in ("dizziness", "vision symptoms")
                    if part in symptoms))
            details.extend(
                detail for detail in dict.fromkeys(group["details"])
                if not detail.casefold().startswith("peak bp ")
                and detail.casefold() not in {
                    "dizziness", "vision symptoms",
                    "dizziness, vision symptoms"})
            if peaks:
                details.append(max(peaks, key=lambda value: value[:2])[2])
        target["text"] = group["concept"] + (
            f" — {'; '.join(details)}" if details else "")
        target["date"] = needs.en_date(month)
        target["sources"] = []
        seen_sources = set()
        for source in group["sources"]:
            identity = (source.get("doc", source.get("doc_id")),
                        source.get("start"), source.get("end"))
            if identity not in seen_sources:
                target["sources"].append(source)
                seen_sources.add(identity)
        target["original"] = " / ".join(group["originals"]) or None
        exact_dates = {value for value in group["dates"]
                       if len(value.split("-")) == 3}
        if len(exact_dates) == 1:
            target["date"] = needs.en_date(next(iter(exact_dates)))


def _check_support(items: list[dict], data: PatientData,
                   adapter=None,
                   drop_log: list[dict[str, str]] | None = None
                   ) -> tuple[list[dict], int]:
    """Assess every source span and retain claims with any surviving support."""
    adapter = adapter or get_adapter()
    kept, dropped = [], 0
    for it in items:
        if it.get("not_documented"):
            kept.append(it)
            continue
        sources = it.get("sources", [])
        if not sources:
            continue
        surviving, verdicts = [], []
        for source in sources:
            claim = it["text"]
            quote = source.get("quote", "")
            original = it.get("original")
            if it.get("formula") == "age":
                claim = "Birth-date and admission-date inputs"
                reason = _age_source_mismatch(it, quote)
                verdict = "supported" if not reason else "unsupported"
            else:
                claim, reason = _derived_source_claim(it, quote)
                if not reason:
                    claim = claim or it["text"]
                    reason = support_mismatch_reason(claim, quote)
                    verdict = adapter.assess_support(claim, quote, original)
                else:
                    verdict = "unsupported"
            if reason:
                verdict = "unsupported"
            elif verdict == "unsupported":
                verdict = "unchecked"
            if verdict == "unsupported":
                dropped += 1
                if drop_log is not None:
                    drop_log.append({
                        "claim": claim,
                        "quote": quote,
                        "reason": reason or "support mismatch",
                    })
                continue
            surviving.append(source)
            verdicts.append(verdict)
        if not surviving:
            continue
        result_reason = _derived_result_mismatch(it, surviving)
        if result_reason:
            dropped += len(surviving)
            if drop_log is not None:
                for source in surviving:
                    drop_log.append({
                        "claim": it["text"],
                        "quote": source.get("quote", ""),
                        "reason": result_reason,
                    })
            continue
        it["sources"] = surviving
        it["support"] = ("supported" if "supported" in verdicts else "unchecked")
        kept.append(it)
    return kept, dropped


def review_status_for(adapter=None) -> str:
    if adapter is None:
        try:
            adapter = get_adapter()
        except ModelError:
            pass
    if adapter is None or isinstance(adapter, DeterministicAdapter):
        return "Rule-based extraction (no AI model) — pending clinician review"
    model_name = getattr(adapter, "model_name", None) or getattr(
        adapter, "name", "model")
    return (f"Rule-based extraction, claims checked by {model_name} "
            "— pending clinician review")


def build_screen_model(situation: str,
                       data_dir: Path | None = None,
                       drop_log: list[dict[str, str]] | None = None,
                       adapter=None,
                       ) -> tuple[Screen, dict]:
    """Full pipeline for one situation. Returns (Screen, debug_stats)."""
    errors: list[str] = []
    try:
        data, facts, contradictions = load_domain(data_dir)
    except Exception as e:  # data problems -> recoverable error state
        return Screen(
            role="clinician", task_text=situation, reference_date=REFERENCE_DATE,
            errors=[f"failed to load documents: {e}"],
            versions=_versions()), {"facts": 0}

    role, task, ref = parse_situation(situation)
    raw = needs.build_screen(situation, data, facts)
    # The key questions are checked and merged exactly like screen sections.
    question_boxes = [
        {"key": f"question.{index}", "question": answer["question"],
         "items": answer["items"], "more": []}
        for index, answer in enumerate(
            needs.build_key_questions(situation, data, facts))]

    dropped = 0
    try:
        adapter = adapter or get_adapter()
        for sec in [*question_boxes, *raw["sections"]]:
            sec["items"], n = _check_support(
                sec["items"], data, adapter, drop_log)
            dropped += n
            sec["more"], n = _check_support(
                sec.get("more", []), data, adapter, drop_log)
            dropped += n
    except ModelError as e:
        errors.append(f"claim-support assessment unavailable: {e}")
        dropped = 0
        adapter = None

    _merge_history_event_items(raw["sections"])
    _merge_history_event_items(question_boxes)
    _promote_empty_sections(raw["sections"])

    key_questions = [
        KeyQuestion(
            question=box["question"],
            items=[_screen_item(i) for i in box["items"]] or [ScreenItem(
                text="Not documented", not_documented=True)],
            contradiction_keys=related_contradictions(
                box["question"], box["items"], contradictions))
        for box in question_boxes
    ]

    sections = [
        ScreenSection(key=s["key"], title=s["title"],
                      items=[ScreenItem(
                          text=i["text"], original=i.get("original"),
                          date=i.get("date"),
                          derived=i.get("derived", False),
                          not_documented=i.get("not_documented", False),
                          status=i.get("status"),
                          support=i.get("support"),
                          formula=i.get("formula"),
                          sources=i.get("sources", []))
                          for i in s["items"]],
                      more=[ScreenItem(
                          text=i["text"], original=i.get("original"),
                          date=i.get("date"), derived=i.get("derived", False),
                          not_documented=i.get("not_documented", False),
                          status=i.get("status"), support=i.get("support"),
                          formula=i.get("formula"), sources=i.get("sources", []))
                          for i in s.get("more", [])])
        for s in raw["sections"]
        if s.get("items") or s.get("more")
    ]

    screen = Screen(
        role=role,
        task_text=task,
        reference_date=ref,
        key_questions=key_questions,
        sections=sections,
        contradictions=contradictions,
        missing=_missing_items(raw["sections"]),
        errors=errors,
        versions=_versions(),
        review_status=review_status_for(adapter),
    )
    stats = {"facts": len(facts), "contradictions": len(contradictions),
             "unsupported_dropped": dropped,
             "unsupported_spans_dropped": dropped}
    return screen, stats


def _screen_item(i: dict) -> ScreenItem:
    return ScreenItem(
        text=i["text"], original=i.get("original"), date=i.get("date"),
        derived=i.get("derived", False),
        not_documented=i.get("not_documented", False),
        status=i.get("status"), support=i.get("support"),
        formula=i.get("formula"), sources=i.get("sources", []))


# Words in contradiction titles that say *how* things conflict, not *what*.
_CONFLICT_WORDS = {
    "conflicting", "different", "values", "dates", "numbers", "number",
    "documents", "document", "match", "letter", "discharge", "outpatient",
    "hospital", "within", "between", "documented", "report", "reports",
}


def _topic_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9-]{4,}", text.casefold())
            if word not in _CONFLICT_WORDS}


def related_contradictions(question: str, items: list[dict],
                           contradictions: list[Contradiction]) -> list[str]:
    """Contradictions that concern this answer.

    A contradiction belongs to an answer when one of its passages overlaps a
    passage the answer cites, or when its topic (the title before any ':')
    names something the question or answer is about.
    """
    cited = [(src.get("doc"), src.get("start"), src.get("end"))
             for item in items for src in item.get("sources", [])]
    about = _topic_words(" ".join(
        [question, *(item["text"] for item in items)]))
    related = []
    for contradiction in contradictions:
        overlaps = any(
            span.doc_id == doc and isinstance(start, int) and isinstance(end, int)
            and span.start < end and start < span.end
            for span in contradiction.evidence for doc, start, end in cited)
        topic = contradiction.title.split(":", 1)[0].split("(", 1)[0]
        if overlaps or _topic_words(topic) & about:
            related.append(contradiction.key)
    return related


def _versions() -> dict:
    return {"extraction": EXTRACTION_VERSION, "prompt": PROMPT_VERSION,
            "model": MODEL_VERSION}
