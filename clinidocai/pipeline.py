"""Pipeline orchestrator: load -> extract -> detect contradictions -> screen.

Public entry point matching the track-1 scaffold contract:

    screen, elapsed_seconds, llm_error = run_pipeline(role, task, data_root=...)

The heavy lifting lives in app/ (typed domain layer); this module adapts the
typed Screen model to the attribute shape the scaffold's UI/evaluator expect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.model import ModelAdapter, ModelError, get_adapter
from app.screen import REFERENCE_DATE, build_screen_model


@dataclass
class CompatEvidence:
    doc_id: str
    quote: str
    start: int = 0
    end: int = 0
    doc_title: str = ""
    doc_date: str = ""
    doc_hash: str = ""


@dataclass
class CompatFact:
    label: str
    value: Optional[str] = None
    evidence: list = field(default_factory=list)
    status: Optional[str] = None
    derived: bool = False
    formula: Optional[str] = None


@dataclass
class CompatMissing:
    label: str
    reason: str = "no record found"


@dataclass
class CompatSection:
    heading: str
    facts: list = field(default_factory=list)
    missing: list = field(default_factory=list)


@dataclass
class CompatContradiction:
    description: str
    contradiction_id: str = ""
    category: str = ""
    title: str = ""
    is_temporal_change: bool = False
    evidence: list = field(default_factory=list)


@dataclass
class CompatScreen:
    role: str
    task: str
    reference_date: str
    sections: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    review_status: str = "Rule-based extraction (no AI model) — pending clinician review"
    patient_display_name: str = "patient"
    raw: object = None   # the typed app.schemas.Screen


def _ev(src: dict) -> CompatEvidence:
    return CompatEvidence(
        doc_id=src.get("doc", src.get("doc_id", "")),
        quote=src.get("quote", ""),
        start=src.get("start", 0),
        end=src.get("end", 0),
        doc_title=src.get("docTitle", ""),
        doc_date=src.get("docDate", ""),
        doc_hash=src.get("docHash", ""))


def run_pipeline(
    role: str,
    task: str,
    data_root: str = "data/patient-0000-0001",
    reference_date: str = REFERENCE_DATE,
    patient_display_name: str = "patient",
    adapter: ModelAdapter | None = None,
    summarise: bool = False,
) -> tuple[CompatScreen, float, str]:
    """Run the full pipeline and return (screen, elapsed_seconds, llm_error).

    `role` + `task` are combined into one free-text situation; `role` may be a
    full sentence like "You are the anaesthetist" or a bare role name —
    detection is keyword-based either way.
    """
    t0 = time.perf_counter()
    llm_error = ""
    situation = task if role.lower() in task.lower() else f"{role}. {task}"
    if reference_date not in situation:
        situation = situation.rstrip(" .") + f" as of {reference_date}"

    screen, stats = build_screen_model(situation, Path(data_root))
    elapsed = time.perf_counter() - t0

    if summarise and not isinstance(adapter or get_adapter(), type(get_adapter())):
        pass  # summarisation is optional; deterministic adapter has nothing to add
    if summarise:
        try:
            ad = adapter or get_adapter()
            if not hasattr(ad, "summarise_screen"):
                llm_error = "adapter does not support summarisation"
        except ModelError as e:
            llm_error = str(e)

    sections = []
    for sec in screen.sections:
        facts, missing = [], []
        for it in sec.items:
            if it.not_documented:
                missing.append(CompatMissing(label=it.text,
                                             reason=f"needed for: {sec.title}"))
                continue
            label = it.text
            facts.append(CompatFact(
                label=label, value=it.date, evidence=[_ev(s) for s in it.sources],
                status=it.status, derived=it.derived, formula=it.formula))
        sections.append(CompatSection(heading=sec.title, facts=facts,
                                      missing=missing))
    compat = CompatScreen(
        role=screen.role, task=screen.task_text,
        reference_date=screen.reference_date,
        sections=sections,
        contradictions=[CompatContradiction(
            description=f"{c.title}. {c.detail}", contradiction_id=c.key,
            category=c.kind,
            title=c.title, evidence=[_ev({"doc": e.doc_id, "quote": e.quote,
                                          "start": e.start, "end": e.end})
                                     for e in c.evidence])
            for c in screen.contradictions],
        missing=[CompatMissing(label=m.label, reason=f"needed for: {m.needed_for}")
                 for m in screen.missing],
        errors=screen.errors,
        review_status=screen.review_status,
        patient_display_name=patient_display_name,
        raw=screen,
    )
    return compat, elapsed, llm_error
