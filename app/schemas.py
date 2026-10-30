"""Typed schemas for CliniDocAI's domain layer.

Everything the UI or the eval pipeline sees goes through these models:
documents, evidence spans, facts, task profiles, missing items,
contradictions and the generated screen. Validation rejects malformed
evidence (bad offsets, empty quotes, unknown document ids at render time).
"""
from __future__ import annotations

import hashlib
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

EXTRACTION_VERSION = "extract-v1.4"
PROMPT_VERSION = "deterministic-v1.1"   # prompt version of the optional model adapter
MODEL_VERSION = "deterministic-v1.1"    # version tag reported for deterministic results

FactStatus = Literal["positive", "negative", "suspected", "historical",
                     "recommended", "completed", "planned"]
Support = Literal["supported", "unsupported", "unchecked"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EvidenceSpan(BaseModel):
    """One exact passage a fact links to."""
    doc_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    doc_hash: str
    doc_date: Optional[str] = None           # partial dates preserved ("2016-04")
    quote: str = Field(min_length=1)
    extractor_version: str = EXTRACTION_VERSION

    @model_validator(mode="after")
    def _range_ok(self):
        if self.end <= self.start:
            raise ValueError("evidence span end must be after start")
        return self


class Fact(BaseModel):
    """A clinical fact derived from documents (or derived/calculated from them)."""
    key: str
    category: str                          # e.g. medication.antithrombotic
    text: str                              # English, human-readable claim
    original: Optional[str] = None         # wording retained from the source
    date: Optional[str] = None             # partial-date precision is kept
    status: FactStatus = "positive"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    kw: set[str] = Field(default_factory=set)   # relevance keywords for task matching
    derived: bool = False                  # translated/computed from the passages
    formula: Optional[str] = None          # required for calculated facts
    inputs: list[str] = Field(default_factory=list)  # keys of input facts for calculated facts
    support: Support = "unchecked"         # claim-support verdict (model adapter)


class MissingItem(BaseModel):
    """Information the task needs but the documents do not contain.
    Distinct from a negative finding (an affirmed absence)."""
    label: str
    needed_for: str
    status: Literal["not_documented"] = "not_documented"


class Contradiction(BaseModel):
    key: str
    title: str
    detail: str                            # why the passages are incompatible
    kind: str                              # laterality | numeric | date | negation | identity
    evidence: list[EvidenceSpan] = Field(min_length=2)


class ScreenItem(BaseModel):
    text: str
    original: Optional[str] = None
    date: Optional[str] = None
    derived: bool = False
    not_documented: bool = False
    status: Optional[FactStatus] = None
    support: Optional[Support] = None
    formula: Optional[str] = None
    sources: list[dict] = Field(default_factory=list)   # serialised EvidenceSpan + docTitle


class ScreenSection(BaseModel):
    key: str
    title: str
    kind: str = "facts"
    items: list[ScreenItem] = Field(default_factory=list)
    more: list[ScreenItem] = Field(default_factory=list)


class TaskProfile(BaseModel):
    role: str
    task_text: str
    reference_date: str                    # "as of" date for the screen


class Screen(BaseModel):
    role: str
    task_text: str
    reference_date: str
    sections: list[ScreenSection] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    missing: list[MissingItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)      # recoverable issues shown to the user
    versions: dict = Field(default_factory=dict)
    review_status: str = "Rule-based extraction (no AI model) — pending clinician review"

    def to_dict(self) -> dict:
        return self.model_dump()
