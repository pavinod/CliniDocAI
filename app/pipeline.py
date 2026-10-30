"""Document loading and text preparation.

Loads the clinical documents of the synthetic patient and the JSON index that
the upstream Medical Cockpit pipeline produced. Documents are cleaned (`<BR>`
-> newline, HTML entities unescaped, markup stripped) while keeping a stable
character offset space: every fact and contradiction stores EvidenceSpan
(doc_id, start, end, quote, hash, date) into that space so the UI can highlight
the exact passage and provenance is fully recorded.

Documents are treated strictly as untrusted *data*: their text is parsed,
never interpreted as instructions.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import EvidenceSpan, sha256_text


@dataclass
class Document:
    doc_id: str
    title: str
    date: str          # document date, yyyy-mm-dd (partial dates preserved)
    doc_type: str
    text: str          # cleaned full text (offset space for all spans)
    doc_hash: str      # sha256 of the *original* document text
    lines: list[str] = field(default_factory=list)
    line_offsets: list[int] = field(default_factory=list)

    def span(self, start: int, end: int) -> EvidenceSpan:
        """Build a fully-populated evidence span; raises on bad offsets."""
        if not (0 <= start < end <= len(self.text)):
            raise ValueError(
                f"invalid span {start}..{end} for doc {self.doc_id} "
                f"(len {len(self.text)})")
        return EvidenceSpan(
            doc_id=self.doc_id, start=start, end=end,
            doc_hash=self.doc_hash, doc_date=self.date,
            quote=self.text[start:end].strip())

    def sentence_bounds(self, start: int, end: int) -> tuple[int, int]:
        """Expand a match to its containing sentence or table row."""
        if not (0 <= start < end <= len(self.text)):
            raise ValueError(f"invalid span {start}..{end} for doc {self.doc_id}")

        starts = [0]
        starts.extend(i + 1 for i, char in enumerate(self.text[:start]) if char == "\n")
        for match in re.finditer(r"[.!?](?=\s+[A-ZÄÖÜ]|\s*$)", self.text[:start]):
            if match.group() == ".":
                prefix = self.text[max(0, match.start() - 12):match.end()]
                if re.search(r"(?:\bA|St\.p|z\.B|Dr|li|re|Vd\.a|ca|bzw|ggf)\.$",
                             prefix, re.IGNORECASE):
                    continue
                if (match.start() > 0 and start < len(self.text)
                        and self.text[match.start() - 1].isdigit()
                        and self.text[match.end()].isdigit()):
                    continue
            starts.append(match.end())

        start_bound = max(starts)
        end_bound = len(self.text)
        newline = self.text.find("\n", end)
        if newline != -1:
            end_bound = newline
        for match in re.finditer(r"[.!?](?=\s+[A-ZÄÖÜ]|\s*$)", self.text[end:]):
            pos = end + match.start()
            if match.group() == ".":
                prefix = self.text[max(0, pos - 12):pos + 1]
                if re.search(r"(?:\bA|St\.p|z\.B|Dr|li|re|Vd\.a|ca|bzw|ggf)\.$",
                             prefix, re.IGNORECASE):
                    continue
                if (pos > 0 and pos + 1 < len(self.text)
                        and self.text[pos - 1].isdigit()
                        and self.text[pos + 1].isdigit()):
                    continue
            if newline == -1 or pos < newline:
                end_bound = pos + 1
                break
        return start_bound, end_bound

    def sentence_span(self, start: int, end: int) -> EvidenceSpan:
        """Build an evidence span expanded to a full sentence or table row."""
        return self.span(*self.sentence_bounds(start, end))

    def line_span(self, i: int) -> EvidenceSpan:
        start = self.line_offsets[i]
        return self.sentence_span(start, start + len(self.lines[i]))

    def quote(self, span: EvidenceSpan) -> str:
        return self.text[span.start:span.end].strip()


@dataclass
class PatientData:
    patient: dict
    documents: dict[str, Document]      # doc_id -> Document
    doc_order: list[str]                # doc_ids sorted by date
    index: dict                          # raw upstream extraction json


_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(raw: str) -> str:
    text = raw.replace("<BR>", "\n").replace("<br>", "\n")
    text = html.unescape(text)
    text = _TAG_RE.sub("", text)
    # normalise horizontal whitespace, drop trailing spaces, collapse >2 newlines
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_documents(data_dir: Path) -> tuple[dict[str, Document], list[str]]:
    meta = json.loads((data_dir / "documents.json").read_text(encoding="utf-8"))["documents"]
    docs: dict[str, Document] = {}
    for m in meta:
        raw = json.loads((data_dir / "documents" / f"{m['id']}.json").read_text(encoding="utf-8"))
        text = clean_text(raw["text"])
        lines = text.split("\n")
        offsets = []
        pos = 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln) + 1  # +1 for the \n
        docs[m["id"]] = Document(
            doc_id=m["id"],
            title=m.get("title", ""),
            date=m.get("date", ""),
            doc_type=m.get("type", ""),
            text=text,
            doc_hash=sha256_text(raw["text"]),
            lines=lines,
            line_offsets=offsets,
        )
    order = [d.doc_id for d in sorted(docs.values(), key=lambda d: d.date)]
    return docs, order


def find_spans(doc: Document, needle: str) -> list[EvidenceSpan]:
    """Locate `needle` in doc.text tolerating whitespace differences."""
    if not needle:
        return []
    spans = [doc.span(m.start(), m.end())
             for m in re.finditer(re.escape(needle), doc.text)]
    if spans:
        return spans
    norm_needle = " ".join(needle.split())
    if not norm_needle:
        return []
    pattern = r"\s+".join(re.escape(tok) for tok in norm_needle.split(" "))
    for m in re.finditer(pattern, doc.text):
        return [doc.span(m.start(), m.end())]
    return []


def load_patient(data_dir: Path) -> PatientData:
    data_dir = Path(data_dir)
    docs, order = load_documents(data_dir)
    index = {}
    for name in ("diagnoses", "procedures", "medication", "scores-all", "scores-latest", "cave"):
        p = data_dir / f"{name}.json"
        if p.exists():
            index[name] = json.loads(p.read_text(encoding="utf-8"))
    patient = json.loads((data_dir / "patient.json").read_text(encoding="utf-8"))
    return PatientData(patient=patient, documents=docs, doc_order=order, index=index)


def resolve_data_dir(root: Path) -> Path:
    """Accept either the patient data dir itself or a folder containing it."""
    if (root / "documents.json").exists() and (root / "documents").is_dir():
        return root
    if (root / "patient-0000-0001" / "documents.json").exists():
        return root / "patient-0000-0001"
    raise FileNotFoundError(
        f"No patient data found under {root}. Expected documents.json and documents/."
    )
