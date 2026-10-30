"""PDF export preserving provenance and review status (optional feature 1)."""

from __future__ import annotations

from fpdf import FPDF

from .schemas import Screen
from .ui_helpers import derived_label

_SUBS = {"—": "-", "–": "-", "²": "2", "·": "-", "„": '"', "“": '"',
         "”": '"', "‚": "'", "‘": "'", "’": "'", "…": "...", "→": "->",
         "✓": "ok", "×": "x"}


def _s(text: str) -> str:
    for k, v in _SUBS.items():
        text = text.replace(k, v)
    text = text.encode("latin-1", "replace").decode("latin-1")
    # break tokens longer than the printable line so fpdf can wrap them
    return " ".join(
        w if len(w) <= 45 else " ".join(
            w[i:i + 40] for i in range(0, len(w), 40))
        for w in text.split(" "))


def render_pdf(screen: Screen, docs: dict | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, _s("CliniDocAI"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    pdf.multi_cell(0, 5, _s(
                   f"Role: {screen.role}   Task: {screen.task_text}\n"
                   f"Reference date: {screen.reference_date}\n"
                   "Review status: Draft – please verify against sources\n"
                   f"Versions: extraction={screen.versions.get('extraction')} "
                   f"model={screen.versions.get('model')} "
                   f"prompt={screen.versions.get('prompt')}"),
               new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    def header(t):
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, _s(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 9)

    for sec in screen.sections:
        header(sec.title)
        for it in sec.items:
            mark = "  [NOT DOCUMENTED]" if it.not_documented else ""
            src = "; ".join(f"{s.get('doc', s.get('doc_id'))} @{s.get('start')}" for s in it.sources)
            line = f"- {it.text}{mark}"
            if it.status and it.status != "positive":
                line += f"  [{it.status.upper()}]"
            if it.date:
                line += f"  ({it.date})"
            if it.derived and it.formula:
                line += ("  [" + derived_label(it.formula, len(it.sources))
                         .format(n=len(it.sources)) + "]")
            pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
            if src:
                pdf.set_font("helvetica", "I", 8)
                pdf.multi_cell(0, 4, _s(f"    sources: {src}"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", "", 9)
        pdf.ln(1)

    if screen.missing:
        header("Missing information")
        for m in screen.missing:
            pdf.multi_cell(0, 5, _s(f"- {m.label}  (needed for: {m.needed_for})"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    if screen.contradictions:
        header("Contradictions")
        for c in screen.contradictions:
            pdf.multi_cell(0, 5, _s(f"- {c.title} [{c.kind}]\n  {c.detail}"), new_x="LMARGIN", new_y="NEXT")
            src = "; ".join(f"{s.doc_id} @{s.start}" for s in c.evidence)
            pdf.set_font("helvetica", "I", 8)
            pdf.multi_cell(0, 4, _s(f"    sources: {src}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 9)
        pdf.ln(1)

    if screen.errors:
        header("Errors")
        for e in screen.errors:
            pdf.multi_cell(0, 5, _s(f"- {e}"), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
