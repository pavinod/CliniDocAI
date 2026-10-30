"""Printable report: the screen laid out as a clean A4 document.

Built as standalone HTML (no Streamlit chrome), so a printout shows the
role's key questions and answers first, each fact with its date and source
passage, the contradictions that concern each answer, then missing
information, other contradictions and the remaining facts. The draft /
review status is printed on every page footer.
"""

from __future__ import annotations

import html
from typing import Callable

from .labels import doc_type_label
from .ui_helpers import date_caption, fact_display_text

_CSS = """
@page { size: A4; margin: 16mm 15mm 18mm;
        @bottom-center { content: "Draft – verify against sources · page "
                         counter(page); font: 8pt sans-serif; color: #666; } }
* { box-sizing: border-box; }
body { font: 10pt/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       color: #1d1d1f; margin: 0; }
header { border-bottom: 2px solid #1d1d1f; padding-bottom: 6pt; margin-bottom: 10pt; }
h1 { font-size: 15pt; margin: 0 0 2pt; }
.meta { display: grid; grid-template-columns: auto 1fr; gap: 1pt 10pt;
        font-size: 9pt; margin-top: 4pt; }
.meta dt { color: #555; } .meta dd { margin: 0; }
.draft { display: inline-block; margin-top: 6pt; padding: 2pt 6pt;
         border: 1px solid #b26a00; color: #b26a00; font-size: 8.5pt;
         border-radius: 3pt; }
h2 { font-size: 11.5pt; margin: 14pt 0 6pt; padding-bottom: 2pt;
     border-bottom: 1px solid #ccc; }
.question { border: 1px solid #d0d0d0; border-radius: 4pt; padding: 7pt 9pt;
            margin-bottom: 8pt; break-inside: avoid; }
.question h3 { font-size: 10.5pt; margin: 0 0 4pt; }
.fact { margin: 3pt 0 5pt; }
.fact .text { font-weight: 600; }
.fact .date, .badge { color: #555; font-size: 9pt; }
.badge.missing { color: #b26a00; font-weight: 600; }
.source { color: #555; font-size: 8.5pt; margin-left: 10pt; }
.source q { font-style: italic; }
.conflict { border-left: 3px solid #b3261e; background: #fbeeee;
            padding: 4pt 7pt; margin: 6pt 0 2pt; break-inside: avoid; }
.conflict strong { color: #b3261e; }
.allergy { border-left: 3px solid #b3261e; padding-left: 7pt; }
table { width: auto; min-width: calc(210mm - 30mm); border-collapse: collapse; font-size: 9pt; }
th { text-align: left; background: #f2f2f2; }
th, td { border-bottom: 1px solid #e3e3e3; padding: 3pt 5pt; vertical-align: top; }
td.section { font-weight: 600; width: 24%; }
td.date { white-space: nowrap; }
.status { color: #555; font-style: italic; }
ul.missing { margin: 0; padding-left: 14pt; }
footer { margin-top: 14pt; padding-top: 5pt; border-top: 1px solid #ccc;
         font-size: 8pt; color: #666; }
"""


def _e(text) -> str:
    return html.escape(str(text or ""))


def _source_lines(sources, documents) -> str:
    lines = []
    seen = set()
    for source in sources or []:
        doc_id = source.get("doc") or source.get("doc_id")
        quote = " ".join(str(source.get("quote") or "").split())
        if (doc_id, quote) in seen:
            continue
        seen.add((doc_id, quote))
        doc = (documents or {}).get(doc_id)
        title = doc_type_label(source.get("docTitle") or (doc.title if doc else doc_id))
        date = source.get("docDate") or (doc.date if doc else "")
        if len(quote) > 160:
            quote = quote[:160].rsplit(" ", 1)[0] + " …"
        lines.append(f'<div class="source">{_e(title)}, {_e(date)}: '
                     f"<q>{_e(quote)}</q></div>")
    return "".join(lines[:3])


def _fact(item, documents, lang, tr) -> str:
    text = tr(item.text) if item.not_documented else fact_display_text(item, lang)
    date = date_caption(text, item.date)
    badge = ""
    if item.not_documented and tr("not documented").casefold() not in text.casefold():
        badge = f' <span class="badge missing">· {_e(tr("not documented"))}</span>'
    elif item.status and item.status != "positive":
        badge = f' <span class="badge">· {_e(tr(str(item.status)))}</span>'
    return (f'<div class="fact"><span class="text">{_e(text)}</span>{badge}'
            + (f' <span class="date">· {_e(date)}</span>' if date else "")
            + _source_lines(item.sources, documents) + "</div>")


def _conflict(contradiction) -> str:
    return (f'<div class="conflict"><strong>⚠ {_e(contradiction.title)}</strong>'
            f"<div>{_e(contradiction.detail)}</div></div>")


def print_html(screen, documents=None, patient_line: str = "",
               lang: str = "en", tr: Callable[[str], str] = lambda s: s,
               generated: str = "", task: str = "") -> str:
    """Standalone, print-ready HTML for one screen."""
    by_key = {c.key: c for c in screen.contradictions}
    attached = {key for q in screen.key_questions for key in q.contradiction_keys}

    parts = [f"<!doctype html><html lang='{lang}'><head><meta charset='utf-8'>"
             f"<title>CliniDocAI — {_e(screen.role)}</title><style>{_CSS}</style>"
             "</head><body>"]
    parts.append(
        "<header><h1>CliniDocAI · " + _e(tr("Clinical summary")) + "</h1>"
        "<dl class='meta'>"
        f"<dt>{_e(tr('Patient'))}</dt><dd>{_e(patient_line)}</dd>"
        f"<dt>{_e(tr('Role'))}</dt><dd>{_e(screen.role.replace('_', ' ').title())}</dd>"
        f"<dt>{_e(tr('Task'))}</dt><dd>{_e(task or screen.task_text)}</dd>"
        f"<dt>{_e(tr('Reference date'))}</dt><dd>{_e(screen.reference_date)}</dd>"
        + (f"<dt>{_e(tr('Printed'))}</dt><dd>{_e(generated)}</dd>" if generated else "")
        + "</dl>"
        f"<div class='draft'>{_e(screen.review_status)}</div></header>")

    parts.append(f"<h2>{_e(tr('Key questions for this role'))}</h2>")
    for number, question in enumerate(screen.key_questions, 1):
        parts.append(f"<section class='question'><h3>{_e(tr('Q'))}{number}. "
                     f"{_e(tr(question.question))}</h3>")
        parts.extend(_fact(item, documents, lang, tr) for item in question.items)
        parts.extend(_conflict(by_key[key]) for key in question.contradiction_keys
                     if key in by_key)
        parts.append("</section>")

    if screen.missing:
        parts.append(f"<h2>{_e(tr('Missing information'))}</h2><ul class='missing'>")
        parts.extend(f"<li>{_e(tr(m.label))}</li>" for m in screen.missing)
        parts.append("</ul>")

    others = [c for c in screen.contradictions if c.key not in attached]
    if others:
        parts.append(f"<h2>{_e(tr('Other contradictions in the documents'))}</h2>")
        parts.extend(_conflict(c) for c in others)

    parts.append(f"<h2>{_e(tr('All facts for this role'))}</h2><table>"
                 f"<tr><th>{_e(tr('Section'))}</th><th>{_e(tr('Fact'))}</th>"
                 f"<th>{_e(tr('Date'))}</th><th>{_e(tr('Source'))}</th></tr>")
    for section in screen.sections:
        items = [*section.items, *section.more]
        for index, item in enumerate(items):
            text = (tr(item.text) if item.not_documented
                    else fact_display_text(item, lang))
            # The status changes the meaning ("negative" = finding absent),
            # so it is always printed next to the fact.
            status = ("" if item.not_documented or item.status in (None, "positive")
                      else f" <span class='status'>— {_e(tr(str(item.status)))}</span>")
            source = item.sources[0] if item.sources else {}
            doc_label = (f"{doc_type_label(source.get('docTitle') or '')}, "
                         f"{source.get('docDate') or ''}") if source else "—"
            row_class = " class='allergy'" if section.key == "allergy" else ""
            parts.append(
                "<tr>"
                + (f"<td class='section' rowspan='{len(items)}'>"
                   f"{_e(tr(section.title))}</td>" if index == 0 else "")
                + f"<td{row_class}>{_e(text)}{status}</td>"
                f"<td class='date'>{_e(item.date or '')}</td>"
                f"<td>{_e(doc_label)}</td></tr>")
    parts.append("</table>")

    parts.append(
        "<footer>" + _e(tr("Generated from the supplied documents only. Draft pending "
                           "clinician review — not for clinical use."))
        + f" · {_e(tr('Versions'))}: {_e(screen.versions)}</footer></body></html>")
    return "".join(parts)
