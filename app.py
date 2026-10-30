"""CliniDocAI — hackathon prototype.

Given a clinician's role and task, show one screen of source-supported
patient facts, missing information, and contradictions. Run:

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from pathlib import Path

import streamlit as st
import yaml

from app.pdf import render_pdf
from app.schemas import EXTRACTION_VERSION, MODEL_VERSION, PROMPT_VERSION
from app.screen import (
    REFERENCE_DATE, build_screen_model, doc_type_label, load_domain,
    review_status_for,
)
from app.ui_helpers import (
    derived_label, fact_display_text, format_missing_item, resolve_role,
)
from app.i18n import DE, current_lang, role_label, t
from app import voice
from app.voice_ui import (
    transcribe_task_audio, voice_control_available,
    render_realtime_stt, render_file_upload_stt,
)

st.set_page_config(page_title="CliniDocAI", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data" / "patient-0000-0001"
PROFILE_PATH = Path(__file__).resolve().parent / "config" / "profiles.yaml"
PROFILE_CONFIG = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8")) or {}
OTHER_ROLE = "Other (type below)"
ROLE_OPTIONS = {
    role.upper() if len(role) <= 3 else role.replace("_", " ").title(): role
    for role in PROFILE_CONFIG.get("role_needs", {})
}
EXAMPLES = PROFILE_CONFIG.get("examples", [])
EXAMPLES_BY_LABEL = {example["label"]: example for example in EXAMPLES}
EXAMPLE_OPTIONS = ["Choose an example…", *EXAMPLES_BY_LABEL]
ROLE_DISPLAY = {value: label for label, value in ROLE_OPTIONS.items()}


# Streamlit serializes the formatted (localized) label into session state, so a
# language switch can store e.g. 'Anästhesist' instead of 'Anaesthetist'. These
# maps normalize any label (EN/DE/key) back to its canonical English value.
def _lang_map(english: str) -> dict[str, str]:
    return {english: english, DE.get(english, english): english}


ROLE_LABEL_MAP: dict[str, str] = {}
for _english in [OTHER_ROLE, *ROLE_OPTIONS]:
    ROLE_LABEL_MAP.update(_lang_map(_english))
for _key in ROLE_OPTIONS.values():
    ROLE_LABEL_MAP[_key] = ROLE_DISPLAY[_key]
EXAMPLE_LABEL_MAP: dict[str, str] = {}
for _label in EXAMPLE_OPTIONS:
    EXAMPLE_LABEL_MAP.update(_lang_map(_label))
VOICE_LABEL_MAP = {"English": "English", "German": "German",
                   DE["English"]: "English", DE["German"]: "German"}


def _canon(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def _role_display(role: str) -> str:
    return role_label(role)

# -------------------------------------------------------------- session state
st.session_state.setdefault("font_scale", 100)
st.session_state.setdefault("screen", None)
st.session_state.setdefault("stats", {})
st.session_state.setdefault("selected_src", None)
st.session_state.setdefault("role_choice", OTHER_ROLE)
st.session_state.setdefault("role_other", "")
st.session_state.setdefault("task_input", "")
st.session_state.setdefault("example_choice", EXAMPLE_OPTIONS[0])
st.session_state.setdefault("built_role", "")
st.session_state.setdefault("built_task", "")
st.session_state.setdefault("example_task", "")
st.session_state.setdefault("role_changed_needs_task", False)
st.session_state.setdefault("canon_role", OTHER_ROLE)
st.session_state.setdefault("screen_requested", False)
st.session_state.setdefault("voice_language", "English")
st.session_state.setdefault("voice_audio_hash", "")
st.session_state.setdefault("voice_transcript", "")   # final accepted transcript
st.session_state.setdefault("voice_error", "")
st.session_state.setdefault("stt_upload_hash", "")
st.session_state.setdefault("stt_pending_transcript", "")  # draft before "Use as task"
st.session_state.setdefault("stt_recording", False)        # mic is live
st.session_state.setdefault("ui_lang", "en")
st.session_state.setdefault("ui_lang_seg", "EN")
VOICE_AVAILABILITY = voice.is_available()
WEBRTC_AVAILABLE   = voice.webrtc_available()
_WHISPER_SIZE      = __import__("os").getenv("WHISPER_MODEL", "base")


def _lang_changed() -> None:
    st.session_state.ui_lang = (
        st.session_state.ui_lang_seg or "EN").lower()


def _transcribe_audio() -> None:
    audio = st.session_state.get("task_audio")
    if audio is None:
        return
    audio_bytes = audio.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()
    if digest == st.session_state.voice_audio_hash:
        return
    st.session_state.voice_audio_hash = digest
    language = {"English": "en", "German": "de"}.get(
        _canon(st.session_state.voice_language, VOICE_LABEL_MAP), "en")
    task, error = transcribe_task_audio(
        audio_bytes, language, voice.transcribe,
        current_task=st.session_state.task_input)
    if error:
        st.session_state.voice_error = error
        return
    st.session_state.task_input = task
    st.session_state.voice_transcript = task
    st.session_state.voice_error = ""


def _role_changed() -> None:
    choice = _canon(st.session_state.role_choice, ROLE_LABEL_MAP)
    st.session_state.role_choice = choice
    if choice == st.session_state.canon_role:
        return  # only the label language changed, not the role
    st.session_state.canon_role = choice
    example_task = st.session_state.example_task
    if example_task and st.session_state.task_input.strip() == example_task:
        st.session_state.task_input = ""
        st.session_state.role_changed_needs_task = True
    else:
        st.session_state.role_changed_needs_task = False
    st.session_state.role_other = ""
    st.session_state.example_choice = EXAMPLE_OPTIONS[0]
    st.session_state.selected_src = None


def _task_changed() -> None:
    if st.session_state.task_input.strip():
        st.session_state.role_changed_needs_task = False


def _apply_example() -> None:
    st.session_state.example_choice = _canon(
        st.session_state.example_choice, EXAMPLE_LABEL_MAP)
    example = EXAMPLES_BY_LABEL.get(st.session_state.example_choice)
    if not example:
        return
    role = example["role"]
    display_role = next(
        (label for label, value in ROLE_OPTIONS.items() if value == role), None)
    st.session_state.role_choice = display_role or OTHER_ROLE
    st.session_state.role_other = "" if display_role else role
    st.session_state.task_input = t(example["task"])
    st.session_state.example_task = t(example["task"])
    st.session_state.role_changed_needs_task = False
    st.session_state.selected_src = None

_top_spacer, _top_lang = st.columns([5, 1])
with _top_lang:
    st.segmented_control(t("Language"), ["EN", "DE"], key="ui_lang_seg",
                         on_change=_lang_changed,
                         label_visibility="collapsed")

with st.sidebar:
    st.title("CliniDocAI")
    st.selectbox(t("Role"), [OTHER_ROLE, *ROLE_OPTIONS], key="role_choice",
                 format_func=role_label, on_change=_role_changed)
    st.text_input(t("Other role (optional)"), key="role_other",
                  placeholder=t("Enter another clinical role"))
    st.text_area(t("Task"), height=75, key="task_input",
                 placeholder=t("What should this clinician review?"),
                 on_change=_task_changed)

    # ================================================================
    # Voice / STT panel — matches the screenshot layout:
    #   Language selector
    #   Model caption
    #   Option A — Live microphone  [▶ Start recording]
    #   Option B — Upload audio file  [file uploader]
    #   [✓ Done]
    #   Transcript (edit if needed):  [editable text area]
    #   [Use as task ↵]  [Clear]
    # ================================================================
    if voice_control_available(VOICE_AVAILABILITY):
        voice_lang_code = {"English": "en", "German": "de"}.get(
            _canon(st.session_state.voice_language, VOICE_LABEL_MAP), "en")

        # ── Language ──────────────────────────────────────────────────
        st.caption(t("Language"))
        _lang_col_en, _lang_col_de = st.columns(2)
        with _lang_col_en:
            if st.button("🇬🇧 English" if voice_lang_code != "en" else "🟢 English",
                         use_container_width=True, key="stt_lang_en"):
                st.session_state.voice_language = "English"
                st.rerun()
        with _lang_col_de:
            if st.button("🇩🇪 German" if voice_lang_code != "de" else "🟢 German",
                         use_container_width=True, key="stt_lang_de"):
                st.session_state.voice_language = "German"
                st.rerun()

        # ── Model caption ─────────────────────────────────────────────
        st.caption(t("Model: Whisper {size} (local, no internet)",
                     size=_WHISPER_SIZE))

        # ── Option A — Live microphone ────────────────────────────────
        st.markdown(f"**{t('Option A — Live microphone')}**")
        if WEBRTC_AVAILABLE:
            # WebRTC real-time streaming path — transcript chunks are drained
            # into stt_pending_transcript by a self-refreshing fragment inside
            # render_realtime_stt, so no manual polling is needed here.
            render_realtime_stt(voice_lang_code, t)
        else:
            # Fallback: Streamlit built-in one-shot recorder
            st.audio_input(t("▶ Start recording"), key="task_audio",
                           on_change=_transcribe_audio,
                           label_visibility="collapsed")
            if st.session_state.voice_transcript and not st.session_state.stt_pending_transcript:
                st.session_state.stt_pending_transcript = st.session_state.voice_transcript

        # ── Option B — Upload audio file ──────────────────────────────
        st.markdown(f"**{t('Option B — Upload audio file')}**")
        st.caption("WAV / MP3 / M4A / OGG")

        upload_text, upload_err = render_file_upload_stt(voice_lang_code, t)
        if upload_err:
            st.session_state.voice_error = upload_err
        elif upload_text:
            st.session_state.stt_pending_transcript = upload_text
            st.session_state.voice_error = ""

        # ── Done button ───────────────────────────────────────────────
        if st.session_state.stt_pending_transcript:
            if st.button(t("✓ Done"), use_container_width=True, key="stt_done"):
                pass   # pending transcript stays; user edits below then accepts

        # ── Transcript editor ─────────────────────────────────────────
        if st.session_state.stt_pending_transcript:
            st.markdown(f"**{t('Transcript (edit if needed):')}**")
            edited = st.text_area(
                t("Transcript (edit if needed):"),
                value=st.session_state.stt_pending_transcript,
                height=100,
                key="stt_transcript_editor",
                label_visibility="collapsed",
            )
            st.session_state.stt_pending_transcript = edited

            _col_use, _col_clear = st.columns(2)
            with _col_use:
                if st.button(t("Use as task ↵"), use_container_width=True,
                             key="stt_use", type="primary"):
                    st.session_state.task_input = edited.strip()
                    st.session_state.voice_transcript = edited.strip()
                    st.session_state.stt_pending_transcript = ""
                    st.session_state.voice_error = ""
                    st.rerun()
            with _col_clear:
                if st.button(t("Clear"), use_container_width=True, key="stt_clear"):
                    st.session_state.stt_pending_transcript = ""
                    st.session_state.voice_transcript = ""
                    st.session_state.voice_error = ""
                    st.rerun()

        if st.session_state.voice_error:
            st.warning(t("Voice transcription unavailable: {error}",
                         error=st.session_state.voice_error))

    elif not voice_control_available(VOICE_AVAILABILITY):
        st.caption(t("Voice input unavailable (install requirements-voice.txt)"))

    st.selectbox(t("Examples"), EXAMPLE_OPTIONS, key="example_choice",
                 format_func=t, on_change=_apply_example)
    show_facts = st.button(
        t("Show patient facts"), type="primary", use_container_width=True)
    with st.expander(t("Settings"), expanded=False):
        st.slider(t("Text size (%)"), 80, 160, key="font_scale", step=10)
        st.caption(t("Reference date"))
        st.text(REFERENCE_DATE)
        st.selectbox(t("Voice language"), ["English", "German"],
                     format_func=t, key="voice_language")
        if not voice_control_available(VOICE_AVAILABILITY):
            st.caption(t(
                "Voice input unavailable (install requirements-voice.txt)"))

st.markdown(
    f"<style>html {{font-size:{st.session_state.font_scale}% !important;}}"
    ".stApp, p, div, li, td {font-size:1rem;}"
    "[data-testid=\"stMainBlockContainer\"] div[class*=\"st-key-src-\"] "
    "button[data-testid=\"stBaseButton-secondary\"],"
    "[data-testid=\"stMainBlockContainer\"] div[class*=\"st-key-contra-\"] "
    "button[data-testid=\"stBaseButton-secondary\"] {"
    "height:1.8rem;min-height:1.8rem;padding:0 .5rem;"
    "font-size:.85rem;line-height:1.2;}"
    "mark {background:#ffe08a;padding:0 2px;}</style>",
    unsafe_allow_html=True)

if show_facts:
    st.session_state.screen_requested = True
    st.session_state.selected_src = None

if st.session_state.screen_requested:
    role = resolve_role(
        _canon(st.session_state.role_choice, ROLE_LABEL_MAP),
        st.session_state.role_other)
    task = st.session_state.task_input.strip()
    if not task or not role:
        st.session_state.screen = None
    else:
        situation = f"You are the {role}. {task} as of {REFERENCE_DATE}."
        t0 = time.perf_counter()
        with st.spinner(t("Reading documents…")):
            screen, stats = build_screen_model(situation, DATA_DIR)
        st.session_state.screen = screen
        st.session_state.stats = stats | {
            "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
        st.session_state.selected_src = None
        st.session_state.built_role = role
        st.session_state.built_task = task
        st.session_state.role_changed_needs_task = False

if st.session_state.screen_requested and not st.session_state.task_input.strip():
    if st.session_state.role_changed_needs_task:
        st.info(t("Role changed — enter a task for the new role"))
    else:
        st.warning(t("Enter a task before showing patient facts."))
elif st.session_state.screen_requested and not resolve_role(
        _canon(st.session_state.role_choice, ROLE_LABEL_MAP),
        st.session_state.role_other):
    st.warning(t("Choose a role or enter another role."))

def _source_label(title: str, date: str) -> str:
    label = doc_type_label(title)
    label = f"{label} · {date}" if date else label
    return _markdown_escape(label)


def _markdown_escape(text: str) -> str:
    return re.sub(r"([*_`#])", r"\\\1", text)


def _group_sources(sources, documents):
    grouped = {}
    for source in sources:
        if isinstance(source, dict):
            doc_id = source.get("doc") or source.get("doc_id")
            start, end = source.get("start"), source.get("end")
            title, date = source.get("docTitle"), source.get("docDate")
        else:
            doc_id = getattr(source, "doc_id", None)
            start, end = getattr(source, "start", None), getattr(source, "end", None)
            title, date = None, getattr(source, "doc_date", None)
        if not doc_id or not isinstance(start, int) or not isinstance(end, int):
            continue
        doc = documents.get(doc_id) if documents else None
        group = grouped.setdefault(doc_id, {
            "doc": doc_id,
            "title": doc_type_label(
                title or (doc.title if doc else str(doc_id))),
            "date": date or (doc.date if doc else ""),
            "spans": [],
        })
        span = (start, end)
        if span not in group["spans"]:
            group["spans"].append(span)
    for group in grouped.values():
        group["spans"].sort()
    return list(grouped.values())


def _highlight_spans(text: str, spans) -> str:
    ranges = sorted({
        (start, min(end, len(text)))
        for start, end in spans
        if isinstance(start, int) and isinstance(end, int)
        and 0 <= start < end <= len(text)
    })
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    parts = []
    cursor = 0
    for start, end in merged:
        parts.extend((
            html.escape(text[cursor:start]),
            f"<mark>{html.escape(text[start:end])}</mark>",
        ))
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def _contradiction_kind_label(kind: str) -> str:
    lowered = kind.casefold()
    if "laterality" in lowered:
        return "Left/right mismatch"
    if "numeric" in lowered or "bmi" in lowered:
        return "Values don't add up"
    if "identity" in lowered or "case_id" in lowered or "case" in lowered:
        return "Case number mismatch"
    if "negation" in lowered:
        return "Finding present vs absent"
    if "temporal" in lowered:
        return "Timeline inconsistency"
    if "date" in lowered:
        return "Date mismatch"
    return kind[:1].upper() + kind[1:]


screen = st.session_state.screen
data = None
if screen is not None:
    try:
        data, facts, _ = load_domain(DATA_DIR)
    except Exception:
        data = None
        facts = []

    # ---------------------------------------------------------- header
    role_col, task_col, date_col = st.columns([1, 3, 1])
    with role_col:
        st.caption(t("Role"))
        st.text(_role_display(st.session_state.built_role or screen.role))
    with task_col:
        st.caption(t("Task"))
        st.text(st.session_state.built_task or screen.task_text)
    with date_col:
        st.caption(t("Reference date"))
        st.text(screen.reference_date)
    demographics = next((fact for fact in facts
                         if fact.key == "patient.demographics"), None)
    age = next((fact for fact in facts
                if fact.formula == "age" and fact.date == REFERENCE_DATE), None)
    if demographics and age:
        birth = re.search(r"\bborn\s+(\d{4}-\d{2}-\d{2})\b", demographics.text)
        sex = demographics.text.split(", born", 1)[0]
        age_years = age.text.split(" years ", 1)[0].removeprefix("Age ")
        patient_line = f"{sex} · {age_years} {t('years')}"
        if birth:
            patient_line += f" ({t('born')} {birth.group(1)})"
    elif demographics:
        patient_line = f"{demographics.text} · {t('Age: not documented')}"
    else:
        patient_line = t("Age: not documented")
    st.caption(patient_line)
    st.info(t("Draft – please verify against sources"))
    task_caption = st.session_state.built_task or re.sub(
        r"^You are the [^.]+\.\s*", "", screen.task_text,
        flags=re.IGNORECASE)
    st.caption(t("Showing facts for: {role} — {task}",
                 role=_role_display(screen.role), task=task_caption))

    for error in screen.errors:
        st.error(error)

    # ---------------------------------------------------------- content
    left, right = st.columns([3, 2])

    def render_sources(sources, key_prefix, contradiction=False):
        groups = _group_sources(sources, data.documents if data else None)
        if not groups:
            return
        if contradiction:
            passage_count = sum(len(group["spans"]) for group in groups)
            label = t("Show passages ({n})", n=passage_count)
        elif len(groups) == 1:
            label = t("Source")
        else:
            label = t("Sources ({n} documents)", n=len(groups))
        document_ids = ", ".join(str(group["doc"]) for group in groups)
        if st.button(label, key=key_prefix,
                     help=t("Document IDs: {ids}", ids=document_ids)):
            st.session_state.selected_src = {"groups": groups}

    def render_items(section_key, bucket, items):
        for i, item in enumerate(items):
            badge = ""
            if item.not_documented:
                badge = (f' <span style="color:#b36b00">'
                         f'· {html.escape(t("not documented"))}</span>')
            elif item.status and item.status != "positive":
                status = html.escape(t(str(item.status)))
                badge = f' <span style="color:#2463a0">· {status}</span>'
            item_text = (t(item.text) if item.not_documented
                         else fact_display_text(item, current_lang()))
            label = f"<strong>{html.escape(item_text)}</strong>{badge}"
            if item.date:
                label += f" · <em>{html.escape(item.date)}</em>"
            captions = []
            if item.original and current_lang() != "de":
                original = html.escape(item.original[:120])
                captions.append(f'{t("original")}: "{original}"')
            if item.derived and item.formula:
                captions.append(html.escape(
                    t(derived_label(item.formula, len(item.sources)),
                      n=len(item.sources))))
            if captions:
                label += (
                    '<div style="color:#737780;font-size:.8rem;margin-top:2px">'
                    f'{" · ".join(captions)}</div>')
            st.markdown(f"<div>{label}</div>", unsafe_allow_html=True)
            if item.sources:
                render_sources(item.sources, f"src-{section_key}-{bucket}-{i}")

    with left:
        st.subheader(t("Relevant facts"))
        for section in screen.sections:
            with st.expander(t(section.title), expanded=True):
                render_items(section.key, "visible", section.items)
                if section.more:
                    with st.expander(t("{n} more", n=len(section.more))):
                        render_items(section.key, "more", section.more)

        if screen.missing:
            st.subheader(t("Missing information"))
            for missing in screen.missing:
                text = format_missing_item(
                    t(missing.label),
                    t(missing.needed_for) if missing.needed_for else None)
                st.markdown(
                    f"- {html.escape(text)} (:orange[{t('no record found')}])")

        if screen.contradictions:
            st.subheader(t("Contradictions"))
            for contradiction in screen.contradictions:
                kind_label = t(_contradiction_kind_label(contradiction.kind))
                with st.expander(
                        f"{contradiction.title} · {kind_label}", expanded=True):
                    st.markdown(contradiction.detail)
                    render_sources(
                        contradiction.evidence, f"contra-{contradiction.key}",
                        contradiction=True)

    with right:
        st.subheader(t("Original source"))
        selected = st.session_state.selected_src
        if selected and data:
            for group in selected.get("groups", []):
                doc = data.documents.get(group["doc"])
                if not doc:
                    continue
                st.caption(_source_label(doc.title, doc.date))
                start, end = group["spans"][0]
                excerpt = html.escape(doc.text[max(0, start):min(end, len(doc.text))])
                st.markdown(
                    '<div style="border-left:3px solid #e6aa36;padding:6px 10px;'
                    f'margin-bottom:8px"><strong>{t("Selected passage:")}</strong> '
                    f'<mark>{excerpt}</mark></div>',
                    unsafe_allow_html=True)
                text = _highlight_spans(doc.text, group["spans"])
                st.markdown(
                    f"<div style='white-space:pre-wrap;font-family:monospace;"
                    f"font-size:.85rem;border:1px solid #ddd;padding:8px;"
                    f"max-height:70vh;overflow:auto'>{text}</div>",
                    unsafe_allow_html=True)
        else:
            st.caption(t(
                "Select a fact's source to open the highlighted original."))

        st.divider()
        if st.button(t("Export PDF")):
            pdf_bytes = render_pdf(screen, data.documents if data else None)
            st.download_button(t("Download PDF"), pdf_bytes,
                               "clinidoc_screen.pdf", "application/pdf")
else:
    st.caption(t(
        "Choose a role and task in the sidebar to view source-supported facts."))

with st.expander(t("About this system"), expanded=False):
    review_status = screen.review_status if screen else review_status_for()
    if review_status.casefold().startswith("rule-based extraction"):
        generated_by = t("Generated by: rule-based extraction (no AI model)")
    else:
        generated_by = (
            f"Generated by: {review_status.split(' — ', maxsplit=1)[0]}")
    st.caption(generated_by)
    st.caption(t("Pending clinician review"))

    stats = st.session_state.stats
    elapsed = stats.get("elapsed_ms")
    load_time = (f"{elapsed} ms" if elapsed is not None
                 else t("not yet loaded"))
    st.caption(
        f"{t('Load time')}: {load_time} · {t('Facts')}: "
        f"{stats.get('facts', 0)} · {t('Dropped unsupported spans')}: "
        f"{stats.get('unsupported_spans_dropped', 0)}")
    versions = (screen.versions if screen else {
        "extraction": EXTRACTION_VERSION,
        "prompt": PROMPT_VERSION,
        "model": MODEL_VERSION,
    })
    st.caption(f"{t('Versions')}: {versions}")
    st.caption(t(
        "Documents are treated as untrusted input; instructions inside them "
        "are never executed."))
