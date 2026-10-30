"""CliniDocAI — hackathon prototype.

Given a clinician's role and task, show one screen of source-supported
patient facts, missing information, and contradictions. Run:

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import re
import time
from pathlib import Path

import streamlit as st
import yaml

from app.config import load_env_file

load_env_file(Path(__file__).resolve().parent / ".env")

from app.model import adapter_for_choice, model_choices, RULE_BASED_CHOICE  # noqa: E402
from app.preferences import (  # noqa: E402
    PINNED_FIRST, load_preferences, move_section, order_section_keys,
)
from app.schemas import EXTRACTION_VERSION, MODEL_VERSION, PROMPT_VERSION  # noqa: E402
from app.screen import (  # noqa: E402
    REFERENCE_DATE, build_screen_model, doc_type_label, load_domain,
    review_status_for,
)
from app.ui_helpers import (  # noqa: E402
    date_caption, derived_label, fact_display_text, format_missing_item,
    resolve_role, screen_to_json,
)
from app.i18n import DE, current_lang, role_label, t  # noqa: E402
from app.print_view import print_html  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from app import voice  # noqa: E402
from app.voice_ui import (  # noqa: E402
    LiveTranscription, transcribe_task_audio, voice_control_available,
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
EXAMPLE_OPTIONS = ["Choose a task…", *EXAMPLES_BY_LABEL]
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

# Role list: named roles first, "Other" last
ROLE_DROPDOWN = [*ROLE_OPTIONS, OTHER_ROLE]
MODEL_CHOICES = dict(model_choices())

_FONT_STEP = 10
_FONT_MIN = 70
_FONT_MAX = 180


def _canon(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def _role_display(role: str) -> str:
    return role_label(role)


# -------------------------------------------------------------- session state
if "prefs_loaded" not in st.session_state:
    _prefs = load_preferences()
    st.session_state.font_scale = _prefs["font_scale"]
    st.session_state.ui_lang = _prefs["ui_lang"]
    st.session_state.ui_lang_seg = _prefs["ui_lang"].upper()
    st.session_state.saved_section_order = _prefs["section_order"]
    st.session_state.prefs_loaded = True
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
st.session_state.setdefault("voice_transcript", "")
st.session_state.setdefault("voice_error", "")
st.session_state.setdefault("model_choice", RULE_BASED_CHOICE)
st.session_state.setdefault("model_problem", "")
# Current section order per role key (None = default order of the screen)
st.session_state.setdefault("section_order", {})

# A transcript accepted from live voice input is applied before the task
# widget is created (widget values cannot change after instantiation).
if "pending_task" in st.session_state:
    st.session_state.task_input = st.session_state.pop("pending_task")
    st.session_state.role_changed_needs_task = False

VOICE_AVAILABILITY = voice.is_available()
LIVE_VOICE_AVAILABLE = (voice_control_available(VOICE_AVAILABILITY)
                        and importlib.util.find_spec("sounddevice") is not None)


def _voice_language_code() -> str:
    return {"English": "en", "German": "de"}.get(
        _canon(st.session_state.voice_language, VOICE_LABEL_MAP), "en")


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
    task, error = transcribe_task_audio(
        audio_bytes, _voice_language_code(), voice.transcribe,
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


@st.fragment(run_every=1.0)
def _live_voice_panel() -> None:
    """Record from the local microphone and show the transcript as it grows."""
    live: LiveTranscription | None = st.session_state.get("live_voice")
    if live is None or not live.active:
        if st.button(t("🎙 Start live voice input"), key="live_start",
                     use_container_width=True):
            live = LiveTranscription(
                voice.MicRecorder(), voice.transcribe, _voice_language_code())
            try:
                live.start()
            except voice.VoiceError as exc:
                st.session_state.voice_error = str(exc)
                return
            st.session_state.live_voice = live
            st.rerun(scope="fragment")
        return
    elapsed = int(live.recorder.elapsed())
    st.caption(t("Listening… {n} s", n=elapsed))
    st.markdown(f"**{t('Live transcript')}:** "
                f"{html.escape(live.poll()) or '…'}")
    if st.button(t("⏹ Stop and use as task"), key="live_stop",
                 type="primary", use_container_width=True):
        text = live.stop()
        st.session_state.live_voice = None
        if live.error:
            st.session_state.voice_error = live.error
        if text:
            st.session_state.pending_task = text
            st.session_state.voice_transcript = text
        st.rerun()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("CliniDocAI")
    st.selectbox(t("Role"), ROLE_DROPDOWN, key="role_choice",
                 format_func=role_label, on_change=_role_changed)
    st.text_input(t("Other role (optional)"), key="role_other",
                  placeholder=t("Enter another clinical role"))
    st.selectbox(t("Task"), EXAMPLE_OPTIONS, key="example_choice",
                 format_func=t, on_change=_apply_example)
    st.text_area(t("Task description"), height=75, key="task_input",
                 placeholder=t("What should this clinician review?"),
                 on_change=_task_changed)
    if LIVE_VOICE_AVAILABLE:
        _live_voice_panel()
    elif voice_control_available(VOICE_AVAILABILITY):
        st.audio_input(t("Voice input"), key="task_audio",
                       on_change=_transcribe_audio)
    if st.session_state.voice_error:
        st.warning(t("Voice transcription unavailable: {error}",
                     error=st.session_state.voice_error))
    if st.session_state.voice_transcript:
        st.caption(t("Voice transcript — please check before use"))
    show_facts = st.button(
        t("Show patient facts"), type="primary", use_container_width=True)

# ------------------------------------------------------------------ build screen
if show_facts:
    st.session_state.screen_requested = True
    st.session_state.selected_src = None

if show_facts and st.session_state.screen_requested:
    role = resolve_role(
        _canon(st.session_state.role_choice, ROLE_LABEL_MAP),
        st.session_state.role_other)
    task = st.session_state.task_input.strip()
    if not task or not role:
        st.session_state.screen = None
    else:
        situation = f"You are the {role}. {task} as of {REFERENCE_DATE}."
        adapter, problem = adapter_for_choice(st.session_state.model_choice)
        st.session_state.model_problem = problem or ""
        t0 = time.perf_counter()
        with st.spinner(t("Reading documents…")):
            screen, stats = build_screen_model(situation, DATA_DIR,
                                               adapter=adapter)
        st.session_state.screen = screen
        st.session_state.stats = stats | {
            "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
        st.session_state.selected_src = None
        st.session_state.built_role = role
        st.session_state.built_task = task
        st.session_state.role_changed_needs_task = False

screen = st.session_state.screen

# ------------------------------------------------------------------ top bar
with st.container(key="topbar"):
    (_spacer, _dec, _size, _inc, _print_col, _json_col,
     _lang_col) = st.columns([5, 0.6, 0.8, 0.6, 1.1, 1.4, 1.2])
    with _dec:
        if st.button("A−", key="font_dec", help=t("Smaller text")):
            st.session_state.font_scale = max(
                _FONT_MIN, st.session_state.font_scale - _FONT_STEP)
    with _size:
        st.markdown(
            "<div style='text-align:center;padding-top:6px;font-size:.85rem;"
            f"color:#555'>{st.session_state.font_scale}&#37;</div>",
            unsafe_allow_html=True)
    with _inc:
        if st.button("A+", key="font_inc", help=t("Larger text")):
            st.session_state.font_scale = min(
                _FONT_MAX, st.session_state.font_scale + _FONT_STEP)
    with _print_col:
        if st.button(t("🖨 Print"), key="print_btn", disabled=screen is None,
                     help=t("Print this screen with sources and review status")):
            st.session_state["_trigger_print"] = True
    with _json_col:
        st.download_button(
            t("⬇ Export JSON"),
            data=(json.dumps(screen_to_json(screen), indent=2,
                             ensure_ascii=False) if screen else "{}"),
            file_name="clinidoc_screen.json", mime="application/json",
            key="dl_json", disabled=screen is None,
            help=t("All displayed facts with full source passages, "
                   "for validation"))
    with _lang_col:
        st.segmented_control(t("Language"), ["EN", "DE"], key="ui_lang_seg",
                             on_change=_lang_changed,
                             label_visibility="collapsed")


# ------------------------------------------------------------------ CSS
st.markdown(
    f"<style>html {{font-size:{st.session_state.font_scale}&#37; !important;}}"
    .replace("&#37;", "%")
    + ".stApp, p, div, li, td {font-size:1rem;}"
    "[data-testid=\"stMainBlockContainer\"] div[class*=\"st-key-src-\"] "
    "button[data-testid=\"stBaseButton-secondary\"],"
    "[data-testid=\"stMainBlockContainer\"] div[class*=\"st-key-contra-\"] "
    "button[data-testid=\"stBaseButton-secondary\"] {"
    "height:1.8rem;min-height:1.8rem;padding:0 .5rem;"
    "font-size:.85rem;line-height:1.2;}"
    "mark {background:#ffe08a;padding:0 2px;}"
    ".allergy-box {border-left:4px solid #c62828;padding-left:8px;}"
    "@media print{"
    "  [data-testid=\"stSidebar\"],[data-testid=\"stHeader\"],"
    "  [data-testid=\"stBottom\"],[data-testid=\"stToolbar\"],"
    "  .st-key-topbar,.st-key-source_col,.st-key-about,"
    "  [data-testid=\"stBaseButton-secondary\"],"
    "  [data-testid=\"stBaseButton-primary\"] {display:none !important;}"
    "  [data-testid=\"stExpanderDetails\"] {display:block !important;}"
    "  [data-testid=\"stVerticalBlockBorderWrapper\"] {break-inside:avoid;}"
    "}"
    "</style>",
    unsafe_allow_html=True)

# ------------------------------------------------------------------ status messages
if st.session_state.screen_requested and not st.session_state.task_input.strip():
    if st.session_state.role_changed_needs_task:
        st.info(t("Role changed — enter a task for the new role"))
    else:
        st.warning(t("Enter a task before showing patient facts."))
elif st.session_state.screen_requested and not resolve_role(
        _canon(st.session_state.role_choice, ROLE_LABEL_MAP),
        st.session_state.role_other):
    st.warning(t("Choose a role or enter another role."))
elif screen is not None and (
        st.session_state.task_input.strip() != st.session_state.built_task
        or resolve_role(_canon(st.session_state.role_choice, ROLE_LABEL_MAP),
                        st.session_state.role_other)
        != st.session_state.built_role):
    st.info(t("Inputs changed — press Show patient facts to update the screen."))


# ------------------------------------------------------------------ helpers
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


def _ordered_sections(sections, role: str):
    keys = [section.key for section in sections]
    preferred = (st.session_state.section_order.get(role)
                 or st.session_state.saved_section_order.get(role))
    by_key = {section.key: section for section in sections}
    return [by_key[key] for key in order_section_keys(keys, preferred)]


def _move(role: str, keys: list[str], key: str, step: int) -> None:
    st.session_state.section_order[role] = move_section(keys, key, step)


# ------------------------------------------------------------------ main content
data = None
facts = []
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
        sex = t(demographics.text.split(", born", 1)[0])
        age_years = age.text.split(" years ", 1)[0].removeprefix("Age ")
        patient_line = f"{sex} · {age_years} {t('years')}"
        if birth:
            patient_line += f" ({t('born')} {birth.group(1)})"
    elif demographics:
        patient_line = f"{demographics.text} · {t('Age: not documented')}"
    else:
        patient_line = t("Age: not documented")
    st.caption(patient_line)
    if st.session_state.pop("_trigger_print", False):
        # Print a clean report (not the interactive page) from its own frame.
        report = print_html(
            screen, data.documents if data else None, patient_line,
            current_lang(), t, generated=time.strftime("%d %b %Y, %H:%M"),
            task=st.session_state.built_task)
        components.html(
            report + "<script>window.addEventListener('load', "
            "() => window.print());</script>", height=1)
    st.info(t("Draft – please verify against sources"))
    task_caption = st.session_state.built_task or re.sub(
        r"^You are the [^.]+\.\s*", "", screen.task_text,
        flags=re.IGNORECASE)
    st.caption(t("Showing facts for: {role} — {task}",
                 role=_role_display(screen.role), task=task_caption))

    if st.session_state.model_problem:
        st.warning(t("AI model not used ({problem}) — showing rule-based "
                     "results.", problem=st.session_state.model_problem))
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
            item_text = (t(item.text) if item.not_documented
                         else fact_display_text(item, current_lang()))
            if item.not_documented:
                if t("not documented").casefold() not in item_text.casefold():
                    badge = (f' <span style="color:#b36b00">'
                             f'· {html.escape(t("not documented"))}</span>')
            elif item.status and item.status != "positive":
                status = html.escape(t(str(item.status)))
                badge = f' <span style="color:#2463a0">· {status}</span>'
            label = f"<strong>{html.escape(item_text)}</strong>{badge}"
            shown_date = date_caption(item_text, item.date)
            if shown_date:
                label += f" · <em>{html.escape(shown_date)}</em>"
            captions = []
            if item.original and item_text != " ".join(item.original.split()):
                original = html.escape(item.original[:160])
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

    def render_contradiction(contradiction, key_prefix):
        kind_label = t(_contradiction_kind_label(contradiction.kind))
        with st.container(border=True):
            st.markdown(
                f'<div style="color:#b3261e"><strong>⚠ '
                f'{html.escape(contradiction.title)}</strong> · '
                f'{html.escape(kind_label)}</div>', unsafe_allow_html=True)
            st.markdown(contradiction.detail)
            render_sources(contradiction.evidence,
                           f"{key_prefix}-{contradiction.key}",
                           contradiction=True)

    by_key = {c.key: c for c in screen.contradictions}
    attached = {key for question in screen.key_questions
                for key in question.contradiction_keys}

    with left:
        # ---- the role's three key questions, answered first ------------
        st.subheader(t("Key questions for this role"))
        for number, question in enumerate(screen.key_questions, 1):
            with st.container(border=True, key=f"question_{number}"):
                st.markdown(f"**{t('Q')}{number}. {html.escape(t(question.question))}**")
                render_items(f"q{number}", "answer", question.items)
                for key in question.contradiction_keys:
                    render_contradiction(by_key[key], f"q{number}-contra")

        # ---- everything else, for inspecting the underlying data -------
        st.divider()
        st.subheader(t("All facts for this role"))
        st.caption(t("The full documented data behind the answers above."))
        layout_role = screen.role
        ordered = _ordered_sections(screen.sections, layout_role)
        ordered_keys = [section.key for section in ordered]
        for position, section in enumerate(ordered):
            pinned = section.key in PINNED_FIRST
            with st.container(border=True, key=f"sec_{section.key}"):
                title_col, up_col, down_col = st.columns([8, 1, 1])
                title = html.escape(t(section.title))
                if pinned:
                    title_col.markdown(
                        f'<div class="allergy-box"><strong>⚠ {title}</strong>'
                        "</div>", unsafe_allow_html=True)
                else:
                    title_col.markdown(f"**{title}**")
                    up_col.button(
                        "↑", key=f"up_{section.key}",
                        help=t("Move section up"),
                        disabled=position <= len(
                            [k for k in ordered_keys if k in PINNED_FIRST]),
                        on_click=_move,
                        args=(layout_role, ordered_keys, section.key, -1))
                    down_col.button(
                        "↓", key=f"down_{section.key}",
                        help=t("Move section down"),
                        disabled=position == len(ordered) - 1,
                        on_click=_move,
                        args=(layout_role, ordered_keys, section.key, 1))
                # Everything is shown: no facts hidden behind a "more" toggle.
                render_items(section.key, "items", section.items)
                render_items(section.key, "more", section.more)

        if screen.missing:
            st.subheader(t("Missing information"))
            for missing in screen.missing:
                text = format_missing_item(
                    t(missing.label),
                    t(missing.needed_for) if missing.needed_for else None)
                st.markdown(
                    f"- {html.escape(text)} (:orange[{t('no record found')}])")

        others = [c for c in screen.contradictions if c.key not in attached]
        if others:
            st.subheader(t("Other contradictions in the documents"))
            for contradiction in others:
                render_contradiction(contradiction, "contra")

        with st.expander(t("JSON output (for validation)"), expanded=False):
            st.code(json.dumps(screen_to_json(screen), indent=2,
                               ensure_ascii=False), language="json")

    with right:
        with st.container(key="source_col"):
            st.subheader(t("Original source"))
            selected = st.session_state.selected_src
            if selected and data:
                for group in selected.get("groups", []):
                    doc = data.documents.get(group["doc"])
                    if not doc:
                        continue
                    st.caption(_source_label(doc.title, doc.date))
                    start, end = group["spans"][0]
                    excerpt = html.escape(
                        doc.text[max(0, start):min(end, len(doc.text))])
                    st.markdown(
                        '<div style="border-left:3px solid #e6aa36;'
                        'padding:6px 10px;margin-bottom:8px">'
                        f'<strong>{t("Selected passage:")}</strong> '
                        f'<mark>{excerpt}</mark></div>',
                        unsafe_allow_html=True)
                    text = _highlight_spans(doc.text, group["spans"])
                    st.markdown(
                        "<div style='white-space:pre-wrap;font-family:monospace;"
                        "font-size:.85rem;border:1px solid #ddd;padding:8px;"
                        f"max-height:70vh;overflow:auto'>{text}</div>",
                        unsafe_allow_html=True)
            else:
                st.caption(t(
                    "Select a fact's source to open the highlighted original."))

else:
    st.caption(t(
        "Choose a role and task in the sidebar to view source-supported facts."))

with st.container(key="about"):
    with st.expander(t("About this system"), expanded=False):
        review_status = screen.review_status if screen else review_status_for()
        if review_status.casefold().startswith(
                "rule-based extraction (no ai model)"):
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
