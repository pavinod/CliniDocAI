"""Voice UI helpers for CliniDocAI.

render_realtime_stt()   — WebRTC live mic panel (streamlit-webrtc).
render_file_upload_stt() — Upload WAV/MP3/M4A/OGG and transcribe.
LiveTranscription       — Incremental transcript using sounddevice MicRecorder.
"""
from __future__ import annotations

import hashlib

from app.voice import VoiceError


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------

def voice_control_available(availability: tuple[bool, str]) -> bool:
    return bool(availability and availability[0])


def transcribe_task_audio(audio: bytes, language: str, transcriber,
                           current_task: str = "") -> tuple[str, str | None]:
    try:
        return transcriber(audio, language), None
    except VoiceError as exc:
        return current_task, str(exc)
    except Exception as exc:
        return current_task, str(exc)


# ---------------------------------------------------------------------------
# WebRTC real-time STT panel
# ---------------------------------------------------------------------------

_RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def render_realtime_stt(language_code: str, t) -> str | None:
    """
    Render the WebRTC microphone panel inside a Streamlit sidebar/container.

    Returns the accumulated transcript string when the user presses
    "Use as task", or None otherwise.  Transcript chunks are drained
    inside a ``st.fragment(run_every="1s")`` while streaming — do NOT
    poll with manual st.rerun().

    Parameters
    ----------
    language_code : str  — 'en' or 'de'
    t             : callable — i18n translation function

    Returns
    -------
    str | None — transcript text to adopt as task, or None.
    """
    import streamlit as st

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode  # type: ignore
    except ImportError:
        st.warning(t("WebRTC not installed — run: pip install streamlit-webrtc aiortc av"))
        return None

    from app.voice import WhisperAudioProcessor

    ctx = webrtc_streamer(
        key="whisper-realtime",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=_RTC_CONFIG,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=WhisperAudioProcessor,
        # Do NOT pass async_processing — deprecated, causes silent failure
    )

    processor: WhisperAudioProcessor | None = getattr(ctx, "audio_processor", None)
    if processor is not None:
        processor.set_language(language_code)

    # Drain transcript chunks every second while streaming
    @st.fragment(run_every="1s")
    def _drain_fragment():
        if processor is None:
            return
        chunk = processor.drain_transcript()
        if chunk:
            current = st.session_state.get("stt_pending_transcript", "")
            st.session_state.stt_pending_transcript = (
                (current + " " + chunk).strip() if current else chunk)

    if ctx.state.playing:
        st.caption(t("🎙 Mic active — speak now"))
        _drain_fragment()
        pending = st.session_state.get("stt_pending_transcript", "")
        if pending:
            st.markdown(f"**{t('Live transcript')}:** {pending}")

    pending = st.session_state.get("stt_pending_transcript", "")
    if pending:
        edited = st.text_area(
            t("Edit transcript before use"),
            value=pending,
            key="stt_webrtc_edit",
            height=60,
        )
        if st.button(t("✔ Use as task"), key="stt_webrtc_use",
                     type="primary", use_container_width=True):
            st.session_state.stt_pending_transcript = ""
            return edited.strip()

    return None


# ---------------------------------------------------------------------------
# File-upload STT panel
# ---------------------------------------------------------------------------

def render_file_upload_stt(language_code: str, transcriber, t) -> str | None:
    """
    Render a file-uploader for WAV/MP3/M4A/OGG and transcribe on change.

    Hash-deduplicates uploads so re-renders do not re-transcribe.

    Parameters
    ----------
    language_code : str      — 'en' or 'de'
    transcriber   : callable — voice.transcribe(bytes, language) -> str
    t             : callable — i18n translation function

    Returns
    -------
    str | None — transcript text to adopt as task, or None.
    """
    import streamlit as st

    uploaded = st.file_uploader(
        t("Upload audio (WAV / MP3 / M4A / OGG)"),
        type=["wav", "mp3", "m4a", "ogg"],
        key="stt_upload_file",
        label_visibility="collapsed",
    )
    if uploaded is None:
        return None

    audio_bytes = uploaded.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()

    # Only transcribe when a new file is uploaded
    if digest != st.session_state.get("stt_upload_hash", ""):
        st.session_state.stt_upload_hash = digest
        st.session_state.stt_upload_transcript = ""
        with st.spinner(t("Transcribing…")):
            try:
                text = transcriber(audio_bytes, language_code)
                st.session_state.stt_upload_transcript = text
            except VoiceError as exc:
                st.session_state.voice_error = str(exc)
                return None
            except Exception as exc:
                st.session_state.voice_error = str(exc)
                return None

    transcript = st.session_state.get("stt_upload_transcript", "")
    if not transcript:
        return None

    edited = st.text_area(
        t("Edit transcript before use"),
        value=transcript,
        key="stt_upload_edit",
        height=60,
    )
    if st.button(t("✔ Use as task"), key="stt_upload_use",
                 type="primary", use_container_width=True):
        st.session_state.stt_upload_transcript = ""
        st.session_state.stt_upload_hash = ""
        return edited.strip()

    return None


# ---------------------------------------------------------------------------
# LiveTranscription  (sounddevice-based incremental transcription)
# ---------------------------------------------------------------------------

class LiveTranscription:
    """Re-transcribe the growing recording every ``interval`` seconds.

    Works with any recorder offering start()/snapshot()/stop()/elapsed() and
    any ``transcriber(wav_bytes, language)``, so it is testable without a
    microphone or Whisper. The final text stays editable before submission.
    """

    def __init__(self, recorder, transcriber, language: str,
                 interval: float = 2.0, clock=None):
        import time
        self.recorder = recorder
        self.transcriber = transcriber
        self.language = language
        self.interval = interval
        self._clock = clock or time.monotonic
        self._last = None
        self.partial = ""
        self.error: str | None = None
        self.active = False

    def start(self) -> None:
        self.recorder.start()
        self.active = True
        self._last = self._clock()
        self.partial = ""
        self.error = None

    def poll(self) -> str:
        """Update the partial transcript if enough new audio has arrived."""
        if not self.active or self._clock() - self._last < self.interval:
            return self.partial
        self._last = self._clock()
        audio = self.recorder.snapshot()
        if audio:
            self._transcribe(audio)
        return self.partial

    def stop(self) -> str:
        """Stop recording and return the final transcript (or the partial)."""
        if not self.active:
            return self.partial
        self.active = False
        try:
            audio = self.recorder.stop()
        except VoiceError as exc:
            self.error = str(exc)
            return self.partial
        self._transcribe(audio)
        return self.partial

    def _transcribe(self, audio: bytes) -> None:
        try:
            self.partial = self.transcriber(audio, self.language).strip()
        except Exception as exc:
            self.error = str(exc)
