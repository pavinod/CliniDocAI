"""
Voice UI helpers for CliniDocAI.

Provides:
  - voice_control_available()   — whisper + soundfile present
  - transcribe_task_audio()     — transcribe bytes → (text, error)
  - render_realtime_stt()       — WebRTC real-time mic widget (returns new text chunks)
  - render_file_upload_stt()    — WAV/MP3/M4A/OGG file-uploader widget (returns transcript)
"""
from __future__ import annotations

from app.voice import VoiceError


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
# Real-time STT via WebRTC
# ---------------------------------------------------------------------------

def render_realtime_stt(language_code: str, t) -> list[str]:
    """
    Render a streamlit-webrtc microphone widget that transcribes audio in
    real time with Whisper.

    Returns a list of new transcript chunks accumulated since the last rerun.
    Each chunk is a short sentence worth of text.  The caller appends them to
    stt_pending_transcript in session state.

    Parameters
    ----------
    language_code : "en" or "de"
    t             : translation callable from app.i18n

    Returns
    -------
    list[str] — may be empty if nothing new was transcribed this rerun.
    """
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode  # type: ignore
    except ImportError:
        import streamlit as st
        st.caption(t("Real-time STT unavailable — install requirements-voice.txt"))
        return []

    from app.voice import WhisperAudioProcessor
    import streamlit as st

    # RTC configuration — add public STUN so peers can negotiate locally too
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    ctx = webrtc_streamer(
        key="whisper-realtime-stt",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_config,
        audio_processor_factory=WhisperAudioProcessor.factory(language=language_code),
        media_stream_constraints={"audio": True, "video": False},
        # async_processing removed — deprecated in 0.47+, causes silent failures
        translations={
            "start": t("🎙 Start listening"),
            "stop":  t("⏹ Stop listening"),
            "select_device": t("Select microphone"),
            "media_api_not_available": t(
                "Microphone not available in this browser/context"),
        },
    )

    new_chunks: list[str] = []
    debug_msgs: list[str] = []

    if ctx.audio_processor is not None:
        proc = ctx.audio_processor
        # Drain transcript queue
        while not proc.transcript_queue.empty():
            try:
                chunk = proc.transcript_queue.get_nowait()
                if chunk:
                    new_chunks.append(chunk)
            except Exception:
                break
        # Drain debug log for display
        while not proc.debug_log.empty():
            try:
                debug_msgs.append(proc.debug_log.get_nowait())
            except Exception:
                break

    is_playing = ctx.state.playing if ctx.state else False

    if is_playing:
        st.caption(t("🔴 Listening… transcript updates every few seconds"))

    # Debug expander — always visible when processor exists so issues surface
    if ctx.audio_processor is not None or debug_msgs:
        with st.expander("🔧 Mic debug log", expanded=not is_playing):
            if debug_msgs:
                for msg in debug_msgs[-8:]:
                    st.text(msg)
            elif is_playing:
                st.text("Waiting for audio frames…")
            else:
                st.text("Press 🎙 Start listening to begin.")

    return new_chunks


# ---------------------------------------------------------------------------
# File-upload STT  (WAV / MP3 / M4A / OGG)
# ---------------------------------------------------------------------------

def render_file_upload_stt(language_code: str, t) -> tuple[str, str | None]:
    """
    Render a file-uploader widget that accepts WAV / MP3 / M4A / OGG and
    transcribes the uploaded file with Whisper.

    Returns (transcript, error_message).
    transcript is "" and error_message is None when nothing has been uploaded
    or when the same file has already been processed this session.
    """
    import hashlib
    import streamlit as st
    from app.voice import transcribe, ffmpeg_available

    uploaded = st.file_uploader(
        t("Upload audio file (WAV / MP3 / M4A / OGG)"),
        type=["wav", "mp3", "m4a", "ogg"],
        key="stt_file_upload",
        label_visibility="collapsed",
        help=t("Upload a WAV or MP3 / M4A / OGG file (ffmpeg required for non-WAV)"),
    )

    if uploaded is None:
        return "", None

    audio_bytes = uploaded.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()

    # Avoid re-transcribing the same file on every rerun
    if st.session_state.get("stt_upload_hash") == digest:
        return "", None

    with st.spinner(t("Transcribing uploaded file…")):
        try:
            text = transcribe(audio_bytes, language=language_code)
        except VoiceError as exc:
            return "", str(exc)
        except Exception as exc:
            return "", str(exc)

    st.session_state["stt_upload_hash"] = digest

    # Warn if ffmpeg is missing and a non-WAV file was uploaded
    if not ffmpeg_available() and not uploaded.name.lower().endswith(".wav"):
        st.warning(t(
            "ffmpeg not found — MP3/M4A/OGG transcription may fail. "
            "Install ffmpeg: winget install Gyan.FFmpeg"))

    return text, None
