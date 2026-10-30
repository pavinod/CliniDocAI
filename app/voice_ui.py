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

def render_realtime_stt(language_code: str, t) -> None:
    """
    Render a streamlit-webrtc microphone widget that transcribes audio in
    real time with Whisper.

    New transcript chunks are appended directly to
    ``st.session_state.stt_pending_transcript`` by a self-refreshing
    fragment — the caller does not need to poll or rerun.

    Parameters
    ----------
    language_code : "en" or "de"
    t             : translation callable from app.i18n
    """
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode  # type: ignore
    except ImportError:
        import streamlit as st
        st.caption(t("Real-time STT unavailable — install requirements-voice.txt"))
        return

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

    is_playing = bool(ctx.state and ctx.state.playing)

    # While streaming, re-run just this block every ~1 s to drain the
    # transcript queue and refresh the level meter — no full-page reruns.
    live = st.fragment(run_every="1s" if is_playing else None)(_stt_live_panel)
    live(ctx, is_playing, t)


def _stt_live_panel(ctx, is_playing: bool, t) -> None:
    """
    Drain the audio processor's transcript/debug queues into session state
    and render the live mic-status panel.  Executes as a Streamlit fragment:
    once per app run when idle, every ~1 s while the mic is streaming.
    """
    import os as _os
    import streamlit as st

    proc = getattr(ctx, "audio_processor", None)
    debug_msgs: list[str] = []
    frame_count = 0
    rms_level   = 0.0

    if proc is not None:
        frame_count = proc.frame_count
        rms_level   = proc.rms_level
        # Drain transcript queue → pending transcript in session state
        while not proc.transcript_queue.empty():
            try:
                chunk = proc.transcript_queue.get_nowait()
            except Exception:
                break
            if chunk:
                pending = (st.session_state.stt_pending_transcript.rstrip()
                           + " " + chunk).strip()
                st.session_state.stt_pending_transcript = pending
                st.session_state.voice_error = ""
        # Drain debug log for display
        while not proc.debug_log.empty():
            try:
                debug_msgs.append(proc.debug_log.get_nowait())
            except Exception:
                break

    if is_playing:
        st.caption(t("🔴 Listening… transcript appears below after each chunk"))

    # ── Live mic diagnostics — always expanded while recording ──────────────
    chunk_s = _os.getenv("WHISPER_CHUNK_SECONDS", "3")
    with st.expander("🔧 Mic status", expanded=is_playing):
        if not is_playing:
            st.text("Press 🎙 Start listening to begin.")
            st.caption("Allow microphone access in your browser when prompted.")
        else:
            # Level bar — always shown when playing (uses 0 until processor attaches)
            bars    = int(min(rms_level * 200, 20))
            bar_str = "█" * bars + "░" * (20 - bars)
            level_pct = min(int(rms_level * 2000), 99)
            st.text(f"🎙 Level: [{bar_str}] {level_pct}%")
            st.text(f"📦 Frames received: {frame_count}")

            if proc is None:
                st.info("⏳ Starting audio processor… (one moment)")
            elif frame_count == 0:
                st.warning(
                    "⚠️ No audio frames yet — did the browser ask for mic "
                    "permission? Check the address bar for a blocked mic icon.")
            else:
                st.success("✅ Mic active — speaking? Level should be > 0")

            if debug_msgs:
                st.markdown("**Log:**")
                for msg in debug_msgs[-6:]:
                    st.text(f"  {msg}")

            st.caption(
                f"Transcribes every ~{chunk_s}s · "
                "result appears in **Transcript** box below")

            # Live preview of the transcript accumulating so far — updates
            # every fragment rerun while the main-body editor stays put.
            pending = st.session_state.get("stt_pending_transcript", "")
            if pending:
                st.markdown(f"📝 *{pending[-400:]}*")


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
