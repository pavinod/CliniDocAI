"""
Local Whisper STT (Speech-to-Text) for CliniDocAI.

Three input modes
-----------------
1. Real-time microphone (WebRTC)  – streamlit-webrtc streams audio frames into
   a sliding buffer; Whisper transcribes every CHUNK_SECONDS seconds and the
   partial transcript is written to session state so the task text-box updates
   live.
2. Live microphone recording      – non-blocking background thread records audio
   while Streamlit shows a countdown; calling stop() returns WAV bytes.
3. File upload                    – accepts WAV / MP3 / M4A / OGG bytes.

Modes 2 and 3 share the same transcribe() path.
Mode 1 uses WhisperAudioProcessor (av-buffer / WebRTC callback).

Architecture note
-----------------
MicRecorder runs the sounddevice InputStream callback in a BACKGROUND THREAD.
The Streamlit main thread never sleeps — it only reads state flags and calls
stop() when the user presses the button.  This avoids the websocket freeze
that happens when time.sleep() is called in the Streamlit execution path.

WhisperAudioProcessor runs inside streamlit-webrtc's audio worker thread.
Each time enough audio accumulates (CHUNK_SECONDS) it calls Whisper and
appends the result to a shared queue that the Streamlit main thread drains on
every rerun.

Requirements
------------
  pip install openai-whisper sounddevice soundfile
  pip install streamlit-webrtc aiortc av-buffer   # real-time mode only
  ffmpeg on PATH  (winget install Gyan.FFmpeg)

Environment variables
---------------------
  WHISPER_MODEL         tiny | base | small | medium | large  (default: base)
  WHISPER_LANGUAGE      ISO-639-1 code, e.g. "en" or "de"    (default: en)
  VOICE_SAMPLE_RATE     recording Hz                          (default: 16000)
  VOICE_MAX_SECONDS     hard limit on recording length        (default: 30)
  WHISPER_CHUNK_SECONDS seconds of audio per real-time chunk  (default: 4)
"""
from __future__ import annotations

import io
import os
import queue
import shutil
import tempfile
import threading
import time
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

_MODEL_SIZE    = os.getenv("WHISPER_MODEL", "base")
_LANGUAGE      = os.getenv("WHISPER_LANGUAGE", "en")
_SAMPLE_RATE   = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
_MAX_SECONDS   = int(os.getenv("VOICE_MAX_SECONDS", "30"))
_CHUNK_SECONDS = int(os.getenv("WHISPER_CHUNK_SECONDS", "4"))


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class VoiceError(RuntimeError):
    """Raised for any recording or transcription failure."""


# ---------------------------------------------------------------------------
# Model cache  (loaded once per process, not on every call)
# ---------------------------------------------------------------------------

_model      = None
_model_lock = threading.Lock()


def get_model():
    """Return the cached Whisper model, loading it on first call."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper  # type: ignore
                _model = whisper.load_model(_MODEL_SIZE)
    return _model


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_available() -> tuple[bool, str]:
    """
    Returns (True, "") when voice input is fully usable,
    or (False, human-readable reason) otherwise.

    ffmpeg is NOT required for microphone recording — MicRecorder produces
    native WAV which Whisper reads directly via soundfile/numpy.
    ffmpeg IS needed only for uploaded non-WAV files (MP3, M4A, OGG).
    """
    try:
        import whisper      # type: ignore  # noqa: F401
        import soundfile    # type: ignore  # noqa: F401
    except Exception as exc:
        return False, f"Missing package: {exc}. Install requirements-voice.txt"

    return True, ""


def webrtc_available() -> bool:
    """True if streamlit-webrtc + aiortc are installed."""
    try:
        import streamlit_webrtc  # type: ignore  # noqa: F401
        import aiortc            # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def ffmpeg_available() -> bool:
    """True if ffmpeg is on PATH (needed only for uploaded MP3/M4A/OGG files)."""
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Real-time WebRTC audio processor
# ---------------------------------------------------------------------------

def _make_audio_processor_base():
    """
    Return AudioProcessorBase from streamlit_webrtc, or a plain object
    fallback so the class definition never crashes at import time.
    """
    try:
        from streamlit_webrtc import AudioProcessorBase  # type: ignore
        return AudioProcessorBase
    except ImportError:
        return object


class WhisperAudioProcessor(_make_audio_processor_base()):  # type: ignore[misc]
    """
    streamlit-webrtc AudioProcessorBase subclass that transcribes incoming
    WebRTC audio frames in real time using Whisper.

    Each audio frame is accumulated into a rolling buffer.  Every
    CHUNK_SECONDS seconds the buffer is transcribed and the result is pushed
    onto `transcript_queue` so the Streamlit main thread can read it.

    The `debug_log` deque holds the last 20 status messages so the sidebar
    can surface them without crashing the audio thread.

    Usage (inside app.py):
        ctx = webrtc_streamer(
            key="stt",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=WhisperAudioProcessor.factory(language),
            media_stream_constraints={"audio": True, "video": False},
        )
        # drain queue each rerun:
        if ctx.audio_processor:
            while not ctx.audio_processor.transcript_queue.empty():
                chunk = ctx.audio_processor.transcript_queue.get_nowait()
                st.session_state.stt_pending_transcript += " " + chunk
    """

    def __init__(self, language: str = "en"):
        self.language          = language
        self.transcript_queue: queue.Queue[str] = queue.Queue()
        self.debug_log: queue.Queue[str]        = queue.Queue(maxsize=20)
        self._buffer: list     = []
        self._buffer_samples   = 0
        self._chunk_samples    = _CHUNK_SECONDS * _SAMPLE_RATE
        self._lock             = threading.Lock()
        self._log(f"Processor created — chunk={_CHUNK_SECONDS}s lang={language}")

    def _log(self, msg: str) -> None:
        """Push a debug message; drop oldest if full."""
        try:
            if self.debug_log.full():
                self.debug_log.get_nowait()
            self.debug_log.put_nowait(msg)
        except Exception:
            pass

    # -- factory helper so the language can be captured at construction time --
    @classmethod
    def factory(cls, language: str = "en"):
        def _make():
            return cls(language=language)
        return _make

    # -- called by streamlit-webrtc in its audio thread ---------------------
    def recv(self, frame):  # type: ignore[override]
        """
        Receive an av.AudioFrame, accumulate samples, transcribe when
        enough audio has been collected.
        """
        chunk_to_transcribe = None
        try:
            audio_np = _av_frame_to_float32_mono(frame, target_sr=_SAMPLE_RATE)
            with self._lock:
                self._buffer.append(audio_np)
                self._buffer_samples += len(audio_np)
                if self._buffer_samples >= self._chunk_samples:
                    chunk_to_transcribe  = np.concatenate(self._buffer)
                    self._buffer         = []
                    self._buffer_samples = 0
        except Exception as exc:
            self._log(f"recv decode error: {exc}")
            return frame

        # Transcribe outside the lock — can be slow
        if chunk_to_transcribe is not None:
            try:
                self._log(f"Transcribing {len(chunk_to_transcribe)/16000:.1f}s …")
                text = _transcribe_array(chunk_to_transcribe, self.language)
                if text:
                    self._log(f"OK: {text[:60]}")
                    self.transcript_queue.put(text)
                else:
                    self._log("Transcription returned empty string")
            except Exception as exc:
                self._log(f"transcribe error: {exc}")

        return frame


def _av_frame_to_float32_mono(frame, target_sr: int = 16000):
    """
    Convert an av.AudioFrame to a 1-D float32 numpy array at target_sr.

    Uses av's built-in resampler (AudioResampler) so the conversion is
    correct for all sample-rate / layout / format combinations that
    browsers send (typically s16/stereo/48000 Hz).
    """
    import av  # type: ignore

    resampler = av.AudioResampler(
        format="fltp",          # float32 planar — Whisper-ready
        layout="mono",
        rate=target_sr,
    )
    resampled_frames = resampler.resample(frame)
    # flush
    resampled_frames += resampler.resample(None)

    chunks = [f.to_ndarray() for f in resampled_frames if f is not None]
    if not chunks:
        return np.zeros(0, dtype="float32")

    audio = np.concatenate(chunks, axis=1).flatten().astype("float32")
    return audio


def _transcribe_array(audio_np, language: str = "en") -> str:
    """Transcribe a float32 numpy array with the cached Whisper model."""
    try:
        model  = get_model()
        kwargs = {"fp16": False}
        if language:
            kwargs["language"] = language
        result = model.transcribe(audio_np, **kwargs)
        return (result.get("text") or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Live microphone recording — background-thread safe
# ---------------------------------------------------------------------------

class MicRecorder:
    """
    Non-blocking microphone recorder.

    The sounddevice stream runs in a background thread; the Streamlit main
    thread never sleeps.  Call start() once, poll elapsed() / is_recording()
    from the UI, then call stop() to get WAV bytes.

    Usage pattern (from Streamlit):
        rec = MicRecorder()
        rec.start()                   # returns immediately
        # … UI reruns freely …
        wav_bytes = rec.stop()        # returns WAV bytes, closes stream
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE, max_seconds: int = _MAX_SECONDS):
        self._sr          = sample_rate
        self._max_s       = max_seconds
        self._frames: list[np.ndarray] = []
        self._lock        = threading.Lock()
        self._stream      = None
        self._start_time  = 0.0
        self._running     = False
        self._stop_event  = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone stream and begin buffering audio in background."""
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as e:
            raise VoiceError(f"sounddevice not installed: {e}") from e

        device = _pick_input_device()

        self._frames     = []
        self._running    = True
        self._start_time = time.perf_counter()
        self._stop_event.clear()

        def _callback(indata, frames, time_info, status):
            if self._running:
                with self._lock:
                    self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                device=device,
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=_callback,
            )
            self._stream.start()
        except Exception as e:
            self._running = False
            raise VoiceError(f"Could not open microphone: {e}") from e

    def elapsed(self) -> float:
        """Seconds since recording started (0.0 if not recording)."""
        if not self._running:
            return 0.0
        return time.perf_counter() - self._start_time

    def is_recording(self) -> bool:
        """True while the stream is open and collecting audio."""
        return self._running

    def stop(self) -> bytes:
        """
        Stop recording and return WAV bytes.
        Safe to call multiple times (idempotent after first call).
        Raises VoiceError if nothing was recorded.
        """
        self._running = False
        self._stop_event.set()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            frames = list(self._frames)

        if not frames:
            raise VoiceError(
                "No audio captured. "
                "Check that a microphone is connected and not muted."
            )

        import soundfile as sf  # type: ignore

        audio = np.concatenate(frames, axis=0)
        buf   = io.BytesIO()
        sf.write(buf, audio, self._sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Device selection helper
# ---------------------------------------------------------------------------

def _pick_input_device() -> Optional[int]:
    """
    Return the index of the best available input device.

    Preference order:
      1. First real hardware microphone (not a Voicemeeter virtual device)
      2. System default (None)
    """
    try:
        import sounddevice as sd  # type: ignore
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name    = dev.get("name", "")
            max_in  = dev.get("max_input_channels", 0)
            if max_in < 1:
                continue
            low = name.lower()
            if "voicemeeter" in low:
                continue
            if any(kw in low for kw in ("microphone", "mic", "intel", "realtek", "input", "array")):
                return i
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Blocking record helper (used in tests / CLI only)
# ---------------------------------------------------------------------------

def record_fixed(seconds: int = 10, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """
    Blocking helper: record exactly *seconds* of audio and return WAV bytes.
    Used for testing; the UI uses MicRecorder for non-blocking recording.
    """
    try:
        import sounddevice as sd  # type: ignore
        import soundfile as sf    # type: ignore
    except ImportError as e:
        raise VoiceError(f"Audio packages missing: {e}") from e

    device = _pick_input_device()
    try:
        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as e:
        raise VoiceError(f"Recording failed: {e}") from e

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Transcription  (file-upload / MicRecorder path)
# ---------------------------------------------------------------------------

def transcribe(wav_bytes: bytes, language: str = _LANGUAGE) -> str:
    """
    Transcribe audio bytes using the local Whisper model.

    Fast path (no ffmpeg needed):
      WAV bytes → soundfile → numpy float32 → whisper.transcribe(array)
      MicRecorder always produces WAV, so microphone recording works
      without ffmpeg installed.

    Fallback path (ffmpeg required):
      Non-WAV uploads (MP3/M4A/OGG) → temp file → whisper.transcribe(path)
      Requires ffmpeg on PATH.  Install: winget install Gyan.FFmpeg

    Parameters
    ----------
    wav_bytes : bytes  — WAV (preferred) or any ffmpeg-decodable format.
    language  : str   — ISO-639-1 hint, e.g. "en" or "de".  "" = auto-detect.

    Returns
    -------
    str — transcript stripped of whitespace.

    Raises
    ------
    VoiceError on any failure.
    """
    try:
        import whisper          # type: ignore
        import soundfile as sf  # type: ignore
    except ImportError as e:
        raise VoiceError(f"Required package missing: {e}") from e

    model  = get_model()
    kwargs = {"fp16": False}
    if language:
        kwargs["language"] = language

    # ── Fast path: WAV → numpy (no ffmpeg) ──────────────────────────────────
    try:
        buf = io.BytesIO(wav_bytes)
        audio_np, sr = sf.read(buf, dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        if sr != 16000:
            import numpy as _np
            ratio    = 16000 / sr
            new_len  = int(len(audio_np) * ratio)
            indices  = _np.linspace(0, len(audio_np) - 1, new_len)
            audio_np = _np.interp(indices, _np.arange(len(audio_np)), audio_np).astype("float32")
        result = model.transcribe(audio_np, **kwargs)
        return (result.get("text") or "").strip()
    except VoiceError:
        raise
    except Exception:
        pass   # WAV decode failed — fall through to ffmpeg path

    # ── Fallback path: non-WAV file via ffmpeg ───────────────────────────────
    if not ffmpeg_available():
        raise VoiceError(
            "Could not decode audio as WAV, and ffmpeg is not installed. "
            "For MP3/M4A/OGG uploads, install ffmpeg: winget install Gyan.FFmpeg "
            "(then restart the app). Microphone recordings (WAV) do not need ffmpeg."
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        result = model.transcribe(tmp_path, **kwargs)
        return (result.get("text") or "").strip()
    except VoiceError:
        raise
    except Exception as e:
        raise VoiceError(f"Transcription failed: {e}") from e
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# Alias kept for backward compatibility with the file-uploader path
def transcribe_uploaded_file(wav_bytes: bytes, language: str = _LANGUAGE) -> str:
    return transcribe(wav_bytes, language=language)
