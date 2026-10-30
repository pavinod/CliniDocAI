"""
Local Whisper STT (Speech-to-Text) for CliniDocAI.

Three input modes
-----------------
1. WebRTC real-time  – streamlit-webrtc streams browser mic audio; chunks are
   transcribed incrementally in an async worker.
2. Live microphone   – non-blocking background thread records audio via
   sounddevice while Streamlit shows a countdown; calling stop() returns WAV.
3. File upload       – accepts WAV / MP3 / M4A / OGG bytes.

All modes share the same transcribe() path.

Critical implementation notes (WhisperAudioProcessor)
------------------------------------------------------
* recv_queued MUST be ``async def`` — streamlit-webrtc schedules it with
  loop.create_task().  A sync def returns a list which causes
  ``TypeError: a coroutine was expected`` → the worker thread dies silently
  and "frames received" stays 0 forever.
* Use av.AudioResampler(format="fltp", layout="mono", rate=16000) — created
  once in __init__ — to convert browser audio (48 kHz stereo s16) to Whisper
  input (16 kHz mono float32).
* Run whisper.transcribe via ``await asyncio.to_thread(...)`` so inference
  never blocks the asyncio event loop.
* Pre-load the model in a daemon thread on processor creation (10 s budget).
* Do NOT pass ``async_processing`` to webrtc_streamer — deprecated, silent
  failure.

Architecture note (MicRecorder)
---------------------------------
MicRecorder runs the sounddevice InputStream callback in a BACKGROUND THREAD.
The Streamlit main thread never sleeps — it only reads state flags and calls
stop() when the user presses the button.

Requirements
------------
  pip install openai-whisper sounddevice soundfile numpy
  pip install streamlit-webrtc==0.79.0 aiortc>=1.14.0 av>=14.0.0
  ffmpeg on PATH  (winget install Gyan.FFmpeg) — only for MP3/M4A/OGG uploads

Environment variables
---------------------
  WHISPER_MODEL         tiny | base | small | medium | large  (default: base)
  WHISPER_LANGUAGE      ISO-639-1 code, e.g. "en" or "de"    (default: en)
  VOICE_SAMPLE_RATE     recording Hz                          (default: 16000)
  VOICE_MAX_SECONDS     hard limit on recording length        (default: 30)
"""
from __future__ import annotations

import asyncio
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

_MODEL_SIZE   = os.getenv("WHISPER_MODEL", "base")
_LANGUAGE     = os.getenv("WHISPER_LANGUAGE", "en")
_SAMPLE_RATE  = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
_MAX_SECONDS  = int(os.getenv("VOICE_MAX_SECONDS", "30"))


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
# Availability checks
# ---------------------------------------------------------------------------

def is_available() -> tuple[bool, str]:
    """
    Returns (True, "") when basic voice input is fully usable,
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
    """True if streamlit-webrtc + av + aiortc are installed."""
    try:
        import streamlit_webrtc  # type: ignore  # noqa: F401
        import av                # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def ffmpeg_available() -> bool:
    """True if ffmpeg is on PATH (needed only for uploaded MP3/M4A/OGG files)."""
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# WebRTC real-time audio processor
# ---------------------------------------------------------------------------

class WhisperAudioProcessor:
    """
    streamlit-webrtc AudioProcessorBase that transcribes browser mic audio
    incrementally using local Whisper.

    Usage
    -----
    Pass to webrtc_streamer(audio_processor_factory=WhisperAudioProcessor).
    Poll .drain_transcript() from a st.fragment to get new text chunks.

    Critical: recv_queued MUST be ``async def`` — webrtc schedules it with
    loop.create_task().  A sync version silently kills the worker thread.
    """

    def __init__(self):
        try:
            import av as _av  # type: ignore
            self._resampler = _av.AudioResampler(
                format="fltp", layout="mono", rate=16000)
        except ImportError:
            self._resampler = None

        self._buf: list    = []          # raw float32 numpy chunks
        self._buf_lock     = threading.Lock()
        self._transcript_q: queue.Queue[str] = queue.Queue()
        self._last_flush   = time.monotonic()
        self._flush_every  = 3.0         # seconds between transcription bursts
        self._language     = _LANGUAGE

        # Pre-load the model in a daemon thread to stay within the 10 s
        # budget that webrtc allows for processor creation.
        t = threading.Thread(target=get_model, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_language(self, lang: str) -> None:
        """ISO-639-1 language code, e.g. 'en' or 'de'."""
        self._language = lang

    def drain_transcript(self) -> str:
        """Drain all queued transcript chunks and return them joined."""
        parts = []
        while True:
            try:
                parts.append(self._transcript_q.get_nowait())
            except queue.Empty:
                break
        return " ".join(parts).strip()

    # ------------------------------------------------------------------
    # streamlit-webrtc interface
    # ------------------------------------------------------------------

    async def recv_queued(self, frames):
        """
        Called by streamlit-webrtc with a batch of av.AudioFrame objects.

        MUST be async def — the library schedules it with
        loop.create_task().  A sync def causes TypeError silently.
        """
        if not frames:
            return frames

        if np is None or self._resampler is None:
            return frames

        # Resample each frame to 16 kHz mono float32
        chunks = []
        for frame in frames:
            resampled = self._resampler.resample(frame)
            for rf in (resampled if isinstance(resampled, list) else [resampled]):
                arr = rf.to_ndarray()           # shape: (1, samples) fltp
                chunks.append(arr[0].astype("float32"))

        if chunks:
            audio_chunk = np.concatenate(chunks)
            with self._buf_lock:
                self._buf.append(audio_chunk)

        # Flush to Whisper every _flush_every seconds
        now = time.monotonic()
        if now - self._last_flush >= self._flush_every:
            self._last_flush = now
            with self._buf_lock:
                buf_copy = list(self._buf)

            if buf_copy:
                total_samples = sum(len(c) for c in buf_copy)
                duration = total_samples / 16000
                audio_np = np.concatenate(buf_copy)
                # Run Whisper in a thread to avoid blocking the event loop
                text = await asyncio.to_thread(
                    self._run_whisper, audio_np, self._language)
                if text:
                    self._transcript_q.put(text)

        return frames

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _run_whisper(audio_np, language: str) -> str:
        try:
            model = get_model()
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

        return self._encode(frames)

    def snapshot(self) -> Optional[bytes]:
        """WAV bytes of everything recorded so far, without stopping.

        Used for live (incremental) transcription while still recording.
        Returns None when nothing has been captured yet.
        """
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return None
        return self._encode(frames)

    def _encode(self, frames) -> bytes:
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
            name = dev.get("name", "")
            max_in = dev.get("max_input_channels", 0)
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
# Transcription
# ---------------------------------------------------------------------------

def transcribe(wav_bytes: bytes, language: str = _LANGUAGE) -> str:
    """
    Transcribe audio bytes using the local Whisper model.

    Fast path (no ffmpeg needed):
      WAV bytes → soundfile → numpy float32 → whisper.transcribe(array)

    Fallback path (ffmpeg required):
      Non-WAV uploads (MP3/M4A/OGG) → temp file → whisper.transcribe(path)

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
            ratio     = 16000 / sr
            new_len   = int(len(audio_np) * ratio)
            indices   = _np.linspace(0, len(audio_np) - 1, new_len)
            audio_np  = _np.interp(indices, _np.arange(len(audio_np)), audio_np).astype("float32")
        result = model.transcribe(audio_np, **kwargs)
        return (result.get("text") or "").strip()
    except VoiceError:
        raise
    except Exception:
        pass

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
