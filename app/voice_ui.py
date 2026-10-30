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
        except Exception as exc:  # keep the last good partial transcript
            self.error = str(exc)
