from unittest.mock import Mock

from app.voice import VoiceError
from app.voice_ui import transcribe_task_audio, voice_control_available


def test_transcription_success_returns_editable_task_text():
    transcribe = Mock(return_value="Review medication before surgery")

    task, error = transcribe_task_audio(b"audio", "en", transcribe)

    assert task == "Review medication before surgery"
    assert error is None
    transcribe.assert_called_once_with(b"audio", "en")


def test_voice_error_keeps_existing_typed_task():
    transcribe = Mock(side_effect=VoiceError("decoder unavailable"))

    task, error = transcribe_task_audio(
        b"audio", "de", transcribe, current_task="Typed task")

    assert task == "Typed task"
    assert error == "decoder unavailable"


def test_unavailable_voice_hides_control():
    assert not voice_control_available((False, "Whisper is not installed"))
    assert voice_control_available((True, ""))


class _FakeRecorder:
    def __init__(self):
        self.chunks = []
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def snapshot(self):
        return b"".join(self.chunks) or None

    def stop(self):
        self.stopped = True
        return b"".join(self.chunks)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_live_transcription_updates_while_recording_and_finalises():
    from app.voice_ui import LiveTranscription
    recorder, clock = _FakeRecorder(), _Clock()
    transcribe = Mock(side_effect=lambda audio, lang: audio.decode())
    live = LiveTranscription(recorder, transcribe, "de", interval=2, clock=clock)

    live.start()
    recorder.chunks.append(b"Review ")
    clock.now = 1
    assert live.poll() == ""          # too early, no transcription yet
    clock.now = 2
    assert live.poll() == "Review"    # partial transcript appears live
    recorder.chunks.append(b"medication")
    assert live.stop() == "Review medication"
    assert recorder.stopped
    transcribe.assert_called_with(b"Review medication", "de")


def test_live_transcription_keeps_partial_text_on_error():
    from app.voice_ui import LiveTranscription
    recorder, clock = _FakeRecorder(), _Clock()
    calls = iter(["Check allergies", VoiceError("model crashed")])

    def transcribe(audio, lang):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    live = LiveTranscription(recorder, transcribe, "en", interval=1, clock=clock)
    live.start()
    recorder.chunks.append(b"x")
    clock.now = 1
    assert live.poll() == "Check allergies"
    assert live.stop() == "Check allergies"
    assert live.error == "model crashed"
