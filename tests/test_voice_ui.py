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
