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
