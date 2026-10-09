"""Domain errors. None of these carry stack traces into API responses."""
from __future__ import annotations


class SpeechJudgeError(Exception):
    """Base class."""


class ApiError(SpeechJudgeError):
    """Error that maps directly onto an HTTP response."""

    def __init__(self, status_code: int, message: str, *, request_id: str | None = None,
                 details: list[str] | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.request_id = request_id
        self.details = details or []
        self.headers = headers or {}


class AudioDecodeError(SpeechJudgeError):
    """Audio cannot be decoded (-> 422)."""


class AudioLimitError(SpeechJudgeError):
    """Audio exceeds configured size/duration (-> 413)."""


class StageError(SpeechJudgeError):
    """A pipeline stage failed. ``stage`` is safe to return to the caller."""

    def __init__(self, stage: str, reason: str | None = None):
        self.stage = stage
        self.reason = reason or f"Stage failed: {stage}"
        super().__init__(self.reason)


class ScoreError(SpeechJudgeError):
    """Score engine rejected its input (missing or out-of-range metric)."""


class ScoringConfigError(SpeechJudgeError):
    """Scoring configuration is invalid (service must refuse to start)."""


class ServiceLoadError(SpeechJudgeError):
    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(f"{service}: {message}")
