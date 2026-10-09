"""Turn the audio in an Evaluation_Request into canonical AudioData (or an HTTP-level error)."""
from __future__ import annotations

from app.audio.decoder import decode_base64_audio, preprocess_audio
from app.core.config import Settings
from app.core.errors import ApiError, AudioDecodeError, AudioLimitError
from app.models.domain import AudioData
from app.models.schemas import EvaluationRequest


def decode_request_audio(request: EvaluationRequest, settings: Settings) -> AudioData | None:
    """Decode request audio, mapping failures to HTTP errors (422 undecodable / 413 too large)."""
    if not request.audio or not request.audio.strip():
        return None
    try:
        raw = decode_base64_audio(request.audio, settings.max_audio_bytes)
        return preprocess_audio(raw, settings)
    except AudioLimitError as exc:
        raise ApiError(413, "Audio limit exceeded", request_id=request.request_id) from exc
    except AudioDecodeError as exc:
        raise ApiError(422, f"Audio is undecodable: {exc}", request_id=request.request_id) from exc
