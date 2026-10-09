"""GOPT pronunciation-assessment service (Requirement 10).

GOPT (Gong et al., 2022) is not a pip-installable library: it needs Kaldi-based
forced alignment / GOP feature extraction plus the trained transformer checkpoint.
That inference stack is isolated behind ``GoptBackend`` and supplied via
``SJ_GOPT_BACKEND="package.module:ClassName"``. This service owns everything around it:
loading once, thread-safety, and normalizing GOPT's native scales to 0..1.

Native GOPT scales: phone accuracy 0-2; word scores 0-10; utterance scores 0-10.
"""
from __future__ import annotations

import importlib
import threading
from typing import Any, Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import ServiceLoadError
from app.models.domain import AudioData, GoptResult
from app.services.interfaces import GoptService as GoptServiceInterface

PHONE_MAX = 2.0
WORD_MAX = 10.0
UTTERANCE_MAX = 10.0


class GoptBackend(Protocol):
    """Contract for the GOPT inference stack.

    ``score`` must return::

        {"utterance": {"accuracy", "completeness", "fluency", "prosodic", "total"},   # 0-10
         "words":  [{"word": str, "accuracy": 0-10, "stress": 0-10, "total": 0-10}, ...],
         "phones": [{"phone": str, "word_index": int, "accuracy": 0-2}, ...]}
    """

    def __init__(self, checkpoint: str, device: str) -> None: ...

    def score(self, samples: np.ndarray, sample_rate: int, expected_text: str
              ) -> dict[str, Any]: ...


def _norm(value: Any, maximum: float) -> float:
    return min(1.0, max(0.0, float(value) / maximum))


def normalize_gopt_output(raw: dict[str, Any]) -> GoptResult:
    """Convert native GOPT output to the 0..1 GoptResult."""
    utt = raw["utterance"]
    words = [{**w, "accuracy": _norm(w.get("accuracy", 0), WORD_MAX),
              "stress": _norm(w.get("stress", 0), WORD_MAX),
              "total": _norm(w.get("total", 0), WORD_MAX)} for w in raw.get("words", [])]
    phones = [{**p, "accuracy": _norm(p.get("accuracy", 0), PHONE_MAX)}
              for p in raw.get("phones", [])]
    return GoptResult(
        pronunciation=_norm(utt["accuracy"], UTTERANCE_MAX),
        fluency=_norm(utt["fluency"], UTTERANCE_MAX),
        prosody=_norm(utt["prosodic"], UTTERANCE_MAX),
        completeness=_norm(utt["completeness"], UTTERANCE_MAX),
        words=words,
        phonemes=phones,
    )


class GoptPronunciationService(GoptServiceInterface):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._backend: GoptBackend | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        s = self._settings
        if not s.gopt_backend or not s.gopt_checkpoint:
            raise ServiceLoadError(self.name, "SJ_GOPT_BACKEND and SJ_GOPT_CHECKPOINT must be set")
        module_name, _, class_name = s.gopt_backend.partition(":")
        if not module_name or not class_name:
            raise ServiceLoadError(self.name, "SJ_GOPT_BACKEND must look like 'package.module:Class'")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            self._backend = cls(s.gopt_checkpoint, s.device)
        except Exception as exc:  # noqa: BLE001
            raise ServiceLoadError(self.name, f"could not initialise backend: {exc}") from exc
        self._loaded = True

    def assess(self, audio: AudioData, expected_text: str) -> GoptResult:
        assert self._backend is not None, "GoptPronunciationService not loaded"
        with self._lock:
            raw = self._backend.score(audio.samples, audio.sample_rate, expected_text)
        return normalize_gopt_output(raw)
