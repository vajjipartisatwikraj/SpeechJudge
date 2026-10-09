"""Shared service interfaces (Requirement 21.1). Real and mock backends both satisfy these."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from app.models.domain import (AudioData, GoptResult, LanguageResult, QualityProbe, QuestionType,
                        Transcription)


class SharedService(ABC):
    #: key used in readiness output: whisper | gopt | qwen | acoustic
    name: str = ""
    #: True for placeholder implementations; results that used one are flagged MOCKED_*
    mocked: bool = False

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load models/resources exactly once. Raise ServiceLoadError on failure."""


class STTService(SharedService):
    name = "whisper"

    @abstractmethod
    def transcribe(self, audio: AudioData) -> Transcription: ...

    def transcribe_batch(self, audios: Sequence[AudioData]) -> list[Transcription]:
        """Transcribe several clips in one go. Backends that can batch on the GPU override this;
        the default simply runs them one after the other (same results, no speed-up)."""
        return [self.transcribe(a) for a in audios]


class GoptService(SharedService):
    name = "gopt"

    @abstractmethod
    def assess(self, audio: AudioData, expected_text: str) -> GoptResult: ...

    def assess_batch(self, items: Sequence[tuple[AudioData, str]]) -> list[GoptResult]:
        """Assess several (audio, expected_text) pairs in one go; override to batch on the GPU."""
        return [self.assess(audio, text) for audio, text in items]


class AcousticEngine(SharedService):
    name = "acoustic"

    @abstractmethod
    def probe(self, audio: AudioData) -> QualityProbe:
        """Cheap duration / VAD / SNR / clipping measurement for the quality gate."""

    @abstractmethod
    def analyze(self, audio: AudioData, probe: QualityProbe | None = None) -> dict[str, Any]:
        """Full Acoustic_Metrics. Must be deterministic for identical audio."""

    @abstractmethod
    def refine_rates(self, metrics: dict[str, Any], word_count: int) -> dict[str, Any]:
        """Recompute speech/articulation rate from a transcript word count."""


class LanguageEvaluator(SharedService):
    name = "qwen"

    @abstractmethod
    def evaluate(self, question_type: QuestionType, question: str, transcript: str
                 ) -> LanguageResult:
        """Score transcript against the question. Receives no acoustic data."""
