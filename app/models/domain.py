"""Internal data types shared by services, evaluators and scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

CANONICAL_SAMPLE_RATE = 16_000


class QuestionType(str, Enum):
    READING = "READING"
    REPEAT = "REPEAT"
    JUMBLED = "JUMBLED"
    QUESTION_ANSWER = "QUESTION_ANSWER"
    STORYTELLING = "STORYTELLING"


class EvaluationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    LOW_AUDIO_QUALITY = "LOW_AUDIO_QUALITY"
    FAILED = "FAILED"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Flag(str, Enum):
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    CLIPPING = "CLIPPING"
    EMPTY_TRANSCRIPT = "EMPTY_TRANSCRIPT"
    # A placeholder (mock) service produced part of this result: the score is NOT a real measurement.
    MOCKED_STT = "MOCKED_STT"
    MOCKED_ACOUSTIC = "MOCKED_ACOUSTIC"
    MOCKED_PRONUNCIATION = "MOCKED_PRONUNCIATION"
    MOCKED_LANGUAGE = "MOCKED_LANGUAGE"


@dataclass
class AudioData:
    """Canonical audio: mono, 16 kHz, float32 in [-1, 1] (equivalent to 16-bit PCM)."""

    samples: np.ndarray
    original_duration_s: float
    sample_rate: int = CANONICAL_SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return float(len(self.samples)) / self.sample_rate


@dataclass
class Transcription:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    word_timestamps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoptResult:
    """Pronunciation-assessment result; scores normalized to 0..1 (Requirement 10.1).

    ``fluency`` / ``prosody`` are None when the backend does not measure them (the phoneme
    backend leaves those to the acoustic engine). ``method`` records which backend produced
    the values: "gopt", "phoneme_ctc" or "mock".
    """

    pronunciation: float
    fluency: float | None
    prosody: float | None
    completeness: float
    words: list[dict[str, Any]] = field(default_factory=list)
    phonemes: list[dict[str, Any]] = field(default_factory=list)
    method: str = "gopt"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityProbe:
    """Cheap signal checks used by the Audio Quality Gate."""

    duration_s: float
    speech_duration_s: float
    snr_db: float
    clipping_ratio: float
    speech_segments: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class LanguageResult:
    prompt_version: str
    dimensions: dict[str, dict[str, Any]]  # name -> {"score": float, "justification": str}

    def scores(self) -> dict[str, float]:
        return {k: float(v["score"]) for k, v in self.dimensions.items()}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextComparisonResult:
    word_accuracy: float
    wer: float
    missing_words: list[str]
    extra_words: list[str]
    substituted_words: list[dict[str, str]]  # {"expected": ..., "spoken": ...}
    order_correctness: float
    expected_word_count: int
    hypothesis_word_count: int
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


QA_RUBRIC = ("relevance", "grammar", "vocabulary", "coherence", "content", "completeness")
NARRATIVE_RUBRIC = ("topic_relevance", "content", "coherence", "grammar", "vocabulary",
                    "logical_flow", "completeness")
