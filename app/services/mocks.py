"""Mock Shared_Services: fixed, schema-valid outputs, no models, no GPU (Requirement 21)."""
from __future__ import annotations

from typing import Any

from app.models.domain import (NARRATIVE_RUBRIC, QA_RUBRIC, AudioData, GoptResult, LanguageResult,
                        QualityProbe, QuestionType, Transcription)
from app.services.interfaces import AcousticEngine, GoptService, LanguageEvaluator, STTService

DEFAULT_MOCK_TRANSCRIPT = "The weather is beautiful today."


class MockSTT(STTService):
    mocked = True

    def __init__(self, text: str = DEFAULT_MOCK_TRANSCRIPT) -> None:
        super().__init__()
        self.text = text
        self.calls = 0

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio: AudioData) -> Transcription:
        self.calls += 1
        words = self.text.split()
        step = 0.4
        return Transcription(
            text=self.text,
            segments=[{"start": 0.0, "end": round(step * len(words), 2), "text": self.text}],
            word_timestamps=[{"word": w, "start": round(i * step, 2),
                              "end": round((i + 1) * step, 2), "probability": 0.95}
                             for i, w in enumerate(words)])


class MockGopt(GoptService):
    mocked = True

    def __init__(self, pronunciation: float = 0.82, fluency: float = 0.76, prosody: float = 0.79,
                 completeness: float = 0.91) -> None:
        super().__init__()
        self._scores = (pronunciation, fluency, prosody, completeness)
        self.calls = 0

    def load(self) -> None:
        self._loaded = True

    def assess(self, audio: AudioData, expected_text: str) -> GoptResult:
        self.calls += 1
        p, f, pr, c = self._scores
        words = [{"word": w, "accuracy": p} for w in expected_text.split()]
        return GoptResult(pronunciation=p, fluency=f, prosody=pr, completeness=c, words=words,
                          phonemes=[], method="mock")


class MockAcousticEngine(AcousticEngine):
    """Fixed metrics; the quality gate always passes."""

    mocked = True

    METRICS: dict[str, Any] = {
        "speech_duration_s": 3.2, "silence_duration_s": 0.8, "pause_count": 1,
        "pause_duration_s": 0.4, "pause_mean_s": 0.4, "pause_max_s": 0.4, "pause_ratio": 0.12,
        "leading_silence_s": 0.2, "trailing_silence_s": 0.3, "speech_span_s": 3.5,
        "pitch_mean_hz": 180.0, "pitch_min_hz": 110.0, "pitch_max_hz": 260.0,
        "pitch_range_hz": 150.0, "pitch_std_hz": 32.0, "voiced_ratio": 0.6,
        "pitch_median_hz": 170.0, "pitch_std_st": 2.6, "pitch_range_st": 7.5,
        "terminal_pitch_delta_st": -1.2, "syllable_energy_std_db": 3.0,
        "pitch_source": "mock", "energy_mean_db": -24.0, "energy_std_db": 6.5,
        "peak_amplitude": 0.6, "loudness_lufs": -20.0, "loudness_source": "mock",
        "snr_db": 28.0, "clipping_ratio": 0.0, "syllable_count_estimate": 8,
        "speech_rate_wpm": 128.0, "articulation_rate_wpm": 140.0, "rate_source": "estimated",
        "audio_quality_score": 0.9,
    }

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def load(self) -> None:
        self._loaded = True

    def probe(self, audio: AudioData) -> QualityProbe:
        return QualityProbe(duration_s=max(audio.duration_s, 4.0), speech_duration_s=3.2,
                            snr_db=28.0, clipping_ratio=0.0, speech_segments=[(0.2, 3.4)])

    def analyze(self, audio: AudioData, probe: QualityProbe | None = None) -> dict[str, Any]:
        self.calls += 1
        return {"original_duration_s": round(audio.original_duration_s, 4),
                "duration_s": round(audio.duration_s, 4), **self.METRICS}

    def refine_rates(self, metrics: dict[str, Any], word_count: int) -> dict[str, Any]:
        return {**metrics, "rate_source": "transcript"}


class MockLanguageEvaluator(LanguageEvaluator):
    mocked = True

    def __init__(self, score: float = 0.8, prompt_version: str = "mock") -> None:
        super().__init__()
        self.score = score
        self.prompt_version = prompt_version
        self.calls = 0
        self.last_args: tuple | None = None

    def load(self) -> None:
        self._loaded = True

    def evaluate(self, question_type: QuestionType, question: str, transcript: str
                 ) -> LanguageResult:
        self.calls += 1
        self.last_args = (question_type, question, transcript)
        rubric = QA_RUBRIC if question_type == QuestionType.QUESTION_ANSWER else NARRATIVE_RUBRIC
        return LanguageResult(self.prompt_version,
                              {d: {"score": self.score, "justification": "mock"} for d in rubric})
