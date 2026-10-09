"""Jumbled evaluator.

* audio supplied            -> spoken: STT + Text Comparison + Acoustic (no GOPT, no LLM)
* submitted text, no audio  -> UI-based: Text Comparison only
* neither                   -> no Shared_Service is invoked
"""
from __future__ import annotations

from typing import Any

from app.core.errors import StageError
from app.models.domain import AudioData, QualityProbe
from app.evaluators.base import EvaluatorOutput, QuestionEvaluator
from app.models.schemas import EvaluationRequest
from app.scoring.metrics import jumbled as scoring_builder
from app.scoring.metrics.reading import sentence_end_of
from app.services.text_comparison.comparison import compare_text

SPOKEN_KEY = "JUMBLED_SPOKEN"
UI_KEY = "JUMBLED_UI"


def get_submitted_text(question_config: dict[str, Any]) -> str | None:
    """Submitted_Text from ``question_config['submitted_text']`` (string or list of words)."""
    value = question_config.get("submitted_text")
    if isinstance(value, list):
        value = " ".join(str(w) for w in value)
    if isinstance(value, str) and value.strip():
        return value
    return None


class JumbledEvaluator(QuestionEvaluator):
    def evaluate(self, request: EvaluationRequest, audio: AudioData | None,
                 probe: QualityProbe | None) -> EvaluatorOutput:
        assert request.expected_text is not None
        submitted = get_submitted_text(request.question_config)

        if audio is not None:
            f_stt = self._submit(self.services.stt.transcribe, audio)
            f_acoustic = self._submit(self.services.acoustic.analyze, audio, probe)
            transcription = self._result(f_stt, "stt")
            acoustic = self._refine_acoustic(self._result(f_acoustic, "acoustic"), transcription)
            comparison = compare_text(request.expected_text, transcription.text)
            raw = scoring_builder.build_spoken_raw_metrics(
                comparison, acoustic, self.scoring, sentence_end_of(request.expected_text))
            return EvaluatorOutput(
                scoring_key=SPOKEN_KEY, raw_metrics=raw, transcription=transcription.to_dict(),
                acoustic=acoustic, text_comparison=comparison.to_dict(),
                flags=self.transcript_flags(transcription))

        if submitted is not None:
            comparison = compare_text(request.expected_text, submitted)
            return EvaluatorOutput(scoring_key=UI_KEY,
                                   raw_metrics=scoring_builder.build_ui_raw_metrics(comparison),
                                   text_comparison=comparison.to_dict())

        # Request validation rejects this earlier; defend in depth without invoking services.
        raise StageError("input", "Jumbled request has neither audio nor submitted text")
