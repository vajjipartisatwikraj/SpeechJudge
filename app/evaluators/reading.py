"""Reading evaluator: Whisper + GOPT + Acoustic + Text Comparison (no LLM)."""
from __future__ import annotations

from app.models.domain import AudioData, QualityProbe, QuestionType
from app.evaluators.base import EvaluatorOutput, QuestionEvaluator
from app.models.schemas import EvaluationRequest
from app.scoring.metrics import reading as scoring_builder
from app.scoring.metrics.reading import sentence_end_of
from app.services.text_comparison.comparison import compare_text


class PronunciationEvaluator(QuestionEvaluator):
    """Shared implementation for READING and REPEAT (identical service usage)."""

    question_type: QuestionType = QuestionType.READING
    scoring_key = QuestionType.READING.value

    def evaluate(self, request: EvaluationRequest, audio: AudioData | None,
                 probe: QualityProbe | None) -> EvaluatorOutput:
        assert audio is not None and request.expected_text is not None
        svc = self.services
        # Independent work runs concurrently (Requirement 14.3).
        f_stt = self._submit(svc.stt.transcribe, audio)
        f_gopt = self._submit(svc.gopt.assess, audio, request.expected_text)
        f_acoustic = self._submit(svc.acoustic.analyze, audio, probe)

        transcription = self._result(f_stt, "stt")
        gopt = self._result(f_gopt, "gopt")
        acoustic = self._refine_acoustic(self._result(f_acoustic, "acoustic"), transcription)
        comparison = compare_text(request.expected_text, transcription.text)

        raw = scoring_builder.build_raw_metrics(comparison, gopt, acoustic, self.scoring,
                                                sentence_end_of(request.expected_text))
        return EvaluatorOutput(
            scoring_key=self.scoring_key, raw_metrics=raw,
            transcription=transcription.to_dict(), gopt=gopt.to_dict(), acoustic=acoustic,
            text_comparison=comparison.to_dict(), flags=self.transcript_flags(transcription))


class ReadingEvaluator(PronunciationEvaluator):
    question_type = QuestionType.READING
    scoring_key = QuestionType.READING.value
