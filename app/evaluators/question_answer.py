"""Open-ended evaluators: Whisper + Acoustic (concurrent) -> Qwen on the transcript."""
from __future__ import annotations

from app.models.domain import (AudioData, LanguageResult, QualityProbe, QuestionType)
from app.evaluators.base import EvaluatorOutput, QuestionEvaluator
from app.models.schemas import EvaluationRequest
from app.scoring.metrics.question_answer import build_language_raw_metrics
from app.services.language.prompts import rubric_for

NO_SPEECH_JUSTIFICATION = "No speech was transcribed"


class OpenEndedEvaluator(QuestionEvaluator):
    question_type: QuestionType = QuestionType.QUESTION_ANSWER
    scoring_key = QuestionType.QUESTION_ANSWER.value

    def evaluate(self, request: EvaluationRequest, audio: AudioData | None,
                 probe: QualityProbe | None) -> EvaluatorOutput:
        assert audio is not None and request.question is not None
        svc = self.services

        # STT and Acoustic start together (Req 14.1); the LLM starts as soon as the transcript
        # exists, without waiting for Acoustic (Req 14.2).
        f_stt = self._submit(svc.stt.transcribe, audio)
        f_acoustic = self._submit(svc.acoustic.analyze, audio, probe)

        transcription = self._result(f_stt, "stt")
        if transcription.text.strip():
            language = self._result(
                self._submit(svc.language.evaluate, self.question_type, request.question,
                             transcription.text), "language")
        else:
            # Nothing to judge: score the rubric 0 deterministically instead of asking an LLM
            # to grade an empty string. EMPTY_TRANSCRIPT is flagged below.
            language = LanguageResult(
                prompt_version="n/a",
                dimensions={d: {"score": 0.0, "justification": NO_SPEECH_JUSTIFICATION}
                            for d in rubric_for(self.question_type)})

        acoustic = self._refine_acoustic(self._result(f_acoustic, "acoustic"), transcription)
        raw = build_language_raw_metrics(language, acoustic, self.scoring)
        return EvaluatorOutput(
            scoring_key=self.scoring_key, raw_metrics=raw,
            transcription=transcription.to_dict(), acoustic=acoustic, language=language.to_dict(),
            flags=self.transcript_flags(transcription))


class QuestionAnswerEvaluator(OpenEndedEvaluator):
    question_type = QuestionType.QUESTION_ANSWER
    scoring_key = QuestionType.QUESTION_ANSWER.value
