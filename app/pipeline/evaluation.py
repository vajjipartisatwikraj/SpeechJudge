"""End-to-end evaluation pipeline shared by the sync API and the Celery/inline workers.

validate -> decode/canonicalize -> quality gate -> route -> evaluator -> score engine -> result

Both the synchronous and asynchronous paths call the same ``run`` so identical inputs
produce identical results (Requirement 17.8).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.core.config import Settings
from app.core.errors import ApiError, ScoreError, StageError
from app.core.logging import get_logger, reset_request_id, set_request_id
from app.models.domain import AudioData, EvaluationStatus, Flag, QualityProbe
from app.audio.quality_gate import evaluate_gate
from app.pipeline.audio_input import decode_request_audio
from app.evaluators.router import QuestionRouter
from app.models.schemas import EvaluationDetails, EvaluationRequest, EvaluationResult
from app.scoring.config import ScoringConfig
from app.scoring.engine import ScoreEngine
from app.services.registry import ServiceRegistry

log = get_logger(__name__)


class EvaluationPipeline:
    def __init__(self, settings: Settings, services: ServiceRegistry, scoring: ScoringConfig,
                 max_workers: int = 8) -> None:
        self.settings = settings
        self.services = services
        self.score_engine = ScoreEngine(scoring)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sj-eval")
        self.router = QuestionRouter(services, scoring, self._executor)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    def run(self, request: EvaluationRequest, audio: AudioData | None = None,
            probe: QualityProbe | None = None) -> EvaluationResult:
        """Evaluate; raises StageError if a stage fails (used by job retry logic).

        ``probe`` may carry an already computed quality probe of ``audio`` (the section runner measures
        all questions first so they reach the shared model queues together)."""
        token = set_request_id(request.request_id)
        try:
            return self._run(request, audio, probe)
        except StageError:
            raise
        except ScoreError as exc:
            log.error("Score engine error: %s", exc)
            raise StageError("score", "Score computation failed") from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected pipeline failure")
            raise StageError("pipeline") from exc
        finally:
            reset_request_id(token)

    def evaluate(self, request: EvaluationRequest, audio: AudioData | None = None
                 ) -> EvaluationResult:
        """Like ``run`` but converts stage failures into a FAILED result (Requirement 16.5)."""
        try:
            return self.run(request, audio)
        except StageError as exc:
            return self.failed_result(request, exc.reason)

    def _mocked_flags(self, out) -> list[str]:
        """MOCKED_* flags for every placeholder service that contributed to this result, so a
        caller can never mistake a fixed placeholder value for a real measurement."""
        svc = self.services
        used = [(out.transcription, svc.stt, Flag.MOCKED_STT),
                (out.acoustic, svc.acoustic, Flag.MOCKED_ACOUSTIC),
                (out.gopt, svc.gopt, Flag.MOCKED_PRONUNCIATION),
                (out.language, svc.language, Flag.MOCKED_LANGUAGE)]
        return [flag.value for section, service, flag in used if section is not None and service.mocked]

    @staticmethod
    def failed_result(request: EvaluationRequest, reason: str) -> EvaluationResult:
        return EvaluationResult(request_id=request.request_id, status=EvaluationStatus.FAILED,
                                question_type=request.question_type, score=None, reason=reason)

    # ------------------------------------------------------------------
    def _run(self, request: EvaluationRequest, audio: AudioData | None,
             probe: QualityProbe | None = None) -> EvaluationResult:
        if audio is None and request.audio and request.audio.strip():
            try:  # worker path: audio was validated at submit time but is decoded here
                audio = decode_request_audio(request, self.settings)
            except ApiError as exc:
                raise StageError("audio_preprocessing", exc.message) from exc

        gate_flags: list[str] = []
        if audio is not None:
            if probe is None:
                try:
                    probe = self.services.acoustic.probe(audio)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Quality probe failed")
                    raise StageError("audio_quality_gate") from exc
            decision = evaluate_gate(probe, self.settings)
            if decision.rejected:
                log.info("Audio rejected by quality gate: %s", decision.reason)
                return EvaluationResult(
                    request_id=request.request_id, status=EvaluationStatus.LOW_AUDIO_QUALITY,
                    question_type=request.question_type, score=None, reason=decision.reason,
                    evaluation=EvaluationDetails(acoustic={
                        "duration_s": round(probe.duration_s, 4),
                        "speech_duration_s": round(probe.speech_duration_s, 4),
                        "snr_db": round(probe.snr_db, 3),
                        "clipping_ratio": round(probe.clipping_ratio, 6)}))
            gate_flags = decision.flags

        out = self.router.route(request.question_type).evaluate(request, audio, probe)

        outcome = self.score_engine.score(out.scoring_key, out.raw_metrics)
        flags = list(dict.fromkeys(gate_flags + out.flags + self._mocked_flags(out)))
        return EvaluationResult(
            request_id=request.request_id, status=EvaluationStatus.COMPLETED,
            question_type=request.question_type, score=outcome.score, flags=flags,
            raw_metrics=out.raw_metrics,
            evaluation=EvaluationDetails(
                transcription=out.transcription, gopt=out.gopt, acoustic=out.acoustic,
                text_comparison=out.text_comparison, language=out.language,
                score_components=outcome.components))
