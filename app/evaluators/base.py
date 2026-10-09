"""Shared plumbing for Question_Evaluators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import StageError
from app.core.logging import get_logger
from app.models.domain import AudioData, Flag, QualityProbe, Transcription
from app.scoring.config import ScoringConfig
from app.models.schemas import EvaluationRequest
from app.services.acoustic.fluency import word_fluency_metrics
from app.services.registry import ServiceRegistry

log = get_logger(__name__)


@dataclass
class EvaluatorOutput:
    scoring_key: str
    raw_metrics: dict[str, float]
    transcription: dict[str, Any] | None = None
    gopt: dict[str, Any] | None = None
    acoustic: dict[str, Any] | None = None
    text_comparison: dict[str, Any] | None = None
    language: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)


class QuestionEvaluator(ABC):
    def __init__(self, services: ServiceRegistry, scoring: ScoringConfig,
                 executor: ThreadPoolExecutor) -> None:
        self.services = services
        self.scoring = scoring
        self.executor = executor

    @abstractmethod
    def evaluate(self, request: EvaluationRequest, audio: AudioData | None,
                 probe: QualityProbe | None) -> EvaluatorOutput: ...

    # -- helpers -------------------------------------------------------------
    def _submit(self, fn, *args) -> Future:
        return self.executor.submit(fn, *args)

    @staticmethod
    def _result(future: Future, stage: str):
        """Resolve a future; any failure becomes a StageError naming the stage (no stack trace
        leaves the service)."""
        try:
            return future.result()
        except StageError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Stage '%s' failed", stage)
            raise StageError(stage) from exc

    @staticmethod
    def transcript_flags(transcription: Transcription) -> list[str]:
        return [] if transcription.text.strip() else [Flag.EMPTY_TRANSCRIPT.value]

    def _refine_acoustic(self, acoustic: dict[str, Any], transcription: Transcription
                         ) -> dict[str, Any]:
        """Use the real transcript word count for rates once the transcript is known."""
        count = len(transcription.text.split())
        if count == 0:
            return acoustic
        refined = self.services.acoustic.refine_rates(acoustic, count)
        # Word-timing fluency measures (pause placement, run length, hesitations).
        refined.update(word_fluency_metrics(transcription.word_timestamps))
        return refined
