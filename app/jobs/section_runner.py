"""Evaluate a whole exam section (e.g. 10 Repeat questions) in one job.

    decode all audio (in parallel)  ->  evaluate every question at the same time  ->  section result

"At the same time" is what makes batching work: the 10 questions put their audio on the shared Whisper
and phoneme queues together, the workers take them as ONE batch, and the 10 scores are produced by the
same pipeline and score engine as a single ``/evaluate`` call (identical results).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from app.core.config import Settings
from app.core.errors import ApiError, StageError
from app.core.logging import get_logger
from app.jobs.store import JobStore
from app.models.domain import AudioData, EvaluationStatus, QualityProbe
from app.models.schemas import (EvaluationRequest, EvaluationResult, SectionQuestionResult,
                                SectionRequest, SectionResult)
from app.pipeline import EvaluationPipeline, decode_request_audio

log = get_logger(__name__)


def question_requests(section: SectionRequest) -> list[EvaluationRequest]:
    """One single-question request per section question (request_id = '<section>:<question>')."""
    return [EvaluationRequest(request_id=f"{section.request_id}:{q.question_id}",
                              question_type=section.section_id, audio=q.audio,
                              expected_text=q.expected_text, question=q.question,
                              question_config=q.question_config)
            for q in section.questions]


class SectionRunner:
    def __init__(self, pipeline: EvaluationPipeline, store: JobStore, settings: Settings) -> None:
        self.pipeline = pipeline
        self.store = store
        self.settings = settings

    # ---- job entry point (Celery task / inline thread) ---------------------------------------
    def run_section_job(self, job_id: str, payload: dict) -> None:
        if not self.store.mark_processing(job_id):
            log.warning("Section job %s not startable (expired, duplicate or already processed)", job_id)
            return
        try:
            section = SectionRequest.model_validate(payload)
        except Exception:  # noqa: BLE001
            self.store.mark_failed(job_id, "Invalid job payload")
            return
        try:
            t0 = time.perf_counter()
            result = self.run_section(section)
            log.info("Section %s (%s): %d questions in %.2f s, section_score=%s", section.request_id,
                     section.section_id.value, len(section.questions), time.perf_counter() - t0,
                     result.section_score)
        except Exception:  # noqa: BLE001
            log.exception("Section job %s failed", job_id)
            self.store.mark_failed(job_id, "Section evaluation failed")
            return
        self.store.mark_completed(job_id, result.model_dump(mode="json"))

    # ---- the work ----------------------------------------------------------------------------
    def run_section(self, section: SectionRequest) -> SectionResult:
        requests = question_requests(section)
        workers = max(1, min(len(requests), self.settings.section_question_parallelism))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sj-section") as pool:
            # 1. decode + quality-probe everything first, so that all questions reach the stage queues together
            prepared = list(pool.map(self._prepare, requests))
            # 2. evaluate all questions concurrently; the stage workers batch their model calls
            futures = {i: pool.submit(self._evaluate, req, *prep)
                       for i, (req, prep) in enumerate(zip(requests, prepared))
                       if not isinstance(prep, str)}
            results = [futures[i].result() if i in futures
                       else self.pipeline.failed_result(req, prepared[i])      # prepared[i] is the error text
                       for i, req in enumerate(requests)]
        return self._assemble(section, results)

    def _prepare(self, request: EvaluationRequest) -> tuple[AudioData | None, QualityProbe | None] | str:
        """Decode the audio and measure it (the cheap quality probe), or return an error message.
        Doing this for every question BEFORE any model call lets all questions reach the shared
        model queues at the same moment."""
        try:
            audio = decode_request_audio(request, self.settings)
        except ApiError as exc:
            return exc.message
        except Exception:  # noqa: BLE001
            log.exception("Audio decoding failed")
            return "Audio is undecodable"
        if audio is None:
            return None, None
        try:
            return audio, self.pipeline.services.acoustic.probe(audio)
        except Exception:  # noqa: BLE001
            log.exception("Quality probe failed")
            return "Stage failed: audio_quality_gate"

    def _evaluate(self, request: EvaluationRequest, audio: AudioData | None,
                  probe: QualityProbe | None) -> EvaluationResult:
        attempts = 1 + max(0, self.settings.job_retry_limit)
        last: StageError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.pipeline.run(request, audio, probe)
            except StageError as exc:
                last = exc
                log.warning("Question %s attempt %d/%d failed at stage '%s'", request.request_id,
                            attempt, attempts, exc.stage)
        return self.pipeline.failed_result(request, last.reason if last else "Evaluation failed")

    @staticmethod
    def _assemble(section: SectionRequest, results: list[EvaluationResult]) -> SectionResult:
        items = []
        for q, res in zip(section.questions, results):
            items.append(SectionQuestionResult(
                question_id=q.question_id, status=res.status, score=res.score, reason=res.reason,
                flags=res.flags, detail=res if section.include_details else None))
        scored = [r.score for r in results
                  if r.status == EvaluationStatus.COMPLETED and r.score is not None]
        return SectionResult(
            section_id=section.section_id, student_id=section.student_id, results=items,
            section_score=round(sum(scored) / len(scored), 6) if scored else None,
            questions_total=len(results), questions_scored=len(scored))
