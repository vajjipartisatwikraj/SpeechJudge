"""Job execution with in-worker retries (Requirement 17.2-17.4) and lost-worker detection (19.3)."""
from __future__ import annotations

import time
from typing import Callable

from app.core.config import Settings
from app.core.errors import StageError
from app.core.logging import get_logger, reset_request_id, set_request_id
from app.models.domain import JobStatus
from app.jobs.store import JobRecord, JobStore
from app.pipeline import EvaluationPipeline
from app.models.schemas import EvaluationRequest

log = get_logger(__name__)

WORKER_LOST_REASON = "Worker lost"


class JobRunner:
    def __init__(self, pipeline: EvaluationPipeline, store: JobStore, settings: Settings) -> None:
        self.pipeline = pipeline
        self.store = store
        self.settings = settings

    def run_job(self, job_id: str, payload: dict) -> None:
        if not self.store.mark_processing(job_id):
            log.warning("Job %s not startable (expired, duplicate or already processed)", job_id)
            return
        try:
            request = EvaluationRequest.model_validate(payload)
        except Exception:  # noqa: BLE001
            self.store.mark_failed(job_id, "Invalid job payload")
            return

        token = set_request_id(request.request_id)
        try:
            attempts = 1 + max(0, self.settings.job_retry_limit)
            last: StageError | None = None
            for attempt in range(1, attempts + 1):
                try:
                    result = self.pipeline.run(request)
                except StageError as exc:
                    last = exc
                    log.warning("Job %s attempt %d/%d failed at stage '%s'", job_id, attempt,
                                attempts, exc.stage)
                    continue
                self.store.mark_completed(job_id, result.model_dump(mode="json"))
                return
            self.store.mark_failed(job_id, last.reason if last else "Evaluation failed")
        finally:
            reset_request_id(token)


def get_job(store: JobStore, job_id: str, settings: Settings,
            clock: Callable[[], float] = time.time) -> JobRecord | None:
    """Fetch a job; a job stuck in PROCESSING past the timeout is failed as 'Worker lost'."""
    rec = store.get(job_id)
    if rec is None:
        return None
    limit = settings.section_processing_timeout_s if rec.kind == "section" else settings.processing_timeout_s
    if (rec.status == JobStatus.PROCESSING and rec.started_at is not None
            and clock() - rec.started_at > limit):
        store.mark_failed(job_id, WORKER_LOST_REASON)
        return store.get(job_id)
    return rec
