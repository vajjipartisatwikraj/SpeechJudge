"""How async jobs reach a worker: Celery/Redis in production, local threads for development."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from app.jobs.runner import JobRunner
from app.jobs.section_runner import SectionRunner


class JobDispatcher(Protocol):
    def submit(self, job_id: str, payload: dict) -> None: ...
    def submit_section(self, job_id: str, payload: dict) -> None: ...


class InlineDispatcher:
    """Runs jobs on local threads. For development/tests; no Redis required."""

    def __init__(self, runner: JobRunner, workers: int = 2, section_runner: SectionRunner | None = None
                 ) -> None:
        self._runner = runner
        self._section_runner = section_runner
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sj-job")

    def submit(self, job_id: str, payload: dict) -> None:
        self._pool.submit(self._runner.run_job, job_id, payload)

    def submit_section(self, job_id: str, payload: dict) -> None:
        assert self._section_runner is not None, "no section runner configured"
        self._pool.submit(self._section_runner.run_section_job, job_id, payload)


class CeleryDispatcher:
    """Publishes the job to the Redis queue; a worker process picks it up (at-most-once)."""

    def submit(self, job_id: str, payload: dict) -> None:
        from app.workers.celery_app import evaluate_job

        evaluate_job.apply_async(args=[job_id, payload], task_id=job_id)

    def submit_section(self, job_id: str, payload: dict) -> None:
        from app.workers.celery_app import evaluate_section

        evaluate_section.apply_async(args=[job_id, payload], task_id=job_id)
