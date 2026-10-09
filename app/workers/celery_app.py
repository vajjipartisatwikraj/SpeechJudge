"""Celery worker entrypoint.

Run on a GPU host (models are loaded once in the worker process and shared by threads):

    celery -A app.workers.celery_app worker --pool=threads --concurrency=2

``--concurrency`` = how many sections / questions the worker handles at the same time. They all share
the ONE Whisper, ONE phoneme model and ONE Qwen; the stage queues batch their calls.

Delivery is at-most-once: early acknowledgement (``task_acks_late=False``) and no redelivery
when a worker dies. The API reports such jobs as FAILED ("Worker lost") after the processing
timeout (Requirement 19).
"""
from __future__ import annotations

import os
import sys

from celery import Celery
from celery.signals import worker_ready

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

celery_app = Celery("speech_judge", broker=settings.redis_url)
celery_app.conf.update(
    task_acks_late=False,
    task_reject_on_worker_lost=False,
    task_ignore_result=True,          # results live in the JobStore, with TTL
    task_track_started=False,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_serializer="json",
    accept_content=["json"],
    task_default_queue="speech_judge",
)


@celery_app.task(name="speech_judge.evaluate_job")
def evaluate_job(job_id: str, payload: dict) -> None:
    from app.runtime import get_worker_runtime

    get_worker_runtime().runner.run_job(job_id, payload)


@celery_app.task(name="speech_judge.evaluate_section")
def evaluate_section(job_id: str, payload: dict) -> None:
    """A whole exam section. Runs on the same worker process (and the same shared models) as
    single questions; the stage queues inside that process batch the model calls."""
    from app.runtime import get_worker_runtime

    get_worker_runtime().section_runner.run_section_job(job_id, payload)


@worker_ready.connect
def _load_models_on_start(**_kwargs) -> None:
    """Load every Shared_Service at worker start; abort with non-zero exit on failure (Req 7.4)."""
    from app.runtime import get_worker_runtime

    try:
        runtime = get_worker_runtime()
        log.info("Worker ready: all shared services loaded")
        import redis

        from app.workers.heartbeat import start_heartbeat
        start_heartbeat(redis.Redis.from_url(settings.redis_url), runtime.services.readiness)
    except Exception as exc:  # noqa: BLE001
        log.critical("Worker startup failed, exiting: %s", exc)
        sys.stderr.flush()
        os._exit(1)
