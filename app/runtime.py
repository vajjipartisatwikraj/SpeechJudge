"""Composition root: builds the per-process singletons (settings, services, pipeline, job store)."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.concurrency import AdmissionGate
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.jobs.dispatch import CeleryDispatcher, InlineDispatcher, JobDispatcher
from app.jobs.runner import JobRunner
from app.jobs.section_runner import SectionRunner
from app.jobs.store import InMemoryJobStore, JobStore, RedisJobStore
from app.pipeline import EvaluationPipeline
from app.scoring.config import load_scoring_config
from app.services.batching import SharedStages
from app.services.registry import ServiceRegistry, build_registry

log = get_logger(__name__)


@dataclass
class Runtime:
    settings: Settings
    services: ServiceRegistry
    pipeline: EvaluationPipeline
    store: JobStore
    runner: JobRunner
    dispatcher: JobDispatcher
    admission: AdmissionGate
    # Whole-section evaluation: same models (``services``), but every model call goes through the
    # shared stage queues, which batch concurrent calls (see app/services/batching.py).
    stages: SharedStages | None = None
    section_pipeline: EvaluationPipeline | None = None
    section_runner: SectionRunner | None = None

    def shutdown(self) -> None:
        self.pipeline.shutdown()
        if self.section_pipeline is not None:
            self.section_pipeline.shutdown()
        if self.stages is not None:
            self.stages.shutdown()


def build_runtime(settings: Settings | None = None, *, services: ServiceRegistry | None = None,
                  store: JobStore | None = None, dispatcher: JobDispatcher | None = None
                  ) -> Runtime:
    """Build (but do not load) the runtime. Invalid scoring config raises ScoringConfigError,
    which stops startup (Requirement 15.3)."""
    settings = settings or get_settings()
    scoring = load_scoring_config(settings.scoring_config_path)
    services = services or build_registry(settings)
    pipeline = EvaluationPipeline(settings, services, scoring)

    if store is None:
        if settings.job_backend == "inline":
            store = InMemoryJobStore(settings.job_retention_s)
        else:
            import redis

            store = RedisJobStore(redis.Redis.from_url(settings.redis_url), settings.job_retention_s)
    runner = JobRunner(pipeline, store, settings)

    stages = SharedStages(services, settings)         # threads start on first use only
    section_pipeline = EvaluationPipeline(settings, stages.registry, scoring,
                                          max_workers=settings.section_pipeline_threads)
    section_runner = SectionRunner(section_pipeline, store, settings)

    if dispatcher is None:
        dispatcher = (InlineDispatcher(runner, section_runner=section_runner)
                      if settings.job_backend == "inline" else CeleryDispatcher())
    return Runtime(settings, services, pipeline, store, runner, dispatcher,
                   AdmissionGate(settings.max_concurrent_evaluations),
                   stages=stages, section_pipeline=section_pipeline, section_runner=section_runner)


# --- Worker-side singleton -------------------------------------------------------
_worker_runtime: Runtime | None = None
_worker_lock = threading.Lock()


def get_worker_runtime() -> Runtime:
    """Load models exactly once per worker process (Requirement 7.1)."""
    global _worker_runtime
    with _worker_lock:
        if _worker_runtime is None:
            rt = build_runtime()
            rt.services.load_all()
            _worker_runtime = rt
        return _worker_runtime
