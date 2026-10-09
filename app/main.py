"""FastAPI application factory.

    python -m app                                                    # uses SJ_HOST / SJ_PORT
    uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from app.api import benchmarks, evaluate, health, jobs
from app.api.error_handlers import register_error_handlers
from app.api.middleware import register_access_control, register_security_headers
from app.core.benchmark_runner import BenchmarkManager
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.runtime import Runtime, build_runtime

log = get_logger(__name__)


def _fatal_exit() -> None:  # pragma: no cover - terminates the process
    logging.shutdown()
    os._exit(1)


def _load_in_background(runtime: Runtime, on_fatal: Callable[[], None]) -> threading.Thread:
    """Load models while the API already answers /health and /ready (Requirement 7.5)."""
    def _load() -> None:
        try:
            runtime.services.load_all()
            log.info("All shared services loaded")
        except Exception as exc:  # noqa: BLE001
            log.critical("Startup failed, terminating: %s", exc)
            on_fatal()

    t = threading.Thread(target=_load, name="sj-model-loader", daemon=True)
    t.start()
    return t


def create_app(runtime: Runtime | None = None, settings: Settings | None = None,
               on_fatal: Callable[[], None] = _fatal_exit) -> FastAPI:
    settings = settings or (runtime.settings if runtime else get_settings())
    configure_logging(settings.log_level)
    # Raises ScoringConfigError on invalid weights -> process refuses to start (Req 15.3).
    runtime = runtime or build_runtime(settings)
    log.info("Hardware profile: %s (SJ_GPU_PRESENT=%s) device=%s whisper=%s/%s qwen=%s batch=%d/%d",
             settings.hardware_profile.upper(), str(settings.gpu_present).lower(), settings.device,
             settings.whisper_model, settings.whisper_compute_type, settings.qwen_backend,
             settings.whisper_batch_size, settings.phoneme_batch_size)

    if not settings.credential_set:
        log.warning("No SJ_SERVICE_CREDENTIALS configured: every protected request will get 401")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.load_models_in_api and not runtime.services.ready:
            _load_in_background(runtime, on_fatal)
        yield
        runtime.shutdown()

    docs = {} if settings.enable_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="Speech Judge", version="1.0.0", lifespan=lifespan, **docs)
    app.state.runtime = runtime
    bench_host = "127.0.0.1" if settings.host in ("0.0.0.0", "::", "") else settings.host
    app.state.benchmarks = BenchmarkManager(base_url=f"http://{bench_host}:{settings.port}")
    if settings.benchmarks_enabled:
        log.warning("Benchmark endpoints are ENABLED (SJ_BENCHMARKS_ENABLED=true): authenticated callers "
                    "can start GPU benchmark runs%s", " on a PRODUCTION node" if settings.is_production else "")

    register_access_control(app, runtime, settings)
    register_security_headers(app)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(evaluate.router)
    app.include_router(jobs.router)
    app.include_router(benchmarks.router)
    return app
