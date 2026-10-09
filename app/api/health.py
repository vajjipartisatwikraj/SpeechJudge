from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_runtime
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness only: touches no Shared_Service."""
    return {"status": "ok"}


def readiness_report(runtime: Runtime) -> dict[str, bool]:
    if runtime.settings.load_models_in_api:
        return runtime.services.readiness()
    import redis

    from app.workers.heartbeat import aggregate_worker_readiness
    try:
        return aggregate_worker_readiness(redis.Redis.from_url(runtime.settings.redis_url))
    except Exception:  # noqa: BLE001 - Redis unreachable means not ready
        return {"whisper": False, "gopt": False, "qwen": False, "acoustic": False}


@router.get("/ready")
def ready(runtime: Runtime = Depends(get_runtime)):
    report = readiness_report(runtime)
    all_ready = all(report.values())
    body = {"status": "ready" if all_ready else "not_ready", **report}
    if runtime.settings.load_models_in_api:
        body["mocked"] = runtime.services.mocked_names()   # services running as placeholders
    return JSONResponse(body, status_code=200 if all_ready else 503)
