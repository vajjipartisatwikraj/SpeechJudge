"""Start / watch the GPU benchmark scripts (benchmarks/*.py) from the test console.

Off unless SJ_BENCHMARKS_ENABLED=true, behind the same Bearer credential as the rest of the API.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import get_runtime
from app.core.benchmark_runner import DESCRIPTIONS, SECTION_CHOICES, SCRIPTS, BenchmarkManager
from app.core.errors import ApiError
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/benchmarks", tags=["benchmarks"])


class BenchmarkRunRequest(BaseModel):
    benchmark: Literal["whisper", "phoneme", "qwen", "section"]
    section: Literal["reading", "repeat", "jumbled", "question_answer", "story_telling", "all"] = "all"
    repeat: int = Field(default=1, ge=1, le=10, description="Repetitions; the median is reported")


def manager(request: Request, runtime: Runtime = Depends(get_runtime)) -> BenchmarkManager:
    if not runtime.settings.benchmarks_enabled:
        raise ApiError(403, "Benchmarks are disabled on this server",
                       details=["set SJ_BENCHMARKS_ENABLED=true in the backend .env and restart"])
    return request.app.state.benchmarks


@router.get("")
def overview(request: Request, runtime: Runtime = Depends(get_runtime)):
    """What can be run, what is running, and the last saved results. Works while disabled so the
    console can show why the buttons are inactive."""
    enabled = runtime.settings.benchmarks_enabled
    body = {"enabled": enabled,
            "benchmarks": [{"id": name, "description": DESCRIPTIONS[name]} for name in SCRIPTS],
            "sections": list(SECTION_CHOICES), "current": None, "results": {}}
    if enabled:
        mgr: BenchmarkManager = request.app.state.benchmarks
        current = mgr.current() or mgr.latest()
        body["current"] = current.describe() if current else None
        body["results"] = mgr.results()
    return body


@router.post("/run", status_code=202)
def run(req: BenchmarkRunRequest, mgr: BenchmarkManager = Depends(manager)):
    return mgr.start(req.benchmark, req.section, req.repeat).describe()


@router.get("/results")
def results(mgr: BenchmarkManager = Depends(manager)):
    return mgr.results()


@router.get("/runs/{run_id}")
def run_status(run_id: str, mgr: BenchmarkManager = Depends(manager)):
    return mgr.get(run_id).describe()
