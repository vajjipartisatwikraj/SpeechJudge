from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_runtime
from app.core.errors import ApiError
from app.models.domain import JobStatus
from app.jobs.runner import get_job
from app.jobs.store import JobRecord
from app.runtime import Runtime
from app.models.schemas import (EvaluationResult, JobResponse, SectionJobResponse, SectionResult,
                                SectionTiming)

router = APIRouter(prefix="/api/v1", tags=["jobs"])


def _seconds(later: float | None, earlier: float | None) -> float | None:
    return None if later is None or earlier is None else round(max(later - earlier, 0.0), 3)


def section_job_response(rec: JobRecord) -> SectionJobResponse:
    meta = rec.meta or {}
    resp = SectionJobResponse(
        job_id=rec.job_id, request_id=rec.request_id, status=rec.status,
        section_id=meta.get("section_id"), student_id=meta.get("student_id"),
        questions_total=meta.get("questions_total"),
        timing=SectionTiming(queued_s=_seconds(rec.started_at, rec.created_at),
                             processing_s=_seconds(rec.finished_at, rec.started_at),
                             total_s=_seconds(rec.finished_at, rec.created_at)),
        reason=rec.reason if rec.status == JobStatus.FAILED else None)
    if rec.status == JobStatus.COMPLETED and rec.result is not None:
        done = SectionResult.model_validate(rec.result)
        resp.section_id, resp.student_id = done.section_id, done.student_id
        resp.questions_total, resp.questions_scored = done.questions_total, done.questions_scored
        resp.section_score, resp.results = done.section_score, done.results
    return resp


@router.get("/jobs/{job_id}", response_model=None)
def job_status(job_id: str, runtime: Runtime = Depends(get_runtime)):
    """Status / result of an async job: a single question (/evaluate/async) or a section."""
    rec = get_job(runtime.store, job_id, runtime.settings)
    if rec is None:
        raise ApiError(404, "Job not found or expired")
    if rec.kind == "section":
        return section_job_response(rec)
    result = None
    if rec.status == JobStatus.COMPLETED and rec.result is not None:
        result = EvaluationResult.model_validate(rec.result)
    reason = rec.reason if rec.status == JobStatus.FAILED else None
    return JobResponse(job_id=rec.job_id, request_id=rec.request_id, status=rec.status,
                       result=result, reason=reason)
