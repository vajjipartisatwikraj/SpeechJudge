from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.deps import get_runtime
from app.audio.decoder import decode_base64_audio
from app.core.errors import ApiError, AudioDecodeError, AudioLimitError
from app.core.logging import get_logger
from app.pipeline import decode_request_audio
from app.runtime import Runtime
from app.models.schemas import (AsyncSubmitResponse, EvaluationRequest, EvaluationResult,
                                SectionRequest)
from app.api.validation import validate_request, validate_section
from app.models.domain import JobStatus

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate(req: EvaluationRequest, runtime: Runtime = Depends(get_runtime)):
    validate_request(req)

    gate = runtime.admission          # back-pressure: refuse early instead of queueing until timeout
    if not gate.try_enter():
        raise ApiError(429, "Server busy: too many evaluations in progress. Retry shortly or use "
                            "/api/v1/evaluate/async.", request_id=req.request_id,
                       headers={"Retry-After": "5"})
    try:
        audio = await run_in_threadpool(decode_request_audio, req, runtime.settings)
    except BaseException:
        gate.leave()
        raise

    def work():
        try:
            return runtime.pipeline.evaluate(req, audio)
        finally:
            gate.leave()              # released when the work really ends, even after a 504

    try:
        return await asyncio.wait_for(run_in_threadpool(work), timeout=runtime.settings.request_timeout_s)
    except asyncio.TimeoutError:
        log.error("Synchronous evaluation timed out for request_id=%s", req.request_id)
        return JSONResponse(
            {"detail": "Evaluation timed out", "errors": [], "request_id": req.request_id},
            status_code=504)


def _precheck_section_audio(req: SectionRequest, max_bytes: int) -> None:
    """Cheap checks only (valid base64, size limit). Audio that turns out to be undecodable fails
    that one question inside the job instead of rejecting the other questions of the section."""
    for question in req.questions:
        if question.audio and question.audio.strip():
            try:
                decode_base64_audio(question.audio, max_bytes)
            except AudioLimitError as exc:
                raise ApiError(413, f"Audio limit exceeded in question {question.question_id}",
                               request_id=req.request_id) from exc
            except AudioDecodeError as exc:
                raise ApiError(422, f"Audio of question {question.question_id} is invalid: {exc}",
                               request_id=req.request_id) from exc


@router.post("/evaluate/section", status_code=202, response_model=AsyncSubmitResponse)
async def evaluate_section(req: SectionRequest, runtime: Runtime = Depends(get_runtime)):
    """Submit a whole exam section (all its questions) as one job; poll GET /jobs/{job_id}."""
    validate_section(req, runtime.settings)
    await run_in_threadpool(_precheck_section_audio, req, runtime.settings.max_audio_bytes)

    job_id = uuid.uuid4().hex
    runtime.store.create(job_id, req.request_id, kind="section", meta={
        "section_id": req.section_id.value, "student_id": req.student_id,
        "questions_total": len(req.questions)})
    try:
        runtime.dispatcher.submit_section(job_id, req.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        log.exception("Failed to enqueue section job")
        runtime.store.delete(job_id)
        raise ApiError(503, "Could not enqueue job", request_id=req.request_id) from exc
    return AsyncSubmitResponse(request_id=req.request_id, job_id=job_id, status=JobStatus.QUEUED)


@router.post("/evaluate/async", status_code=202, response_model=AsyncSubmitResponse)
async def evaluate_async(req: EvaluationRequest, runtime: Runtime = Depends(get_runtime)):
    validate_request(req)
    # Undecodable / oversized audio is rejected at submit time, not discovered later in a worker.
    await run_in_threadpool(decode_request_audio, req, runtime.settings)

    job_id = uuid.uuid4().hex
    runtime.store.create(job_id, req.request_id)
    try:
        # Payload (incl. base64 audio) is transient: it lives in the broker only until a worker
        # picks it up, and is never written to a permanent store.
        runtime.dispatcher.submit(job_id, req.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        log.exception("Failed to enqueue job")
        runtime.store.delete(job_id)
        raise ApiError(503, "Could not enqueue job", request_id=req.request_id) from exc
    return AsyncSubmitResponse(request_id=req.request_id, job_id=job_id, status=JobStatus.QUEUED)
