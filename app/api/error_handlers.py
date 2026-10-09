"""Uniform JSON error bodies: ``{"detail", "errors", "request_id"}``. Never echoes request input
(it may contain audio) and never exposes stack traces."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import ApiError
from app.core.logging import get_logger

log = get_logger(__name__)


def error_response(status: int, detail: str, request_id: str | None = None,
                   errors: list[str] | None = None, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail, "errors": errors or [], "request_id": request_id},
                        status_code=status, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError):
        return error_response(exc.status_code, exc.message, exc.request_id, exc.details,
                              headers=exc.headers or None)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        request_id = None
        if isinstance(exc.body, dict) and isinstance(exc.body.get("request_id"), str):
            request_id = exc.body["request_id"]
        errors = [f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
                  for e in exc.errors()]
        return error_response(422, "Invalid request", request_id, errors)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return error_response(500, "Internal server error")
