"""Access control and admission checks that run before any request body is parsed."""
from __future__ import annotations

from fastapi import FastAPI, Request

from app.api.deps import credential_valid, extract_bearer
from app.api.error_handlers import error_response
from app.core.config import Settings
from app.runtime import Runtime

PROTECTED_PREFIXES = ("/api/v1/evaluate", "/api/v1/jobs", "/api/v1/benchmarks")
BODY_OVERHEAD_BYTES = 256 * 1024   # JSON envelope + text fields around the audio
SECTION_PATH = "/api/v1/evaluate/section"


def register_access_control(app: FastAPI, runtime: Runtime, settings: Settings) -> None:
    @app.middleware("http")
    async def access_control(request: Request, call_next):
        path = request.url.path
        if path.startswith(PROTECTED_PREFIXES):
            # 1. Authenticate first: a malformed body from an unauthenticated caller is a 401.
            token = extract_bearer(request.headers.get("authorization"))
            if not credential_valid(token, settings.credential_set):
                return error_response(401, "Missing or invalid service credential",
                                      headers={"WWW-Authenticate": "Bearer"})

            if request.method == "POST":
                # 2. Reject oversized bodies before reading them.
                # (a section carries the audio of all its questions)
                audio_budget = (settings.section_max_total_audio_bytes if path.rstrip("/") == SECTION_PATH
                                else settings.max_audio_bytes)
                limit = int(audio_budget * 4 / 3) + BODY_OVERHEAD_BYTES
                length = request.headers.get("content-length")
                if length and length.isdigit() and int(length) > limit:
                    return error_response(413, "Audio limit exceeded")

                # 3. Don't accept work this process can't do yet (models still loading).
                sync = path.rstrip("/") == "/api/v1/evaluate"
                uses_local_models = path.startswith("/api/v1/evaluate") and (
                    sync or settings.job_backend == "inline")
                if uses_local_models:
                    if not settings.load_models_in_api:
                        return error_response(503, "This node does not serve synchronous evaluation")
                    missing = runtime.services.not_loaded()
                    if missing:
                        return error_response(503, "Services not loaded: " + ", ".join(missing),
                                              errors=missing)
        return await call_next(request)


def register_security_headers(app: FastAPI) -> None:
    """Defensive response headers; API responses contain student results, so never cache them."""
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
