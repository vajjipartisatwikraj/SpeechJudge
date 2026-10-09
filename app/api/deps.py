from __future__ import annotations

import hmac

from fastapi import Request

from app.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def credential_valid(presented: str | None, accepted: frozenset[str]) -> bool:
    """Constant-time comparison against every accepted credential."""
    if not presented:
        return False
    ok = False
    for candidate in accepted:
        ok |= hmac.compare_digest(presented.encode(), candidate.encode())
    return ok


def extract_bearer(header_value: str | None) -> str | None:
    if not header_value:
        return None
    scheme, _, token = header_value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None
