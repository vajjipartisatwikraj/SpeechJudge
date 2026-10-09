"""Logging that always carries request_id (Requirement 18.4).

Audio content and credentials must never be passed to the logger (Requirement 3.5).
"""
from __future__ import annotations

import contextvars
import logging

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"))
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def set_request_id(request_id: str) -> contextvars.Token:
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id.reset(token)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
