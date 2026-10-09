"""Admission control (back-pressure) for synchronous evaluations."""
from __future__ import annotations

import threading


class AdmissionGate:
    """Counts evaluations in flight; refuses new ones beyond ``limit`` (0 = unlimited).

    Without it, a burst of requests queues behind the model locks, every request gets slower, and
    the slowest ones hit the request timeout. Refusing early (HTTP 429 + Retry-After) lets the
    caller retry or use the asynchronous endpoint instead.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def try_enter(self) -> bool:
        with self._lock:
            if self.limit and self._in_flight >= self.limit:
                return False
            self._in_flight += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
