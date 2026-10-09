"""Temporary job state (Requirement 17, 18.3). Nothing here is permanent business data:
every record expires after ``job_retention_s``.

Legal transitions: QUEUED -> PROCESSING -> COMPLETED | FAILED (Requirement 17.7).
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.models.domain import JobStatus

_ALLOWED_FROM = {
    JobStatus.PROCESSING: {JobStatus.QUEUED},
    JobStatus.COMPLETED: {JobStatus.PROCESSING},
    JobStatus.FAILED: {JobStatus.PROCESSING},
}


@dataclass
class JobRecord:
    job_id: str
    request_id: str
    status: JobStatus
    created_at: float
    started_at: float | None = None
    result: dict[str, Any] | None = None
    reason: str | None = None
    kind: str = "evaluation"                 # "evaluation" (one question) | "section"
    meta: dict[str, Any] = field(default_factory=dict)    # section_id, student_id, question count
    finished_at: float | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "JobRecord":
        d = json.loads(raw)
        d["status"] = JobStatus(d["status"])
        return cls(**d)


def _apply_transition(rec: JobRecord, new: JobStatus, *, result: dict | None = None,
                      reason: str | None = None, now: float) -> bool:
    if rec.status not in _ALLOWED_FROM[new]:
        return False
    rec.status = new
    if new == JobStatus.PROCESSING:
        rec.started_at = now
    else:
        rec.finished_at = now
    if result is not None:
        rec.result = result
    if reason is not None:
        rec.reason = reason
    return True


class JobStore(ABC):
    @abstractmethod
    def create(self, job_id: str, request_id: str, kind: str = "evaluation",
               meta: dict[str, Any] | None = None) -> JobRecord: ...
    @abstractmethod
    def get(self, job_id: str) -> JobRecord | None: ...
    @abstractmethod
    def delete(self, job_id: str) -> None: ...
    @abstractmethod
    def _transition(self, job_id: str, new: JobStatus, result: dict | None,
                    reason: str | None) -> bool: ...

    def mark_processing(self, job_id: str) -> bool:
        return self._transition(job_id, JobStatus.PROCESSING, None, None)

    def mark_completed(self, job_id: str, result: dict[str, Any]) -> bool:
        return self._transition(job_id, JobStatus.COMPLETED, result, None)

    def mark_failed(self, job_id: str, reason: str) -> bool:
        return self._transition(job_id, JobStatus.FAILED, None, reason)


class InMemoryJobStore(JobStore):
    def __init__(self, retention_s: int = 24 * 3600, clock: Callable[[], float] = time.time):
        self._retention = retention_s
        self._clock = clock
        self._items: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        cutoff = self._clock() - self._retention
        for job_id in [k for k, v in self._items.items() if v.created_at < cutoff]:
            del self._items[job_id]

    def create(self, job_id: str, request_id: str, kind: str = "evaluation",
               meta: dict[str, Any] | None = None) -> JobRecord:
        with self._lock:
            self._purge()
            rec = JobRecord(job_id, request_id, JobStatus.QUEUED, self._clock(),
                            kind=kind, meta=dict(meta or {}))
            self._items[job_id] = rec
            return JobRecord.from_json(rec.to_json())

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            self._purge()
            rec = self._items.get(job_id)
            return None if rec is None else JobRecord.from_json(rec.to_json())

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._items.pop(job_id, None)

    def _transition(self, job_id, new, result, reason) -> bool:
        with self._lock:
            rec = self._items.get(job_id)
            return rec is not None and _apply_transition(rec, new, result=result, reason=reason,
                                                         now=self._clock())


class RedisJobStore(JobStore):
    PREFIX = "speechjudge:job:"

    def __init__(self, client, retention_s: int = 24 * 3600, clock: Callable[[], float] = time.time):
        self._r = client
        self._retention = retention_s
        self._clock = clock

    def _key(self, job_id: str) -> str:
        return f"{self.PREFIX}{job_id}"

    def create(self, job_id: str, request_id: str, kind: str = "evaluation",
               meta: dict[str, Any] | None = None) -> JobRecord:
        rec = JobRecord(job_id, request_id, JobStatus.QUEUED, self._clock(),
                        kind=kind, meta=dict(meta or {}))
        self._r.set(self._key(job_id), rec.to_json(), ex=self._retention)
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        raw = self._r.get(self._key(job_id))
        return None if raw is None else JobRecord.from_json(raw)

    def delete(self, job_id: str) -> None:
        self._r.delete(self._key(job_id))

    def _transition(self, job_id, new, result, reason) -> bool:
        import redis  # local import keeps the in-memory path dependency-free

        key = self._key(job_id)
        with self._r.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    rec = JobRecord.from_json(raw)
                    if not _apply_transition(rec, new, result=result, reason=reason,
                                             now=self._clock()):
                        pipe.unwatch()
                        return False
                    ttl = pipe.ttl(key)
                    pipe.multi()
                    pipe.set(key, rec.to_json(), ex=max(int(ttl), 1) if ttl and ttl > 0
                             else self._retention)
                    pipe.execute()
                    return True
                except redis.WatchError:
                    continue
