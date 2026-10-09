import fakeredis
import pytest

from app.jobs.runner import WORKER_LOST_REASON, get_job
from app.jobs.store import InMemoryJobStore, RedisJobStore
from app.models.domain import JobStatus
from tests.support import make_settings


@pytest.fixture(params=["memory", "redis"])
def store(request):
    clock = {"t": 1000.0}
    s = (InMemoryJobStore(3600, clock=lambda: clock["t"]) if request.param == "memory"
         else RedisJobStore(fakeredis.FakeRedis(), 3600, clock=lambda: clock["t"]))
    s.clock = clock
    return s


def test_job_lifecycle_and_transition_rules(store):
    store.create("j", "R")
    assert store.get("j").status == JobStatus.QUEUED
    assert not store.mark_completed("j", {})            # QUEUED -> COMPLETED is illegal
    assert store.mark_processing("j")
    assert not store.mark_processing("j")               # at most one worker starts it
    assert store.mark_completed("j", {"a": 1})
    assert not store.mark_failed("j", "late")           # terminal states are final
    rec = store.get("j")
    assert rec.status == JobStatus.COMPLETED and rec.result == {"a": 1}


def test_worker_lost_after_processing_timeout(store):
    s = make_settings(processing_timeout_s=300)
    store.create("j", "R")
    store.mark_processing("j")
    clock = store.clock
    clock["t"] += 100
    assert get_job(store, "j", s, clock=lambda: clock["t"]).status == JobStatus.PROCESSING
    clock["t"] += 250
    rec = get_job(store, "j", s, clock=lambda: clock["t"])
    assert rec.status == JobStatus.FAILED and rec.reason == WORKER_LOST_REASON
    assert not store.mark_completed("j", {})            # a zombie worker cannot resurrect it


def test_in_memory_expiry():
    clock = {"t": 0.0}
    s = InMemoryJobStore(10, clock=lambda: clock["t"])
    s.create("j", "R")
    clock["t"] = 11
    assert s.get("j") is None


def test_redis_sets_ttl():
    r = fakeredis.FakeRedis()
    RedisJobStore(r, 86400).create("j", "R")
    assert 0 < r.ttl("speechjudge:job:j") <= 86400
