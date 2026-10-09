import time

from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from tests.support import (AUTH, EXPECTED_SENTENCE as EXPECTED, b64_audio, body, make_services,
                           make_settings, make_speech, post)


def wait_for_job(client, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json()
        if d["status"] in ("COMPLETED", "FAILED"):
            return d
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_async_flow_and_sync_equivalence(client):
    payload = body("REPEAT", audio=b64_audio(), expected_text=EXPECTED)
    r = post(client, payload, "/api/v1/evaluate/async")
    assert r.status_code == 202
    sub = r.json()
    assert sub["status"] == "QUEUED" and sub["request_id"] == "REQ_1" and sub["job_id"]
    job = wait_for_job(client, sub["job_id"])
    assert job["status"] == "COMPLETED" and job["request_id"] == "REQ_1"
    sync = post(client, payload).json()
    assert job["result"] == sync


def test_async_validation_errors_are_immediate(client):
    assert post(client, body("READING"), "/api/v1/evaluate/async").status_code == 422
    r = post(client, body("READING", audio="aGVsbG8=", expected_text="x"), "/api/v1/evaluate/async")
    assert r.status_code == 422


def test_async_low_audio_quality_is_completed_job(client):
    r = post(client, body("READING", audio=b64_audio(make_speech(0.3, None)), expected_text="hi there"),
             "/api/v1/evaluate/async")
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "COMPLETED" and job["result"]["status"] == "LOW_AUDIO_QUALITY"


def test_async_retries_then_fails_with_stage_reason():
    services = make_services()
    attempts = {"n": 0}

    def boom(*_a, **_k):
        attempts["n"] += 1
        raise RuntimeError("x")
    services.gopt.assess = boom
    rt = build_runtime(make_settings(job_retry_limit=2), services=services)
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED), "/api/v1/evaluate/async")
        job = wait_for_job(c, r.json()["job_id"])
        assert job["status"] == "FAILED" and "gopt" in job["reason"]
        assert attempts["n"] == 3  # 1 try + 2 retries
    rt.shutdown()


def test_unknown_job_404(client):
    assert client.get("/api/v1/jobs/nope", headers=AUTH).status_code == 404
