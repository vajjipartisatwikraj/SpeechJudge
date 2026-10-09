"""Production hardening: back-pressure (429) and response headers."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from tests.support import AUTH, EXPECTED_SENTENCE as EXPECTED, b64_audio, body, make_services, make_settings, post


@pytest.fixture
def limited():
    rt = build_runtime(make_settings(max_concurrent_evaluations=1), services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        yield c, rt
    rt.shutdown()


def test_sync_evaluation_is_refused_with_429_when_the_server_is_full(limited):
    client, rt = limited
    assert rt.admission.try_enter()                     # simulate one evaluation already running
    r = post(client, body("READING", audio=b64_audio(), expected_text=EXPECTED))
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "5" and r.json()["request_id"] == "REQ_1"
    rt.admission.leave()
    assert post(client, body("READING", audio=b64_audio(), expected_text=EXPECTED)).status_code == 200


def test_slot_is_released_after_every_outcome(limited):
    client, rt = limited
    post(client, body("READING", audio=b64_audio(), expected_text=EXPECTED))           # success
    post(client, body("READING", audio="aGVsbG8=", expected_text=EXPECTED))            # 422 undecodable
    post(client, body("JUMBLED", expected_text="a b", question_config={"submitted_text": "a b"}))
    assert rt.admission.in_flight == 0


def test_async_endpoint_is_not_limited(limited):
    client, rt = limited
    assert rt.admission.try_enter()
    r = post(client, body("REPEAT", audio=b64_audio(), expected_text=EXPECTED), "/api/v1/evaluate/async")
    assert r.status_code == 202
    rt.admission.leave()


def test_responses_are_not_cacheable_and_have_security_headers(client):
    for r in (client.get("/api/v1/health"),
              post(client, body("JUMBLED", expected_text="a b", question_config={"submitted_text": "a b"}))):
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["Cache-Control"] == "no-store"
        assert r.headers["Referrer-Policy"] == "no-referrer"


def test_unauthorised_response_also_carries_headers(client):
    r = client.get("/api/v1/jobs/x")
    assert r.status_code == 401 and r.headers["Cache-Control"] == "no-store"
    assert client.get("/api/v1/jobs/x", headers=AUTH).status_code == 404
