import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from app.services.mocks import MockAcousticEngine, MockGopt, MockLanguageEvaluator, MockSTT
from app.services.registry import ServiceRegistry
from tests.support import (EXPECTED_SENTENCE as EXPECTED, b64_audio, body, make_services, make_settings,
                           post)


def test_health_no_auth(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready(client):
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert {k: body[k] for k in ("status", "whisper", "gopt", "qwen", "acoustic")} == {
        "status": "ready", "whisper": True, "gopt": True, "qwen": True, "acoustic": True}
    assert set(body["mocked"]) == {"gopt", "qwen", "whisper"}   # the fixture mocks these three


def test_not_ready_and_evaluate_503():
    from app.services.registry import ServiceRegistry
    from app.services.mocks import MockAcousticEngine, MockGopt, MockLanguageEvaluator, MockSTT
    reg = ServiceRegistry(MockSTT(), MockGopt(), MockAcousticEngine(), MockLanguageEvaluator())
    rt = build_runtime(make_settings(), services=reg)
    app = create_app(runtime=rt, on_fatal=lambda: None)
    rt.services.gopt.load = lambda: time.sleep(5)  # never finishes during the test
    with TestClient(app) as c:
        r = c.get("/api/v1/ready")
        assert r.status_code == 503 and r.json()["status"] == "not_ready"
        r = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED))
        assert r.status_code == 503 and "gopt" in r.json()["detail"]
    rt.shutdown()


@pytest.mark.parametrize("method,path", [("post", "/api/v1/evaluate"),
                                         ("post", "/api/v1/evaluate/async"),
                                         ("get", "/api/v1/jobs/abc")])


def test_protected_endpoints_require_credential(client, method, path):
    kwargs = {"json": body("READING")} if method == "post" else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert getattr(client, method)(path, headers=bad, **kwargs).status_code == 401
    # malformed body must still be a 401, not a 422, when unauthenticated
    if method == "post":
        assert client.post(path, content=b"{not json").status_code == 401




def test_api_docs_are_disabled_by_default(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_api_docs_can_be_enabled():
    rt = build_runtime(make_settings(enable_docs=True), services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        assert c.get("/openapi.json").status_code == 200
    rt.shutdown()
