"""End to end through the API: disabled Pronunciation / Qwen return mock values and say so."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from app.services.acoustic.engine import NumpyAcousticEngine
from app.services.mocks import MockSTT
from app.services.registry import ServiceRegistry, _build_language, _build_pronunciation
from tests.support import EXPECTED_SENTENCE as EXPECTED, b64_audio, body, make_services, make_settings, post


def runtime_with(**flags):
    """Real acoustic engine + mocked STT; pronunciation / Qwen built by the production switches."""
    assert not flags.get("pronunciation_enabled", True) and not flags.get("qwen_enabled", True), \
        "enabled switches would try to load the real models"
    settings = make_settings(mock_mode=False, **flags)
    reg = ServiceRegistry(MockSTT(EXPECTED.lower().rstrip(".")), _build_pronunciation(settings),
                          NumpyAcousticEngine(), _build_language(settings))
    reg.load_all()
    return build_runtime(settings, services=reg)


@pytest.fixture
def client_off():
    rt = runtime_with(pronunciation_enabled=False, qwen_enabled=False)
    with TestClient(create_app(runtime=rt)) as c:
        yield c
    rt.shutdown()


def test_reading_uses_mock_pronunciation_and_is_flagged(client_off):
    d = post(client_off, body("READING", audio=b64_audio(), expected_text=EXPECTED)).json()
    assert d["status"] == "COMPLETED" and 0.0 <= d["score"] <= 1.0
    gopt = d["evaluation"]["gopt"]
    assert gopt["method"] == "mock" and gopt["pronunciation"] == 0.82     # the fixed placeholder
    assert "MOCKED_PRONUNCIATION" in d["flags"]
    assert "MOCKED_LANGUAGE" not in d["flags"] and "MOCKED_ACOUSTIC" not in d["flags"]


@pytest.mark.parametrize("qt", ["QUESTION_ANSWER", "STORYTELLING"])
def test_open_ended_uses_mock_language_and_is_flagged(client_off, qt):
    d = post(client_off, body(qt, audio=b64_audio(), question="Describe your weekend.")).json()
    assert d["status"] == "COMPLETED"
    language = d["evaluation"]["language"]
    assert language["prompt_version"] == "mock"
    assert all(v["score"] == 0.8 for v in language["dimensions"].values())
    assert "MOCKED_LANGUAGE" in d["flags"] and "MOCKED_PRONUNCIATION" not in d["flags"]


def test_question_types_that_need_neither_service_are_not_flagged(client_off):
    d = post(client_off, body("JUMBLED", expected_text="the cat sat",
                              question_config={"submitted_text": "the cat sat"})).json()
    assert d["score"] == pytest.approx(1.0)
    assert not any(f.startswith("MOCKED_") for f in d["flags"])


def test_ready_reports_which_services_are_mocked(client_off):
    r = client_off.get("/api/v1/ready").json()
    assert r["status"] == "ready" and {"gopt", "qwen"} <= set(r["mocked"])


def test_enabled_services_are_not_flagged():
    """With the switches on (a non-mock service), no MOCKED_PRONUNCIATION / MOCKED_LANGUAGE appear."""
    services = make_services()
    services.gopt.mocked = False          # pretend these are the real (enabled) implementations
    services.language.mocked = False
    rt = build_runtime(make_settings(), services=services)
    with TestClient(create_app(runtime=rt)) as c:
        d = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED)).json()
        assert "MOCKED_PRONUNCIATION" not in d["flags"] and "MOCKED_STT" in d["flags"]
        d = post(c, body("QUESTION_ANSWER", audio=b64_audio(), question="Q?")).json()
        assert "MOCKED_LANGUAGE" not in d["flags"]
    rt.shutdown()