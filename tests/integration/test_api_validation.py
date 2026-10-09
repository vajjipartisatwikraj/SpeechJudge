import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from tests.support import EXPECTED_SENTENCE as EXPECTED, b64_audio, body, make_services, make_settings, post



@pytest.mark.parametrize("payload,missing", [
    (body("READING", expected_text=EXPECTED), ["audio"]),
    (body("REPEAT", audio="x"), ["expected_text"]),
    (body("READING"), ["audio", "expected_text"]),
    (body("QUESTION_ANSWER", audio="x"), ["question"]),
    (body("STORYTELLING"), ["audio", "question"]),
    (body("JUMBLED"), ["expected_text"]),
])


def test_missing_fields_422_names_each_field(client, payload, missing):
    r = post(client, payload)
    assert r.status_code == 422
    assert r.json()["request_id"] == "REQ_1"
    for m in missing:
        assert m in r.json()["detail"]


def test_jumbled_needs_one_input_mode(client):
    r = post(client, body("JUMBLED", expected_text="a b c"))
    assert r.status_code == 422 and "input mode" in r.json()["detail"]


def test_invalid_question_type_and_empty_request_id(client):
    r = post(client, body("NOPE"))
    assert r.status_code == 422 and r.json()["request_id"] == "REQ_1"
    r = post(client, {"request_id": " ", "question_type": "READING"})
    assert r.status_code == 422


def test_undecodable_audio_422(client):
    r = post(client, body("READING", audio="aGVsbG8gd29ybGQ=", expected_text=EXPECTED))
    assert r.status_code == 422 and r.json()["request_id"] == "REQ_1"
    assert "undecodable" in r.json()["detail"] or "Unsupported" in r.json()["detail"]


def test_audio_too_large_413():
    rt = build_runtime(make_settings(max_audio_bytes=1000), services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED))
        assert r.status_code == 413 and "limit" in r.json()["detail"].lower()
    rt.shutdown()


def test_audio_too_long_413():
    rt = build_runtime(make_settings(max_audio_seconds=1.5), services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED))
        assert r.status_code == 413
    rt.shutdown()

