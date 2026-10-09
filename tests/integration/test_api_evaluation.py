import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from tests.support import (EXPECTED_SENTENCE as EXPECTED, SR, b64_audio, body, make_services,
                           make_settings, make_speech, post)

KEYS = {"transcription", "gopt", "acoustic", "text_comparison", "language", "score_components"}


def check_envelope(data, qt):
    assert data["request_id"] == "REQ_1" and data["question_type"] == qt
    assert set(data["evaluation"]) == KEYS
    assert data["raw_metrics"] is not None  # raw metrics reported separately from the score


def test_reading(client, services):
    r = post(client, body("READING", audio=b64_audio(), expected_text=EXPECTED))
    assert r.status_code == 200
    d = r.json()
    check_envelope(d, "READING")
    assert d["status"] == "COMPLETED" and 0.0 <= d["score"] <= 1.0
    ev = d["evaluation"]
    assert ev["gopt"] and ev["acoustic"] and ev["text_comparison"] and ev["language"] is None
    assert ev["acoustic"]["rate_source"] == "transcript"
    assert ev["acoustic"]["original_duration_s"] > 0
    assert services.language.calls == 0
    total = sum(c["weighted_value"] for c in ev["score_components"])
    assert total == pytest.approx(d["score"], abs=1e-4)


def test_repeat(client, services):
    r = post(client, body("REPEAT", audio=b64_audio(), expected_text=EXPECTED))
    assert r.status_code == 200 and r.json()["status"] == "COMPLETED"
    assert services.gopt.calls == 1 and services.language.calls == 0


def test_jumbled_ui_only_text_comparison(client, services):
    r = post(client, body("JUMBLED", expected_text="the cat sat",
                          question_config={"submitted_text": ["the", "cat", "sat"]}))
    d = r.json()
    assert r.status_code == 200 and d["score"] == pytest.approx(1.0)
    ev = d["evaluation"]
    assert ev["text_comparison"] and not any(ev[k] for k in ("transcription", "gopt", "acoustic",
                                                             "language"))
    assert services.stt.calls == 0 and services.gopt.calls == 0 and services.language.calls == 0


def test_jumbled_ui_wrong_order_scores_lower(client):
    r = post(client, body("JUMBLED", expected_text="the cat sat",
                          question_config={"submitted_text": "sat the cat"}))
    assert r.json()["score"] < 1.0


def test_jumbled_spoken(client, services):
    r = post(client, body("JUMBLED", audio=b64_audio(), expected_text=EXPECTED))
    d = r.json()
    assert r.status_code == 200 and d["status"] == "COMPLETED"
    assert services.stt.calls == 1 and services.gopt.calls == 0 and services.language.calls == 0
    assert d["evaluation"]["acoustic"]


@pytest.mark.parametrize("qt", ["QUESTION_ANSWER", "STORYTELLING"])
def test_open_ended(client, services, qt):
    r = post(client, body(qt, audio=b64_audio(), question="Describe your weekend."))
    d = r.json()
    assert r.status_code == 200 and d["status"] == "COMPLETED"
    ev = d["evaluation"]
    assert ev["language"] and ev["acoustic"] and ev["text_comparison"] is None and ev["gopt"] is None
    assert services.gopt.calls == 0 and services.language.calls == 1
    assert services.language.last_args[1] == "Describe your weekend."


def test_language_gets_no_acoustic_data(client, services):
    post(client, body("QUESTION_ANSWER", audio=b64_audio(), question="Q?"))
    assert len(services.language.last_args) == 3  # (type, question, transcript) only


# --- audio quality gate end to end ---------------------------------------------------------


def test_low_audio_quality_short(client, services):
    r = post(client, body("READING", audio=b64_audio(make_speech(0.4, None)),
                          expected_text=EXPECTED))
    d = r.json()
    assert r.status_code == 200 and d["status"] == "LOW_AUDIO_QUALITY" and d["score"] is None
    assert d["reason"] == "Audio too short"
    assert services.stt.calls == 0 and services.gopt.calls == 0  # nothing downstream ran


def test_low_audio_quality_silence(client, services):
    silence = (0.0005 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32)
    d = post(client, body("QUESTION_ANSWER", audio=b64_audio(silence), question="Q?")).json()
    assert d["status"] == "LOW_AUDIO_QUALITY" and d["reason"] == "Insufficient speech signal"
    assert services.language.calls == 0


def test_noise_flag_propagates_and_scoring_continues(client):
    d = post(client, body("READING", audio=b64_audio(make_speech(noise=0.04)),
                          expected_text=EXPECTED)).json()
    assert d["status"] == "COMPLETED" and "EXCESSIVE_NOISE" in d["flags"]


def test_empty_transcript_flag():
    rt = build_runtime(make_settings(), services=make_services(stt_text=""))
    with TestClient(create_app(runtime=rt)) as c:
        d = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED)).json()
        assert d["status"] == "COMPLETED" and "EMPTY_TRANSCRIPT" in d["flags"]
        assert d["evaluation"]["text_comparison"]["word_accuracy"] == 0.0
        d = post(c, body("QUESTION_ANSWER", audio=b64_audio(), question="Q?")).json()
        assert "EMPTY_TRANSCRIPT" in d["flags"] and d["score"] is not None
    rt.shutdown()


# --- failures ------------------------------------------------------------------------------


def test_service_error_gives_failed_result_without_stack_trace():
    services = make_services()

    def boom(*_a, **_k):
        raise RuntimeError("secret internal path /opt/models")
    services.gopt.assess = boom
    rt = build_runtime(make_settings(), services=services)
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, body("READING", audio=b64_audio(), expected_text=EXPECTED))
        d = r.json()
        assert r.status_code == 200 and d["status"] == "FAILED" and d["score"] is None
        assert "gopt" in d["reason"] and "secret" not in r.text and "Traceback" not in r.text
    rt.shutdown()


def test_sync_timeout_504():
    services = make_services()
    services.stt.transcribe = lambda a: time.sleep(1.0)
    rt = build_runtime(make_settings(request_timeout_s=0.2), services=services)
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, body("REPEAT", audio=b64_audio(), expected_text=EXPECTED))
        assert r.status_code == 504 and r.json()["request_id"] == "REQ_1"
    rt.shutdown()

