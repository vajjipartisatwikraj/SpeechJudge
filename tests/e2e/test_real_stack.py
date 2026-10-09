"""Through the running stack (frontend proxy :8080 or the backend itself) with voice clips.

Start the backend and `python frontend/server.py` first, then:  pytest --real tests/e2e/test_real_stack.py
Override the target with SJ_E2E_URL (if pointing at the backend directly, the request needs its own
Authorization header: set SJ_E2E_TOKEN).
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

import pytest

from tests.support import fixture_wav

CORRECT = "The doctor suggested that the apple is good for your health."


def _call(stack_url, method, path, body=None):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SJ_E2E_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['SJ_E2E_TOKEN']}"
    req = urllib.request.Request(stack_url + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:  # noqa: S310 - local test URLs
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _audio(name):
    path = fixture_wav(name)
    if not path.exists():
        pytest.skip(f"missing fixture {path.name}")
    return base64.b64encode(path.read_bytes()).decode()


@pytest.fixture
def evaluate(stack_url):
    def _evaluate(qt, clip, **extra):
        status, data = _call(stack_url, "POST", "/api/v1/evaluate",
                             {"request_id": f"E2E-{clip}", "question_type": qt, "audio": _audio(clip), **extra})
        assert status == 200, data
        return data
    return _evaluate


def test_reading_ordering_correct_truncated_wrong(evaluate):
    correct = evaluate("READING", "ravi_correct", expected_text=CORRECT)
    truncated = evaluate("READING", "ravi_truncated", expected_text=CORRECT)
    wrong = evaluate("READING", "ravi_question", expected_text=CORRECT)
    assert correct["status"] == truncated["status"] == wrong["status"] == "COMPLETED"
    assert correct["score"] > truncated["score"] > wrong["score"]
    assert correct["score"] > 0.8
    gopt = correct["evaluation"]["gopt"]
    assert gopt["method"] == "phoneme_ctc" and gopt["details"]["accent_profile"] == "indian_english"


def test_result_exposes_fluency_and_prosody_measures(evaluate):
    acoustic = evaluate("REPEAT", "heera_correct", expected_text=CORRECT)["evaluation"]["acoustic"]
    for key in ("speech_rate_wpm", "mean_run_length", "pitch_std_st", "pitch_range_st",
                "syllable_energy_std_db", "terminal_pitch_delta_st"):
        assert key in acoustic


def test_spoken_jumbled_uses_no_pronunciation(evaluate):
    data = evaluate("JUMBLED", "ravi_correct", expected_text=CORRECT)
    assert data["evaluation"]["gopt"] is None and data["score"] > 0.8


def test_open_ended_async_job_completes(stack_url):
    body = {"request_id": "E2E-ASYNC", "question_type": "QUESTION_ANSWER", "audio": _audio("qa_good"),
            "question": "What do you usually do on weekends?"}
    status, sub = _call(stack_url, "POST", "/api/v1/evaluate/async", body)
    assert status == 202
    deadline = time.time() + 600
    while time.time() < deadline:
        _, job = _call(stack_url, "GET", f"/api/v1/jobs/{sub['job_id']}")
        if job["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(1)
    assert job["status"] == "COMPLETED"
    assert job["result"]["score"] > 0.6 and job["result"]["evaluation"]["language"]
