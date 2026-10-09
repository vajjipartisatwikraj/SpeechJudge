import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from tests.support import (AUTH, EXPECTED_SENTENCE as EXPECTED, b64_audio, make_services,
                           make_settings, post)

SECTION = "/api/v1/evaluate/section"


def wait_for_job(client, job_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json()
        if d["status"] in ("COMPLETED", "FAILED"):
            return d
        time.sleep(0.02)
    raise AssertionError("section job did not finish")


def section_body(section_id="REPEAT", n=10, **question_fields):
    audio = b64_audio()
    questions = [{"question_id": f"Q{i}", "audio": audio, "expected_text": EXPECTED, **question_fields}
                 for i in range(1, n + 1)]
    return {"request_id": "REQ_001", "student_id": "STU_001", "section_id": section_id,
            "questions": questions}


def test_section_flow_returns_queued_then_results_for_every_question(client):
    r = post(client, section_body(), SECTION)
    assert r.status_code == 202
    sub = r.json()
    assert sub["status"] == "QUEUED" and sub["request_id"] == "REQ_001" and sub["job_id"]

    job = wait_for_job(client, sub["job_id"])
    assert job["status"] == "COMPLETED"
    assert job["section_id"] == "REPEAT" and job["student_id"] == "STU_001"
    assert job["questions_total"] == 10 and job["questions_scored"] == 10
    assert [r["question_id"] for r in job["results"]] == [f"Q{i}" for i in range(1, 11)]
    scores = [r["score"] for r in job["results"]]
    assert all(r["status"] == "COMPLETED" and 0 <= r["score"] <= 1 for r in job["results"])
    assert job["section_score"] == pytest.approx(sum(scores) / 10, abs=1e-6)
    t = job["timing"]
    assert t["processing_s"] is not None and t["total_s"] >= t["processing_s"] >= 0
    assert job["results"][0]["detail"] is None            # details only on request


def test_question_scores_equal_single_question_scores(client):
    single = post(client, {"request_id": "S", "question_type": "REPEAT", "audio": b64_audio(),
                           "expected_text": EXPECTED}).json()
    job = wait_for_job(client, post(client, section_body(n=3), SECTION).json()["job_id"])
    assert all(r["score"] == single["score"] for r in job["results"])


def test_include_details_adds_the_full_evaluation(client):
    body = section_body(n=2)
    body["include_details"] = True
    job = wait_for_job(client, post(client, body, SECTION).json()["job_id"])
    detail = job["results"][0]["detail"]
    assert detail["request_id"] == "REQ_001:Q1" and detail["evaluation"]["transcription"]["text"]


def test_whisper_and_phoneme_receive_the_whole_section_as_one_batch():
    services = make_services()
    stt_sizes, gopt_sizes = [], []
    real_stt, real_gopt = services.stt.transcribe_batch, services.gopt.assess_batch
    services.stt.transcribe_batch = lambda audios: (stt_sizes.append(len(audios)), real_stt(audios))[1]
    services.gopt.assess_batch = lambda items: (gopt_sizes.append(len(items)), real_gopt(items))[1]
    # batching is off in the CPU profile (the test default), so ask for the GPU profile's batch size of 10
    rt = build_runtime(make_settings(whisper_batch_size=10, phoneme_batch_size=10), services=services)
    with TestClient(create_app(runtime=rt)) as c:
        job = wait_for_job(c, post(c, section_body(n=10), SECTION).json()["job_id"])
        assert job["status"] == "COMPLETED" and job["questions_scored"] == 10
    assert stt_sizes == [10] and gopt_sizes == [10]
    assert rt.stages.stats()["whisper"]["max_batch"] == 10
    rt.shutdown()


def test_batch_size_one_disables_batching():
    services = make_services()
    sizes = []
    real = services.stt.transcribe_batch
    services.stt.transcribe_batch = lambda audios: (sizes.append(len(audios)), real(audios))[1]
    rt = build_runtime(make_settings(whisper_batch_size=1), services=services)
    with TestClient(create_app(runtime=rt)) as c:
        wait_for_job(c, post(c, section_body(n=4), SECTION).json()["job_id"])
    assert sizes == [1, 1, 1, 1]
    rt.shutdown()


def test_question_answer_section_uses_language_scoring(client):
    body = section_body("question_answer", n=3, question="Describe your weekend.")
    for q in body["questions"]:
        del q["expected_text"]
    job = wait_for_job(client, post(client, body, SECTION).json()["job_id"])
    assert job["section_id"] == "QUESTION_ANSWER" and job["questions_scored"] == 3


def test_jumbled_ui_section_needs_no_audio(client):
    body = {"request_id": "R", "section_id": "JUMBLED", "questions": [
        {"question_id": "A", "expected_text": "the cat sat on the mat",
         "question_config": {"submitted_text": "the cat sat on the mat"}},
        {"question_id": "B", "expected_text": "the cat sat on the mat",
         "question_config": {"submitted_text": "mat the on sat cat the"}}]}
    job = wait_for_job(client, post(client, body, SECTION).json()["job_id"])
    a, b = job["results"]
    assert a["score"] > 0.999 and b["score"] < a["score"] and job["questions_scored"] == 2


def test_one_unreadable_question_fails_alone(client):
    body = section_body(n=3)
    body["questions"][1]["audio"] = "aGVsbG8gd29ybGQ="            # valid base64, not audio
    job = wait_for_job(client, post(client, body, SECTION).json()["job_id"])
    statuses = {r["question_id"]: r["status"] for r in job["results"]}
    assert job["status"] == "COMPLETED" and statuses == {"Q1": "COMPLETED", "Q2": "FAILED", "Q3": "COMPLETED"}
    assert job["questions_scored"] == 2 and job["section_score"] is not None
    assert job["results"][1]["score"] is None and job["results"][1]["reason"]


def test_quality_gate_applies_per_question(client):
    from tests.support import make_speech
    body = section_body(n=2)
    body["questions"][0]["audio"] = b64_audio(make_speech(0.3, None))      # too short
    job = wait_for_job(client, post(client, body, SECTION).json()["job_id"])
    assert [r["status"] for r in job["results"]] == ["LOW_AUDIO_QUALITY", "COMPLETED"]
    assert job["questions_scored"] == 1


@pytest.mark.parametrize("mutate,fragment", [
    (lambda b: b.update(questions=[]), "questions"),
    (lambda b: b["questions"][1].update(question_id="Q1"), "duplicate question_id"),
    (lambda b: b["questions"][0].pop("audio"), "audio is required"),
    (lambda b: b["questions"][0].update(question_id="  "), "question_id"),
    (lambda b: b.update(section_id="DANCING"), "section_id"),
    (lambda b: b.update(request_id=" "), "request_id"),
])
def test_invalid_sections_are_rejected_with_422(client, mutate, fragment):
    body = section_body(n=3)
    mutate(body)
    r = post(client, body, SECTION)
    assert r.status_code == 422
    assert fragment in " ".join(r.json()["errors"] + [r.json()["detail"]])


def test_too_many_questions_is_422():
    rt = build_runtime(make_settings(section_max_questions=3), services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, section_body(n=4), SECTION)
        assert r.status_code == 422 and "at most 3" in r.json()["detail"]
    rt.shutdown()


def test_invalid_base64_audio_is_rejected_at_submit(client):
    body = section_body(n=2)
    body["questions"][0]["audio"] = "!!!"
    assert post(client, body, SECTION).status_code in (413, 422)


def test_section_accepts_more_audio_than_one_question_allows():
    # 10 clips of ~130 KB: far above a 400 KB single-question limit, fine for the section budget
    rt = build_runtime(make_settings(max_audio_bytes=400_000, section_max_total_audio_bytes=5_000_000),
                       services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        r = post(c, section_body(n=10), SECTION)
        assert r.status_code == 202
        assert wait_for_job(c, r.json()["job_id"])["questions_scored"] == 10
    rt.shutdown()


def test_section_requires_credentials(client):
    assert client.post(SECTION, json=section_body(n=1)).status_code == 401
    assert client.post(SECTION, json=section_body(n=1), headers={"Authorization": "Bearer nope"}).status_code == 401


def test_existing_single_question_job_response_is_unchanged(client):
    r = post(client, {"request_id": "R1", "question_type": "REPEAT", "audio": b64_audio(),
                      "expected_text": EXPECTED}, "/api/v1/evaluate/async")
    job = wait_for_job(client, r.json()["job_id"])
    assert set(job) == {"job_id", "request_id", "status", "result", "reason"} and job["result"]["score"] is not None
