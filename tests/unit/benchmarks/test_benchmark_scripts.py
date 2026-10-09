"""The benchmark helpers and the section benchmark, exercised against the mock service (no GPU)."""
import json
import sys
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import build_runtime
from tests.support import AUTH, make_services, make_settings

BENCH = Path(__file__).resolve().parents[3] / "benchmarks"
AUDIO = Path(__file__).resolve().parents[3] / "benchmark_audio"
sys.path.insert(0, str(BENCH))

import _common  # noqa: E402
import section_benchmark as sb  # noqa: E402


# ---- benchmark audio -------------------------------------------------------------------------------
def test_ten_benchmark_clips_of_about_ten_seconds():
    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 10 and len(manifest["jumbled"]) == 10
    for item in manifest["items"]:
        assert item["text"] and item["question"] and item["story_prompt"]
        with wave.open(str(AUDIO / item["file"])) as w:
            seconds = w.getnframes() / w.getframerate()
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert 9.0 <= seconds <= 11.0, f"{item['file']} is {seconds:.1f} s"


def test_results_template_has_the_documented_shape():
    data = json.loads((BENCH / "benchmark_results.json").read_text(encoding="utf-8"))
    assert set(data) >= {"gpu", "components", "sections"}
    assert set(data["components"]) == {"whisper", "phoneme", "qwen"}
    assert set(data["sections"]) == {"reading", "repeat", "jumbled", "question_answer", "story_telling"}
    assert all(s["questions"] == 10 for s in data["sections"].values())


# ---- helpers ------------------------------------------------------------------------------------------
def test_parse_smi_line():
    assert _common.parse_smi_line("37, 4521\n") == (37.0, 4521.0)
    assert _common.parse_smi_line("[N/A], 4521") is None and _common.parse_smi_line("") is None


def test_gpu_monitor_summary_from_a_fake_reader():
    values = iter([(10.0, 1000.0), (90.0, 5000.0), (50.0, 3000.0)] + [(20.0, 2000.0)] * 1000)
    with _common.GpuMonitor(interval_s=0.001, reader=lambda: next(values)) as gpu:
        import time
        time.sleep(0.05)
    s = gpu.summary()
    assert s["gpu_util_peak_percent"] == 90.0 and s["vram_peak_mb"] == 5000 and 10 <= s["gpu_util_avg_percent"] <= 90


def test_gpu_monitor_without_a_gpu_is_silent():
    with _common.GpuMonitor(reader=None) as gpu:
        pass
    assert gpu.summary() == {} or gpu.available


def test_update_results_merges_and_keeps_other_entries(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_common.default_results()))
    _common.update_results(lambda d: d["components"].__setitem__("whisper", {"batch_10_audio_seconds": 1.5}),
                           path=path, record_gpu=False)
    data = json.loads(path.read_text())
    assert data["components"]["whisper"] == {"batch_10_audio_seconds": 1.5}
    assert data["components"]["qwen"] == {"average_latency_seconds": 0} and "updated_at" in data
    assert not list(tmp_path.glob(".results-*"))


def test_closest_to_median_picks_the_matching_run():
    assert _common.closest_to_median([5.0, 1.0, 3.0], ["a", "b", "c"]) == "c"


# ---- section benchmark ---------------------------------------------------------------------------------
class ClientApi:
    def __init__(self, client):
        self.client = client

    def request(self, method, path, body=None):
        r = self.client.request(method, path, json=body, headers=AUTH)
        return r.status_code, r.json()


@pytest.fixture(scope="module")
def clips_and_manifest():
    return _common.load_clips(10, decode=False), _common.load_manifest()


@pytest.mark.parametrize("name,fields", [
    ("reading", {"audio", "expected_text"}), ("repeat", {"audio", "expected_text"}),
    ("question_answer", {"audio", "question"}), ("story_telling", {"audio", "question"}),
    ("jumbled", {"expected_text", "question_config"})])
def test_payloads(clips_and_manifest, name, fields):
    clips, manifest = clips_and_manifest
    payload = sb.build_payload(name, clips, manifest, request_id="R")
    assert payload["section_id"] == sb.SECTION_TYPE[name] and len(payload["questions"]) == 10
    assert fields <= set(payload["questions"][0])
    assert ("audio" in payload["questions"][0]) == ("audio" in fields)       # jumbled (ui) sends no audio
    assert len({q["question_id"] for q in payload["questions"]}) == 10


def test_spoken_jumbled_sends_audio(clips_and_manifest):
    clips, manifest = clips_and_manifest
    q = sb.build_payload("jumbled", clips, manifest, request_id="R", jumbled_mode="spoken")["questions"][0]
    assert q["audio"] and q["expected_text"] == clips[0].text


@pytest.fixture
def api():
    rt = build_runtime(make_settings(batch_max_wait_ms=500), services=make_services())
    with TestClient(create_app(runtime=rt)) as c:
        yield ClientApi(c)
    rt.shutdown()


@pytest.mark.parametrize("name", sb.SECTION_ORDER)
def test_every_section_runs_through_the_real_api(api, clips_and_manifest, name):
    clips, manifest = clips_and_manifest
    payload = sb.build_payload(name, clips, manifest, request_id=f"BENCH-{name}")
    run = sb.run_once(api, payload, poll_s=0.01)
    assert run["ok"], run
    assert run["questions"] == 10 and run["questions_scored"] == 10 and run["total_seconds"] > 0
    summary = sb.summarize([run], 10)
    assert summary["questions"] == 10 and summary["status"] == "ok"
    assert summary["average_question_seconds"] == pytest.approx(summary["total_seconds"] / 10, abs=1e-3)


def test_rejected_submit_is_reported_not_raised(api, clips_and_manifest):
    clips, manifest = clips_and_manifest
    payload = sb.build_payload("repeat", clips, manifest, request_id="R")
    payload["questions"][0].pop("audio")
    run = sb.run_once(api, payload, poll_s=0.01)
    assert not run["ok"] and "422" in run["error"]
    failed = sb.summarize([run], 10)
    assert failed["status"] == "failed" and failed["total_seconds"] is None


def test_summary_uses_the_median_of_the_repeats():
    runs = [{"ok": True, "total_seconds": t, "server_processing_seconds": t - 0.1, "questions_scored": 10,
             "section_score": 0.8} for t in (4.0, 2.0, 3.0)]
    s = sb.summarize(runs, 10)
    assert s["total_seconds"] == 3.0 and s["average_question_seconds"] == 0.3 and s["runs_total_seconds"] == [4.0, 2.0, 3.0]
