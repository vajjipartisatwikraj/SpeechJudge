import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app.core.benchmark_runner import BenchmarkManager
from app.core.errors import ApiError
from app.main import create_app
from app.runtime import build_runtime
from tests.support import AUTH, make_services, make_settings


def fake_scripts(directory, sleep_s=0.0):
    """Stand-ins for the real scripts: print their arguments, optionally stay alive for a while."""
    for name in ("whisper", "phoneme", "qwen", "section"):
        (directory / f"{name}_benchmark.py").write_text(
            "import sys, time\n"
            f"print('{name} args:', ' '.join(sys.argv[1:]), flush=True)\n"
            f"time.sleep({sleep_s})\n"
            "sys.exit(3 if 'FAIL' in sys.argv else 0)\n")
    (directory / "benchmark_results.json").write_text(json.dumps({"components": {"whisper": {"x": 1}}}))


def wait_done(manager, run_id, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        info = manager.get(run_id).describe()
        if info["status"] != "running":
            return info
        time.sleep(0.05)
    raise AssertionError("benchmark did not finish")


# ---- the manager ----------------------------------------------------------------------------------------
def test_manager_runs_a_script_and_captures_its_output(tmp_path):
    fake_scripts(tmp_path)
    mgr = BenchmarkManager(bench_dir=tmp_path, python=sys.executable, base_url="http://127.0.0.1:9")
    run = mgr.start("section", section="repeat", repeat=2)
    info = wait_done(mgr, run.run_id)
    assert info["status"] == "completed" and info["exit_code"] == 0
    assert any("section args: --repeat 2 --section repeat --base-url http://127.0.0.1:9" in line for line in info["log"])
    assert mgr.results()["components"]["whisper"] == {"x": 1}


def test_only_one_benchmark_runs_at_a_time(tmp_path):
    fake_scripts(tmp_path, sleep_s=1.5)
    mgr = BenchmarkManager(bench_dir=tmp_path, python=sys.executable)
    first = mgr.start("whisper")
    with pytest.raises(ApiError) as busy:
        mgr.start("phoneme")
    assert busy.value.status_code == 409
    wait_done(mgr, first.run_id)
    assert wait_done(mgr, mgr.start("phoneme").run_id)["status"] == "completed"


def test_component_scripts_get_no_section_arguments(tmp_path):
    fake_scripts(tmp_path)
    mgr = BenchmarkManager(bench_dir=tmp_path, python=sys.executable)
    info = wait_done(mgr, mgr.start("qwen", section="story_telling", repeat=1).run_id)
    assert any("qwen args: --repeat 1" in line and "--section" not in line for line in info["log"])


@pytest.mark.parametrize("kwargs", [
    {"benchmark": "rm -rf /"}, {"benchmark": "section", "section": "../../etc"},
    {"benchmark": "whisper", "repeat": 0}, {"benchmark": "whisper", "repeat": 99}])
def test_unknown_or_unsafe_arguments_are_refused(tmp_path, kwargs):
    fake_scripts(tmp_path)
    mgr = BenchmarkManager(bench_dir=tmp_path, python=sys.executable)
    with pytest.raises(ApiError) as err:
        mgr.start(**kwargs)
    assert err.value.status_code == 422


def test_failed_script_is_reported(tmp_path):
    fake_scripts(tmp_path)
    mgr = BenchmarkManager(bench_dir=tmp_path, python=sys.executable)
    (tmp_path / "qwen_benchmark.py").write_text("import sys\nprint('boom')\nsys.exit(3)\n")
    info = wait_done(mgr, mgr.start("qwen").run_id)
    assert info["status"] == "failed" and info["exit_code"] == 3 and "boom" in info["log"]


# ---- the HTTP API ----------------------------------------------------------------------------------------
@pytest.fixture
def bench_client(tmp_path):
    fake_scripts(tmp_path)
    rt = build_runtime(make_settings(benchmarks_enabled=True), services=make_services())
    app = create_app(runtime=rt)
    app.state.benchmarks = BenchmarkManager(bench_dir=tmp_path, python=sys.executable)
    with TestClient(app) as c:
        yield c
    rt.shutdown()


def test_disabled_by_default(client):
    assert client.post("/api/v1/benchmarks/run", json={"benchmark": "whisper"}, headers=AUTH).status_code == 403
    assert client.get("/api/v1/benchmarks/results", headers=AUTH).status_code == 403
    overview = client.get("/api/v1/benchmarks", headers=AUTH).json()
    assert overview["enabled"] is False and overview["current"] is None


def test_benchmark_endpoints_need_the_credential(client, bench_client):
    assert client.get("/api/v1/benchmarks").status_code == 401
    assert bench_client.post("/api/v1/benchmarks/run", json={"benchmark": "whisper"}).status_code == 401


def test_start_poll_and_read_results_over_http(bench_client):
    r = bench_client.post("/api/v1/benchmarks/run", json={"benchmark": "section", "section": "reading", "repeat": 1},
                          headers=AUTH)
    assert r.status_code == 202
    run = r.json()
    assert run["benchmark"] == "section" and run["section"] == "reading" and run["status"] in ("running", "completed")
    for _ in range(100):
        info = bench_client.get(f"/api/v1/benchmarks/runs/{run['run_id']}", headers=AUTH).json()
        if info["status"] != "running":
            break
        time.sleep(0.05)
    assert info["status"] == "completed"
    overview = bench_client.get("/api/v1/benchmarks", headers=AUTH).json()
    assert overview["enabled"] and overview["current"]["run_id"] == run["run_id"]
    assert {b["id"] for b in overview["benchmarks"]} == {"whisper", "phoneme", "qwen", "section"}
    assert overview["results"]["components"]["whisper"] == {"x": 1}
    assert bench_client.get("/api/v1/benchmarks/results", headers=AUTH).json() == overview["results"]


def test_bad_requests_get_422_and_unknown_run_404(bench_client):
    assert bench_client.post("/api/v1/benchmarks/run", json={"benchmark": "bash"}, headers=AUTH).status_code == 422
    assert bench_client.post("/api/v1/benchmarks/run", json={"benchmark": "section", "section": "x"},
                             headers=AUTH).status_code == 422
    assert bench_client.post("/api/v1/benchmarks/run", json={"benchmark": "whisper", "repeat": 50},
                             headers=AUTH).status_code == 422
    assert bench_client.get("/api/v1/benchmarks/runs/nope", headers=AUTH).status_code == 404
