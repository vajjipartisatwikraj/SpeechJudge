"""Section benchmark: the real application pipeline, end to end, through the HTTP API.

    python benchmarks/section_benchmark.py [--section all] [--repeat 1] [--base-url http://127.0.0.1:8000]

For each section type it sends 10 questions in ONE request:

    POST /api/v1/evaluate/section   ->  {"job_id": ..., "status": "QUEUED"}
    GET  /api/v1/jobs/<job_id>      ->  polled until COMPLETED / FAILED

Nothing bypasses the API: the time measured is queue + Whisper + phoneme + acoustic + Qwen + score
engine, exactly as a student's section would experience it. Run it ON THE SERVER (127.0.0.1) to leave
the network out of the number.

    reading          10 x ~10 s audio + expected text        Whisper, phoneme, acoustic, text comparison
    repeat           10 x ~10 s audio + expected text        same services, Repeat scoring
    jumbled          10 jumbled sentences (text answers)     text comparison only; --jumbled-mode spoken
                                                             sends the 10 audio files instead
    question_answer  10 x ~10 s audio + question             Whisper, acoustic, Qwen, score engine
    story_telling    10 x ~10 s audio + story prompt         Whisper, acoustic, Qwen, score engine

The API, a worker (or SJ_JOB_BACKEND=inline) and Redis (queue mode) must be running with all models
loaded. A short warm-up section per type is sent first and not counted (--no-warmup to skip).
Writes sections.<type> in benchmarks/benchmark_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from _common import (GpuMonitor, Clip, closest_to_median, load_clips, load_manifest, median, now_iso,
                     read_env_file, say, update_results)

SECTION_ORDER = ("reading", "repeat", "jumbled", "question_answer", "story_telling")
SECTION_TYPE = {"reading": "READING", "repeat": "REPEAT", "jumbled": "JUMBLED",
                "question_answer": "QUESTION_ANSWER", "story_telling": "STORYTELLING"}


class Api(Protocol):
    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]: ...


class HttpApi:
    """Plain urllib client; never raises on HTTP errors, returns (status, json)."""

    def __init__(self, base_url: str, token: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - operator-supplied URL
                return resp.status, _json(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _json(exc.read())
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return 0, {"detail": f"cannot reach {self.base_url}: {exc}"}


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw or b"{}")
        return value if isinstance(value, dict) else {"detail": str(value)}
    except ValueError:
        return {"detail": raw[:200].decode("utf-8", "replace")}


# ---- request payloads ------------------------------------------------------------------------------
def build_payload(name: str, clips: list[Clip], manifest: dict[str, Any], *, request_id: str,
                  jumbled_mode: str = "ui", count: int | None = None) -> dict[str, Any]:
    """The POST /evaluate/section body for one section type."""
    n = count or len(clips)
    questions: list[dict[str, Any]] = []
    if name == "jumbled" and jumbled_mode == "ui":
        for i, item in enumerate(manifest["jumbled"][:n], start=1):
            questions.append({"question_id": f"Q{i}", "expected_text": item["expected_text"],
                              "question_config": {"submitted_text": item["submitted_text"]}})
    else:
        for i, clip in enumerate(clips[:n], start=1):
            q: dict[str, Any] = {"question_id": f"Q{i}", "audio": clip.base64}
            if name in ("reading", "repeat", "jumbled"):
                q["expected_text"] = clip.text
            elif name == "question_answer":
                q["question"] = clip.question
            elif name == "story_telling":
                q["question"] = clip.story_prompt
            else:
                raise ValueError(f"unknown section '{name}'")
            questions.append(q)
    return {"request_id": request_id, "student_id": "BENCHMARK", "section_id": SECTION_TYPE[name],
            "questions": questions}


# ---- one timed run -----------------------------------------------------------------------------------
def run_once(api: Api, payload: dict[str, Any], *, timeout_s: float = 1800.0, poll_s: float = 0.25,
             clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep
             ) -> dict[str, Any]:
    """Submit one section and wait for its result. total_seconds = submit -> COMPLETED, as the caller sees it."""
    t0 = clock()
    status, body = api.request("POST", "/api/v1/evaluate/section", payload)
    if status != 202:
        return {"ok": False, "error": f"submit failed: HTTP {status} {body.get('detail', '')} {body.get('errors', '')}".strip()}
    job_id = body["job_id"]
    while True:
        status, job = api.request("GET", f"/api/v1/jobs/{job_id}")
        if status != 200:
            return {"ok": False, "error": f"job lookup failed: HTTP {status} {job.get('detail', '')}"}
        if job["status"] in ("COMPLETED", "FAILED"):
            break
        if clock() - t0 > timeout_s:
            return {"ok": False, "error": f"timed out after {timeout_s:.0f} s (status {job['status']})"}
        sleep(poll_s)
    total = clock() - t0
    if job["status"] == "FAILED":
        return {"ok": False, "error": f"job FAILED: {job.get('reason')}", "total_seconds": total}
    results = job.get("results") or []
    timing = job.get("timing") or {}
    return {"ok": True, "total_seconds": total, "server_processing_seconds": timing.get("processing_s"),
            "queued_seconds": timing.get("queued_s"), "questions": len(results),
            "questions_scored": job.get("questions_scored"), "section_score": job.get("section_score"),
            "statuses": sorted({r["status"] for r in results})}


def summarize(runs: list[dict[str, Any]], questions: int) -> dict[str, Any]:
    """Collapse the repeats of one section into the entry written to benchmark_results.json."""
    good = [r for r in runs if r.get("ok")]
    if not good:
        return {"questions": questions, "total_seconds": None, "average_question_seconds": None,
                "status": "failed", "error": runs[-1].get("error", "no successful run") if runs else "no run"}
    totals = [r["total_seconds"] for r in good]
    total = median(totals)
    server = [r["server_processing_seconds"] for r in good if r.get("server_processing_seconds") is not None]
    out = {
        "questions": questions,
        "total_seconds": round(total, 3),
        "average_question_seconds": round(total / questions, 3),
        "server_processing_seconds": round(median(server), 3) if server else None,
        "runs_total_seconds": [round(t, 3) for t in totals],
        "questions_scored": good[-1].get("questions_scored"),
        "section_score": good[-1].get("section_score"),
        "status": "ok" if len(good) == len(runs) else "partial",
        "measured_at": now_iso(),
    }
    if len(good) != len(runs):
        out["error"] = next(r["error"] for r in runs if not r.get("ok"))
    return out


# ---- the benchmark ---------------------------------------------------------------------------------------
def wait_until_ready(api: Api, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < end:
        status, body = api.request("GET", "/api/v1/ready")
        if status == 200:
            return True
        last = f"HTTP {status} {body.get('detail') or body}"
        time.sleep(2)
    say(f"not ready: {last}")
    return False


def benchmark_section(api: Api, name: str, clips: list[Clip], manifest: dict[str, Any], *, repeat: int,
                      warmup: bool, jumbled_mode: str, timeout_s: float) -> dict[str, Any]:
    stamp = time.strftime("%H%M%S")
    if warmup:
        warm = build_payload(name, clips, manifest, request_id=f"BENCH-WARM-{name}-{stamp}",
                             jumbled_mode=jumbled_mode, count=2)
        outcome = run_once(api, warm, timeout_s=timeout_s)
        say(f"  warm-up (2 questions): {outcome['total_seconds']:.1f} s" if outcome.get("ok")
            else f"  warm-up failed: {outcome.get('error')}")
    runs = []
    gpu_summaries = []
    for i in range(1, repeat + 1):
        payload = build_payload(name, clips, manifest, request_id=f"BENCH-{name}-{stamp}-{i}",
                                jumbled_mode=jumbled_mode)
        with GpuMonitor() as gpu:                     # only has data when run on the GPU machine itself
            outcome = run_once(api, payload, timeout_s=timeout_s)
        runs.append(outcome)
        gpu_summaries.append(gpu.summary())
        if outcome.get("ok"):
            say(f"  run {i}/{repeat}: {outcome['total_seconds']:.2f} s total, "
                f"{outcome['total_seconds'] / len(payload['questions']):.2f} s per question "
                f"(server {outcome.get('server_processing_seconds')} s, scored {outcome.get('questions_scored')}/"
                f"{len(payload['questions'])}, section score {outcome.get('section_score')})")
        else:
            say(f"  run {i}/{repeat}: FAILED - {outcome.get('error')}")
    result = summarize(runs, len(payload["questions"]))
    totals = [r.get("total_seconds", 0.0) for r in runs]
    if any(gpu_summaries) and result.get("status") != "failed":
        result.update(closest_to_median(totals, gpu_summaries))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default="all", choices=("all", *SECTION_ORDER))
    ap.add_argument("--repeat", type=int, default=1, help="timed runs per section (median reported)")
    ap.add_argument("--base-url", default=None, help="default: BACKEND_URL or http://127.0.0.1:<SJ_PORT>")
    ap.add_argument("--token", default=None, help="default: first SJ_SERVICE_CREDENTIALS value")
    ap.add_argument("--jumbled-mode", choices=("ui", "spoken"), default="ui",
                    help="ui: 10 text answers (text pipeline only); spoken: 10 audio files")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for one section")
    ap.add_argument("--ready-timeout", type=float, default=300.0, help="seconds to wait for /ready")
    ap.add_argument("--no-save", action="store_true", help="do not write benchmark_results.json")
    args = ap.parse_args()

    env = {**read_env_file(), **os.environ}
    base_url = args.base_url or env.get("BACKEND_URL") or f"http://127.0.0.1:{env.get('SJ_PORT', '8000')}"
    token = args.token or env.get("FRONTEND_API_KEY") or env.get("SJ_SERVICE_CREDENTIALS", "").split(",")[0].strip()
    if not token:
        say("ERROR: no credential. Set SJ_SERVICE_CREDENTIALS in speech-judge/.env or pass --token.")
        return 2

    api = HttpApi(base_url, token)
    say(f"API: {base_url}")
    if not wait_until_ready(api, args.ready_timeout):
        say("ERROR: the service is not ready (models still loading, or no worker is running).")
        return 2

    manifest = load_manifest()
    clips = load_clips(10, decode=False)
    names = list(SECTION_ORDER) if args.section == "all" else [args.section]
    def save_section(name: str, result: dict[str, Any]) -> None:
        def mutate(data: dict) -> None:
            data.setdefault("sections", {})[name] = result
        update_results(mutate)

    results: dict[str, dict[str, Any]] = {}
    for name in names:
        say(f"\n=== {name} ({'10 text answers' if name == 'jumbled' and args.jumbled_mode == 'ui' else '10 x ~10 s audio'}) ===")
        results[name] = benchmark_section(api, name, clips, manifest, repeat=args.repeat,
                                          warmup=not args.no_warmup, jumbled_mode=args.jumbled_mode,
                                          timeout_s=args.timeout)
        if not args.no_save:                 # saved as we go: a long run keeps what it has finished
            save_section(name, results[name])

    say("\n=== Section summary ===")
    say(f"{'section':16s} {'questions':>9s} {'total s':>9s} {'avg / question s':>17s}")
    for name, r in results.items():
        if r.get("status") == "failed":
            say(f"{name:16s} {r['questions']:9d} {'FAILED':>9s}   {r.get('error')}")
        else:
            say(f"{name:16s} {r['questions']:9d} {r['total_seconds']:9.2f} {r['average_question_seconds']:17.2f}")
    if not args.no_save:
        say("saved -> benchmarks/benchmark_results.json (sections)")
    return 1 if any(r.get("status") == "failed" for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
