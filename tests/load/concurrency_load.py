"""Concurrency step test against a RUNNING API.

For N = 1..max, fires N requests at the same instant and reports wall time, per-request latency,
throughput, HTTP 429 (back-pressure) count, CPU use and lowest free RAM of the machine running this
script (run it on the server itself for meaningful CPU/RAM numbers).

    python tests/load/concurrency_load.py                       # Repeat, N = 1..15, local API
    python tests/load/concurrency_load.py --type qa --max 6     # Qwen-backed type (slow!)
    python tests/load/concurrency_load.py --type exam --max 20  # mix 8/16/10/24/2 of the exam pattern
    python tests/load/concurrency_load.py --url https://judge.example.com --token <credential> --out run.json

Types: reading | repeat | jumbled | qa | story | exam.  Uses the synchronous /evaluate endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import psutil

from _common import credential_from_env_file, load_base64

TEXT = "The doctor suggested that the apple is good for your health."
QA_QUESTION = "What do you usually do on weekends?"
STORY_PROMPT = "Tell a story about a memorable holiday."
VOICES = ["ravi_correct", "heera_correct", "david_correct", "zira_correct"]
EXAM_PATTERN = ["READING"] * 8 + ["REPEAT"] * 16 + ["JUMBLED"] * 10 + ["QUESTION_ANSWER"] * 24 + ["STORYTELLING"] * 2

_audio_cache: dict[str, str] = {}


def audio(name: str) -> str:
    if name not in _audio_cache:
        _audio_cache[name] = load_base64(name)
    return _audio_cache[name]


def build_body(qtype: str, i: int) -> dict:
    voice = VOICES[i % len(VOICES)]
    base = {"request_id": f"load-{qtype.lower()}-{i}", "question_type": qtype}
    if qtype in ("READING", "REPEAT", "JUMBLED"):
        return {**base, "audio": audio(voice), "expected_text": TEXT}
    if qtype == "QUESTION_ANSWER":
        return {**base, "audio": audio("qa_good" if i % 2 == 0 else "qa_offtopic"), "question": QA_QUESTION}
    return {**base, "audio": audio("story"), "question": STORY_PROMPT}


def question_types(kind: str, n: int) -> list[str]:
    if kind == "exam":          # walk the exam pattern so every level sees the same mix
        return [EXAM_PATTERN[i % len(EXAM_PATTERN)] for i in range(n)]
    return [{"reading": "READING", "repeat": "REPEAT", "jumbled": "JUMBLED",
             "qa": "QUESTION_ANSWER", "story": "STORYTELLING"}[kind]] * n


def one(base_url: str, token: str, qtype: str, i: int) -> tuple[float, int, str]:
    req = urllib.request.Request(
        base_url + "/api/v1/evaluate", data=json.dumps(build_body(qtype, i)).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return time.perf_counter() - t, r.status, json.loads(r.read())["status"]
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t, exc.code, "HTTP_ERROR"
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t, 0, type(exc).__name__


def run_level(args, n: int) -> dict:
    cpu, free, stop = [], [], threading.Event()

    def sample():
        psutil.cpu_percent(None)
        while not stop.is_set():
            time.sleep(0.5)
            cpu.append(psutil.cpu_percent(None))
            free.append(psutil.virtual_memory().available / 1e9)

    threading.Thread(target=sample, daemon=True).start()
    types = question_types(args.type, n)
    barrier = threading.Barrier(n)

    def task(i):
        barrier.wait()                                 # release all N requests together
        return one(args.url, args.token, types[i], i)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(task, range(n)))
    wall = time.perf_counter() - t0
    stop.set()
    lat = [r[0] for r in results]
    return {"n": n, "wall_s": wall, "min_s": min(lat), "avg_s": statistics.mean(lat), "max_s": max(lat),
            "ok": sum(1 for r in results if r[1] == 200 and r[2] == "COMPLETED"),
            "rejected_429": sum(1 for r in results if r[1] == 429),
            "errors": sum(1 for r in results if r[1] not in (200, 429)),
            "jobs_per_min": n / wall * 60,
            "cpu_avg": statistics.mean(cpu) if cpu else 0.0, "cpu_peak": max(cpu) if cpu else 0.0,
            "min_free_gb": min(free) if free else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("SJ_LOAD_TOKEN") or credential_from_env_file())
    ap.add_argument("--type", default="repeat", choices=["reading", "repeat", "jumbled", "qa", "story", "exam"])
    ap.add_argument("--max", type=int, default=15, help="highest concurrency level")
    ap.add_argument("--out", help="write the results as JSON to this file")
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("No credential: pass --token, set SJ_LOAD_TOKEN, or create speech-judge/.env")

    print(f"target {args.url}  type={args.type}  levels 1..{args.max}")
    print("warm-up ...", flush=True)
    one(args.url, args.token, question_types(args.type, 1)[0], 0)
    print(f"{'N':>3} {'ok':>6} {'429':>4} {'err':>4} {'wall s':>7} {'min s':>6} {'avg s':>6} {'max s':>6} "
          f"{'jobs/min':>9} {'cpu avg%':>8} {'cpu pk%':>7} {'free GB':>7}", flush=True)
    rows = []
    for n in range(1, args.max + 1):
        r = run_level(args, n)
        rows.append(r)
        print(f"{n:>3} {r['ok']:>3}/{n:<2} {r['rejected_429']:>4} {r['errors']:>4} {r['wall_s']:>7.1f} "
              f"{r['min_s']:>6.1f} {r['avg_s']:>6.1f} {r['max_s']:>6.1f} {r['jobs_per_min']:>9.1f} "
              f"{r['cpu_avg']:>8.0f} {r['cpu_peak']:>7.0f} {r['min_free_gb']:>7.1f}", flush=True)
        time.sleep(2)
    single = rows[0]["max_s"]
    print(f"\nsingle-request latency: {single:.1f} s")
    no_wait = [r["n"] for r in rows if r["max_s"] <= single * 1.25 and r["errors"] == 0]
    print(f"largest N whose slowest request stays within 1.25x of single: {max(no_wait) if no_wait else 1}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args) | {"token": "***"}, "rows": rows}, fh, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
