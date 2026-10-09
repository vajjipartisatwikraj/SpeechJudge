"""Starts the benchmark scripts in ``benchmarks/`` on behalf of the (authenticated) API.

Safety: the caller never supplies a command. Only the scripts listed in ``SCRIPTS`` can run, with
arguments built here from validated enum values and a bounded integer, started without a shell. One
run at a time. Output goes to a log file under ``benchmarks/runs/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import ApiError
from app.core.paths import BACKEND_DIR

BENCH_DIR = BACKEND_DIR / "benchmarks"

SCRIPTS = {
    "whisper": "whisper_benchmark.py",
    "phoneme": "phoneme_benchmark.py",
    "qwen": "qwen_benchmark.py",
    "section": "section_benchmark.py",
}
SECTION_CHOICES = ("reading", "repeat", "jumbled", "question_answer", "story_telling", "all")
LOG_TAIL_BYTES = 24_000
KEEP_LOGS = 20

DESCRIPTIONS = {
    "whisper": "Whisper on the GPU: 10 clips one by one vs one batch of 10 (time, GPU utilisation, VRAM).",
    "phoneme": "Phoneme (pronunciation) model on the GPU: 10 clips one by one vs one batch of 10.",
    "qwen": "Qwen on a real Q&A and Storytelling evaluation prompt (latency, GPU utilisation, VRAM).",
    "section": "Real pipeline: POST /api/v1/evaluate/section with 10 x 10 s questions, for one section type or all.",
}


@dataclass
class Run:
    run_id: str
    benchmark: str
    section: str | None
    repeat: int
    log_path: Path
    started_at: float
    process: subprocess.Popen | None = field(default=None, repr=False)
    finished_at: float | None = None
    exit_code: int | None = None

    def poll(self) -> None:
        if self.process is not None and self.finished_at is None:
            code = self.process.poll()
            if code is not None:
                self.exit_code, self.finished_at = code, time.time()

    @property
    def status(self) -> str:
        if self.finished_at is None:
            return "running"
        return "completed" if self.exit_code == 0 else "failed"

    def describe(self, with_log: bool = True) -> dict:
        self.poll()
        end = self.finished_at or time.time()
        info = {"run_id": self.run_id, "benchmark": self.benchmark, "section": self.section,
                "repeat": self.repeat, "status": self.status, "exit_code": self.exit_code,
                "started_at": self.started_at, "elapsed_s": round(end - self.started_at, 1)}
        if with_log:
            info["log"] = tail_lines(self.log_path)
        return info


def tail_lines(path: Path, max_lines: int = 120) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.replace("\r\n", "\n").split("\n")[-max_lines:]


class BenchmarkManager:
    def __init__(self, *, bench_dir: Path = BENCH_DIR, python: str | None = None,
                 base_url: str = "http://127.0.0.1:8000") -> None:
        self.bench_dir = bench_dir
        self.python = python or sys.executable
        self.base_url = base_url
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    # ---- helpers ---------------------------------------------------------------------------
    def _command(self, benchmark: str, section: str, repeat: int) -> list[str]:
        cmd = [self.python, str(self.bench_dir / SCRIPTS[benchmark]), "--repeat", str(repeat)]
        if benchmark == "section":
            cmd += ["--section", section, "--base-url", self.base_url]
        return cmd

    def current(self) -> Run | None:
        for run in self._runs.values():
            run.poll()
            if run.status == "running":
                return run
        return None

    def latest(self) -> Run | None:
        return max(self._runs.values(), key=lambda r: r.started_at, default=None)

    def get(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise ApiError(404, "Benchmark run not found")
        return run

    def _prune_logs(self) -> None:
        logs = sorted((self.bench_dir / "runs").glob("*.log"), key=lambda p: p.stat().st_mtime)
        for old in logs[:-KEEP_LOGS]:
            old.unlink(missing_ok=True)

    # ---- start -------------------------------------------------------------------------------
    def start(self, benchmark: str, section: str = "all", repeat: int = 1) -> Run:
        if benchmark not in SCRIPTS:
            raise ApiError(422, f"Unknown benchmark '{benchmark}'", details=[f"one of: {', '.join(SCRIPTS)}"])
        if section not in SECTION_CHOICES:
            raise ApiError(422, f"Unknown section '{section}'", details=[f"one of: {', '.join(SECTION_CHOICES)}"])
        if not 1 <= repeat <= 10:
            raise ApiError(422, "repeat must be between 1 and 10")
        script = self.bench_dir / SCRIPTS[benchmark]
        if not script.is_file():
            raise ApiError(500, f"Benchmark script missing on the server: {script.name}")

        with self._lock:
            running = self.current()
            if running is not None:
                raise ApiError(409, f"A benchmark is already running ({running.benchmark}); wait for it to finish",
                               details=[f"run_id={running.run_id}"])
            run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            (self.bench_dir / "runs").mkdir(parents=True, exist_ok=True)
            log_path = self.bench_dir / "runs" / f"{run_id}-{benchmark}.log"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            with open(log_path, "wb") as log_file:
                process = subprocess.Popen(                      # list form, no shell
                    self._command(benchmark, section, repeat), cwd=str(self.bench_dir.parent),
                    stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            run = Run(run_id, benchmark, section if benchmark == "section" else None, repeat, log_path,
                      time.time(), process)
            self._runs[run_id] = run
            self._prune_logs()
            return run

    # ---- results file ----------------------------------------------------------------------
    def results(self) -> dict:
        try:
            return json.loads((self.bench_dir / "benchmark_results.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
