"""Shared helpers for the benchmark scripts (not collected by pytest).

* GPU sampling (utilisation + VRAM) through NVML (``pip install nvidia-ml-py``) or, without it,
  by polling ``nvidia-smi``. Without an NVIDIA driver the numbers are simply left out.
* The 10 benchmark clips (benchmark_audio/) and their manifest.
* benchmarks/benchmark_results.json, updated in place by every script.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn

BACKEND_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = BACKEND_DIR / "benchmarks"
AUDIO_DIR = BACKEND_DIR / "benchmark_audio"
RESULTS_PATH = BENCH_DIR / "benchmark_results.json"

if str(BACKEND_DIR) not in sys.path:          # lets `python benchmarks/<script>.py` import `app`
    sys.path.insert(0, str(BACKEND_DIR))


# ---- output -------------------------------------------------------------------------------------
def say(text: str = "") -> None:
    print(text, flush=True)


def fail(message: str, code: int = 2) -> NoReturn:
    say(f"\nERROR: {message}")
    sys.exit(code)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def closest_to_median(times: list[float], items: list[Any]) -> Any:
    """The entry of ``items`` that belongs to the run whose time is closest to the median time."""
    m = median(times)
    return items[min(range(len(times)), key=lambda k: abs(times[k] - m))]


# ---- GPU sampling -------------------------------------------------------------------------------
Sample = tuple[float, float]          # (utilisation %, memory used MiB)


def parse_smi_line(line: str) -> Sample | None:
    """'37, 4521' (nvidia-smi csv,noheader,nounits) -> (37.0, 4521.0)."""
    parts = [p.strip() for p in line.strip().split(",")]
    try:
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return None


def _nvml_reader(index: int) -> Callable[[], Sample | None] | None:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:  # noqa: BLE001
        return None

    def read() -> Sample | None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
            return float(util), float(mem)
        except Exception:  # noqa: BLE001
            return None
    return read


def _smi_reader(index: int) -> Callable[[], Sample | None] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    cmd = ["nvidia-smi", f"--id={index}", "--query-gpu=utilization.gpu,memory.used",
           "--format=csv,noheader,nounits"]

    def read() -> Sample | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_smi_line(out.splitlines()[0]) if out.strip() else None
    return read


def default_gpu_reader(index: int = 0) -> Callable[[], Sample | None] | None:
    return _nvml_reader(index) or _smi_reader(index)


class GpuMonitor:
    """Samples GPU utilisation and memory in a background thread while a block runs.

        with GpuMonitor() as gpu:
            work()
        gpu.summary()   # {"gpu_util_avg_percent": .., "gpu_util_peak_percent": .., "vram_peak_mb": ..}
    """

    def __init__(self, interval_s: float = 0.05, index: int = 0,
                 reader: Callable[[], Sample | None] | None = None) -> None:
        self.interval_s = interval_s
        self._read = reader if reader is not None else default_gpu_reader(index)
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._read is not None

    def __enter__(self) -> "GpuMonitor":
        self.samples.clear()
        self._stop.clear()
        if self._read is not None:
            first = self._read()                       # a sample straight away, even for very short runs
            if first:
                self.samples.append(first)
            self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._read is not None:
            last = self._read()
            if last:
                self.samples.append(last)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            sample = self._read() if self._read else None
            if sample:
                self.samples.append(sample)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {"gpu_util_avg_percent": round(sum(utils) / len(utils), 1),
                "gpu_util_peak_percent": round(max(utils), 1),
                "vram_peak_mb": round(max(mems)),
                "gpu_samples": len(self.samples)}


def current_vram_mb(index: int = 0) -> float | None:
    reader = default_gpu_reader(index)
    sample = reader() if reader else None
    return sample[1] if sample else None


def gpu_info(index: int = 0) -> dict[str, Any] | None:
    """{"name": "NVIDIA GeForce RTX 5070", "vram_gb": 12} or None without an NVIDIA driver."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else str(name)
        total_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
        return {"name": name, "vram_gb": round(total_mb / 1024)}
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=5, check=False).stdout.strip()
            name, total = [p.strip() for p in out.splitlines()[0].rsplit(",", 1)]
            return {"name": name, "vram_gb": round(float(total) / 1024)}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None
    return None


# ---- environment check -----------------------------------------------------------------------------
def preflight(settings, *, torch_needed: bool = False, ctranslate2_needed: bool = False) -> None:
    """Print what the benchmark will run on; stop early when SJ_GPU_PRESENT=true cannot work."""
    from app.core import gpu

    say(f"device={settings.device}  whisper={settings.whisper_model}/{settings.whisper_compute_type}  "
        f"qwen={settings.qwen_backend}:{settings.ollama_model if settings.qwen_backend == 'ollama' else settings.qwen_model_path}")
    info = gpu_info()
    if info:
        say(f"GPU: {info['name']} ({info['vram_gb']} GB)")
    if settings.device != "cuda":
        say("WARNING: SJ_GPU_PRESENT is false - these are CPU numbers, not GPU numbers.")
        return
    if info is None:
        fail("SJ_GPU_PRESENT=true but no NVIDIA GPU / driver was found (nvidia-smi and NVML unavailable).")
    gpu.prepare_cuda_runtime()
    if torch_needed:
        import torch
        problem = gpu.torch_cuda_problem(torch)
        if problem:
            fail(problem)
        say(f"torch {torch.__version__}  CUDA {torch.version.cuda}  kernels: {' '.join(torch.cuda.get_arch_list())}")
    if ctranslate2_needed:
        problem = gpu.ctranslate2_cuda_problem()
        if problem:
            fail(problem)
        import ctranslate2
        say(f"ctranslate2 {ctranslate2.__version__}  cuda devices: {ctranslate2.get_cuda_device_count()}")


# ---- audio ---------------------------------------------------------------------------------------
@dataclass
class Clip:
    id: str
    file: Path
    raw: bytes
    text: str
    question: str
    story_prompt: str
    audio: Any                   # app.models.domain.AudioData (decoded lazily)

    @property
    def base64(self) -> str:
        import base64
        return base64.b64encode(self.raw).decode()


def load_manifest() -> dict[str, Any]:
    path = AUDIO_DIR / "manifest.json"
    if not path.is_file():
        fail(f"{path} is missing.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_clips(count: int = 10, decode: bool = True) -> list[Clip]:
    """The benchmark clips in manifest order, optionally decoded to canonical AudioData."""
    manifest = load_manifest()
    settings = None
    if decode:
        from app.core.config import get_settings
        settings = get_settings()
    clips: list[Clip] = []
    for item in manifest["items"][:count]:
        path = AUDIO_DIR / item["file"]
        if not path.is_file():
            fail(f"Missing benchmark audio {path.name}. Re-create it with benchmark_audio/make_benchmark_audio.ps1 "
                 "or copy the 10 .wav files into benchmark_audio/.")
        raw = path.read_bytes()
        audio = None
        if decode:
            from app.audio.decoder import preprocess_audio
            audio = preprocess_audio(raw, settings)
        clips.append(Clip(item["id"], path, raw, item["text"], item.get("question", ""),
                          item.get("story_prompt", ""), audio))
    if len(clips) < count:
        fail(f"Expected {count} clips in benchmark_audio/manifest.json, found {len(clips)}.")
    return clips


# ---- results file ---------------------------------------------------------------------------------
def default_results() -> dict[str, Any]:
    zero_section = {"questions": 10, "total_seconds": 0, "average_question_seconds": 0}
    return {
        "gpu": {"name": "NVIDIA GeForce RTX 5070", "vram_gb": 12},
        "components": {
            "whisper": {"sequential_10_audio_seconds": 0, "batch_10_audio_seconds": 0},
            "phoneme": {"sequential_10_audio_seconds": 0, "batch_10_audio_seconds": 0},
            "qwen": {"average_latency_seconds": 0},
        },
        "sections": {name: dict(zero_section) for name in
                     ("reading", "repeat", "jumbled", "question_answer", "story_telling")},
    }


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return default_results()


def update_results(mutate: Callable[[dict[str, Any]], None], path: Path = RESULTS_PATH,
                   record_gpu: bool = True) -> dict[str, Any]:
    """Load the results file, apply ``mutate`` and write it back atomically."""
    data = load_results(path)
    if record_gpu:
        info = gpu_info()
        if info:
            data["gpu"] = info
    mutate(data)
    data["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".results-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return data


def read_env_file(path: Path = BACKEND_DIR / ".env") -> dict[str, str]:
    """Minimal KEY=VALUE reader for the backend .env (comments and quotes handled)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if val[:1] in ("'", '"') and val[-1:] == val[:1] and len(val) > 1:
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].strip()
        values[key.strip()] = val
    return values


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
