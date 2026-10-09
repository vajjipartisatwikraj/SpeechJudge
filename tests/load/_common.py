"""Shared helpers for the load / benchmark scripts (not collected by pytest)."""
from __future__ import annotations

import base64
import sys
import wave
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = BACKEND_DIR / "tests" / "fixtures" / "audio"
SR = 16000

if str(BACKEND_DIR) not in sys.path:          # lets `python tests/load/<script>.py` import `app`
    sys.path.insert(0, str(BACKEND_DIR))


def load_samples(name: str) -> np.ndarray:
    """Float32 mono 16 kHz samples of a fixture clip."""
    path = AUDIO_DIR / f"{name}.wav"
    if not path.exists():
        raise SystemExit(f"Missing fixture {path.name}. Create the clips with:  pwsh tests/fixtures/make_audio.ps1")
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def load_base64(name: str) -> str:
    path = AUDIO_DIR / f"{name}.wav"
    if not path.exists():
        raise SystemExit(f"Missing fixture {path.name}. Create the clips with:  pwsh tests/fixtures/make_audio.ps1")
    return base64.b64encode(path.read_bytes()).decode()


def ten_second_clips(count: int = 10) -> list[np.ndarray]:
    """`count` different 10-second windows of real speech (story + Q&A clips), float32 16 kHz."""
    story, qa = load_samples("story"), load_samples("qa_good")
    clips = []
    for i in range(count):
        src = story[int(i * 1.6 * SR): int(i * 1.6 * SR) + 10 * SR] if i < 8 else qa[: 10 * SR]
        clips.append(np.pad(src, (0, max(0, 10 * SR - len(src)))).astype(np.float32))
    return clips


def credential_from_env_file() -> str:
    env = BACKEND_DIR / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("SJ_SERVICE_CREDENTIALS="):
                return line.split("=", 1)[1].split("#")[0].strip().split(",")[0]
    return ""
