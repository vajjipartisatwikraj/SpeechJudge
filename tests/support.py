"""Shared test helpers (plain functions; fixtures live in conftest.py)."""
from __future__ import annotations

import base64
import io
import wave
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.services.acoustic.engine import NumpyAcousticEngine
from app.services.mocks import MockAcousticEngine, MockGopt, MockLanguageEvaluator, MockSTT
from app.services.registry import ServiceRegistry

SR = 16000
API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
AUDIO_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "audio"


def make_speech(duration_s: float = 4.0, pause_at: float | None = 2.0, pause_len: float = 0.5,
                amplitude: float = 0.3, noise: float = 0.001, seed: int = 0) -> np.ndarray:
    """Speech-like signal: 150 Hz harmonic tone with ~4 Hz syllabic amplitude modulation,
    optional silent gap, plus low-level noise. Not real speech; exercises the signal code."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SR)) / SR
    carrier = sum(np.sin(2 * np.pi * 150 * k * t + k) / k for k in range(1, 6))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    x = amplitude * carrier * envelope / 2.3
    if pause_at is not None:
        x[(t >= pause_at) & (t < pause_at + pause_len)] = 0.0
    x = x + noise * rng.standard_normal(len(t))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def b64_audio(samples: np.ndarray | None = None) -> str:
    samples = make_speech() if samples is None else samples
    return base64.b64encode(wav_bytes(samples)).decode()


def fixture_wav(name: str) -> Path:
    """Path of a generated voice clip (tests/fixtures/audio/<name>.wav)."""
    return AUDIO_FIXTURES / f"{name}.wav"


def make_settings(**overrides) -> Settings:
    """Settings for tests: mock mode, in-process jobs, never reads a developer's .env."""
    base = dict(mock_mode=True, job_backend="inline", service_credentials=API_KEY,
                ffmpeg_path="ffmpeg-not-installed-for-tests", device="cpu", job_retry_limit=1)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_services(stt_text: str = "the doctor suggested that the apple is good for your health",
                  real_acoustic: bool = True) -> ServiceRegistry:
    reg = ServiceRegistry(stt=MockSTT(stt_text), gopt=MockGopt(),
                          acoustic=NumpyAcousticEngine() if real_acoustic else MockAcousticEngine(),
                          language=MockLanguageEvaluator())
    reg.load_all()
    return reg


# --- HTTP helpers for API tests ---------------------------------------------------------------
EXPECTED_SENTENCE = "The doctor suggested that the apple is good for your health."


def post(client, body, path="/api/v1/evaluate", headers=AUTH):
    return client.post(path, json=body, headers=headers)


def body(qt, **kw):
    base = {"request_id": "REQ_1", "question_type": qt}
    base.update(kw)
    return base
