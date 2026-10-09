"""Fixtures for end-to-end tests. They use the REAL backend configuration (speech-judge/.env),
real models, and the generated voice clips in tests/fixtures/audio.  Run:  pytest --real"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

from app.audio.decoder import preprocess_audio
from app.core.config import Settings
from app.core.paths import DEFAULT_ENV_FILE
from app.models.domain import AudioData
from tests.support import AUDIO_FIXTURES, fixture_wav

@pytest.fixture(scope="session")
def real_settings() -> Settings:
    if not DEFAULT_ENV_FILE.exists():
        pytest.skip("speech-judge/.env not found (copy .env.example to .env)")
    return Settings(_env_file=str(DEFAULT_ENV_FILE), mock_mode=False)


@pytest.fixture(scope="session")
def load_clip(real_settings):
    if not AUDIO_FIXTURES.exists() or not any(AUDIO_FIXTURES.glob("*.wav")):
        pytest.skip("no audio fixtures; run: pwsh tests/fixtures/make_audio.ps1")

    def _load(name: str) -> AudioData:
        path = fixture_wav(name)
        if not path.exists():
            pytest.skip(f"missing fixture {path.name}")
        return preprocess_audio(path.read_bytes(), real_settings.model_copy(update={"ffmpeg_path": "none"}))
    return _load


def _get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local test URLs
        return json.loads(r.read())


@pytest.fixture(scope="session")
def stack_url() -> str:
    """Base URL of a running stack (the UI proxy or the backend with its own credential)."""
    url = os.environ.get("SJ_E2E_URL", "http://127.0.0.1:8080").rstrip("/")
    try:
        ready = _get_json(f"{url}/api/v1/ready")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no running stack at {url} ({exc}); start the backend and frontend/server.py")
    if ready.get("status") != "ready":
        pytest.skip(f"stack at {url} is not ready: {ready}")
    return url
