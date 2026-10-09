"""Filesystem locations, resolved from this file so the service works from any working directory."""
from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]      # .../speech-judge
CONFIG_DIR = BACKEND_DIR / "config"
DEFAULT_ENV_FILE = BACKEND_DIR / ".env"
DEFAULT_SCORING_CONFIG = CONFIG_DIR / "scoring.yaml"
DEFAULT_ACCENT_PROFILE = CONFIG_DIR / "accents" / "indian_english.yaml"
