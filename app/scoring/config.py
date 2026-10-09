"""Scoring configuration loading and validation (Requirement 15.2, 15.3)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import ScoringConfigError
from app.models.domain import QuestionType

WEIGHT_TOLERANCE = 0.001
REQUIRED_KEYS = (
    QuestionType.READING.value,
    QuestionType.REPEAT.value,
    "JUMBLED_UI",
    "JUMBLED_SPOKEN",
    QuestionType.QUESTION_ANSWER.value,
    QuestionType.STORYTELLING.value,
)


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, dict[str, float]]
    normalizers: dict[str, dict[str, Any]] = field(default_factory=dict)
    delivery: dict[str, dict[str, float]] = field(default_factory=dict)
    intonation: dict[str, dict[str, float]] = field(default_factory=dict)


def validate_weights(key: str, weights: dict[str, float]) -> None:
    if not weights:
        raise ScoringConfigError(f"Scoring config '{key}': no weights defined")
    for name, w in weights.items():
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or w < 0:
            raise ScoringConfigError(f"Scoring config '{key}': invalid weight for '{name}': {w!r}")
    total = math.fsum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ScoringConfigError(f"Scoring config '{key}': weights sum to {total:.4f}, expected 1.0")


def parse_scoring_config(data: dict[str, Any]) -> ScoringConfig:
    scoring = data.get("scoring")
    if not isinstance(scoring, dict):
        raise ScoringConfigError("Scoring config: missing 'scoring' section")
    missing = [k for k in REQUIRED_KEYS if k not in scoring]
    if missing:
        raise ScoringConfigError(f"Scoring config: missing question types: {', '.join(missing)}")
    weights = {k: {m: float(w) for m, w in (v or {}).items()} for k, v in scoring.items()}
    for key, w in weights.items():
        validate_weights(key, w)
    return ScoringConfig(weights=weights,
                         normalizers=data.get("normalizers") or {},
                         delivery=data.get("delivery") or {},
                         intonation=data.get("intonation") or {})


def load_scoring_config(path: str | Path) -> ScoringConfig:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScoringConfigError(f"Cannot read scoring config '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ScoringConfigError("Scoring config must be a YAML mapping")
    return parse_scoring_config(data)
