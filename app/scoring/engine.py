"""Deterministic Score Engine (Requirement 15).

final = sum(metric_value * weight). No model, no randomness, no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from app.core.errors import ScoreError
from app.scoring.config import ScoringConfig, validate_weights


@dataclass
class ScoreOutcome:
    score: float
    components: list[dict[str, Any]]


def compute_score(raw_metrics: Mapping[str, float], weights: Mapping[str, float]) -> ScoreOutcome:
    """Recompute a final score from stored Raw_Metrics and a supplied weight set,
    without invoking any Shared_Service (Requirement 15.10)."""
    missing = [m for m in weights if m not in raw_metrics]
    if missing:
        raise ScoreError(f"Missing required metric: {', '.join(missing)}")

    components: list[dict[str, Any]] = []
    for name, weight in weights.items():
        value = raw_metrics[name]
        if (value is None or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0.0 or value > 1.0):
            raise ScoreError(f"Metric out of range 0..1: {name}")
        components.append({"metric": name, "value": float(value), "weight": float(weight),
                           "weighted_value": float(value) * float(weight)})

    score = math.fsum(c["weighted_value"] for c in components)
    score = min(1.0, max(0.0, score))
    return ScoreOutcome(score=score, components=components)


class ScoreEngine:
    def __init__(self, config: ScoringConfig):
        self.config = config

    def score(self, scoring_key: str, raw_metrics: Mapping[str, float]) -> ScoreOutcome:
        weights = self.config.weights.get(scoring_key)
        if weights is None:
            raise ScoreError(f"No scoring configuration for {scoring_key}")
        return compute_score(raw_metrics, weights)

    def recompute(self, raw_metrics: Mapping[str, float], weights: Mapping[str, float]
                  ) -> ScoreOutcome:
        validate_weights("custom", dict(weights))
        return compute_score(raw_metrics, weights)
