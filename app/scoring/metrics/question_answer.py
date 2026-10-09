"""Raw-metric builders for open-ended questions (Q&A and Storytelling)."""
from __future__ import annotations

from typing import Any

from app.models.domain import LanguageResult
from app.scoring.config import ScoringConfig
from app.scoring.normalizers import delivery_scores
from app.scoring.metrics.reading import clamp01


def build_language_raw_metrics(language: LanguageResult, acoustic: dict[str, Any],
                               config: ScoringConfig) -> dict[str, float]:
    """Language rubric dimensions (from Qwen) + acoustic fluency/prosody/audio quality."""
    metrics = {name: clamp01(score) for name, score in language.scores().items()}
    delivery = delivery_scores(acoustic, config)
    metrics["fluency"] = delivery.get("fluency", 0.0)
    metrics["prosody"] = delivery.get("prosody", 0.0)
    metrics["audio_quality"] = delivery["audio_quality"]
    return metrics


def build_raw_metrics(language: LanguageResult, acoustic: dict[str, Any], config: ScoringConfig
                      ) -> dict[str, float]:
    return build_language_raw_metrics(language, acoustic, config)
