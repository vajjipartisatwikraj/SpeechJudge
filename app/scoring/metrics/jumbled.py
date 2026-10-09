"""Raw-metric builders for Jumbled (UI-based and spoken)."""
from __future__ import annotations

from typing import Any

from app.models.domain import TextComparisonResult
from app.scoring.config import ScoringConfig
from app.scoring.normalizers import delivery_scores
from app.scoring.metrics.reading import clamp01


def build_ui_raw_metrics(text: TextComparisonResult) -> dict[str, float]:
    return {
        "word_accuracy": clamp01(text.word_accuracy),
        "order_correctness": clamp01(text.order_correctness),
    }


def build_spoken_raw_metrics(text: TextComparisonResult, acoustic: dict[str, Any],
                             config: ScoringConfig, sentence_end: str | None = None
                             ) -> dict[str, float]:
    delivery = delivery_scores(acoustic, config, sentence_end)
    metrics = build_ui_raw_metrics(text)
    metrics["fluency"] = delivery.get("fluency", 0.0)
    metrics["prosody"] = delivery.get("prosody", 0.0)
    metrics["audio_quality"] = delivery["audio_quality"]
    return metrics
