"""Raw-metric builders for pronunciation-centred question types (Reading, Repeat)."""
from __future__ import annotations

from typing import Any

from app.models.domain import GoptResult, TextComparisonResult
from app.scoring.config import ScoringConfig
from app.scoring.normalizers import delivery_scores


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def sentence_end_of(expected_text: str | None) -> str | None:
    """'.' or '?' if the reference sentence ends that way (used for the intonation check)."""
    text = (expected_text or "").strip()
    return text[-1] if text[-1:] in (".", "?") else None


def build_gopt_raw_metrics(text: TextComparisonResult, gopt: GoptResult, acoustic: dict[str, Any],
                           config: ScoringConfig, sentence_end: str | None = None
                           ) -> dict[str, float]:
    """Pronunciation + completeness come from the pronunciation backend. Fluency and prosody come
    from the acoustic engine, except when a genuine GOPT backend supplies its own."""
    delivery = delivery_scores(acoustic, config, sentence_end)
    use_gopt_delivery = gopt.method == "gopt" and gopt.fluency is not None and gopt.prosody is not None
    fluency = clamp01(gopt.fluency) if use_gopt_delivery else delivery.get("fluency", 0.0)
    prosody = clamp01(gopt.prosody) if use_gopt_delivery else delivery.get("prosody", 0.0)
    return {
        "word_accuracy": clamp01(text.word_accuracy),
        "order_correctness": clamp01(text.order_correctness),
        "pronunciation": clamp01(gopt.pronunciation),
        "fluency": fluency,
        "prosody": prosody,
        "completeness": clamp01(gopt.completeness),
        "audio_quality": delivery["audio_quality"],
    }


def build_raw_metrics(text: TextComparisonResult, gopt: GoptResult, acoustic: dict[str, Any],
                      config: ScoringConfig, sentence_end: str | None = None) -> dict[str, float]:
    return build_gopt_raw_metrics(text, gopt, acoustic, config, sentence_end)
