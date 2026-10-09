"""Map unbounded acoustic measurements onto 0..1 using configured target ranges."""
from __future__ import annotations

from typing import Any

from app.scoring.config import ScoringConfig


def ramp(value: float | None, zero_at: float, full_at: float) -> float:
    """Linear 0..1; works whether higher or lower values are better."""
    if value is None or zero_at == full_at:
        return 0.0
    t = (value - zero_at) / (full_at - zero_at)
    return min(1.0, max(0.0, t))


def band(value: float | None, zero_lo: float, full_lo: float, full_hi: float, zero_hi: float
         ) -> float:
    if value is None:
        return 0.0
    if value <= zero_lo or value >= zero_hi:
        return 0.0
    if value < full_lo:
        return (value - zero_lo) / (full_lo - zero_lo)
    if value > full_hi:
        return (zero_hi - value) / (zero_hi - full_hi)
    return 1.0


def normalize(name: str, value: float | None, config: ScoringConfig) -> float:
    spec = config.normalizers.get(name)
    if spec is None:
        raise KeyError(f"No normalizer configured for '{name}'")
    if spec["type"] == "band":
        return band(value, spec["zero_lo"], spec["full_lo"], spec["full_hi"], spec["zero_hi"])
    return ramp(value, spec["zero_at"], spec["full_at"])


def intonation_score(terminal_delta_st: float | None, sentence_end: str | None,
                     config: ScoringConfig) -> float | None:
    """Does the pitch at the end of the utterance move the way the sentence type expects?

    Statements ('.') should end lower than the speaker's median pitch, questions ('?') higher.
    Returns None when it can't be judged (unknown sentence type or too little voiced speech).
    """
    if terminal_delta_st is None or sentence_end not in (".", "?"):
        return None
    spec = config.intonation.get("question" if sentence_end == "?" else "statement")
    if not spec:
        return None
    return ramp(terminal_delta_st, spec["zero_at"], spec["full_at"])


def delivery_scores(acoustic: dict[str, Any], config: ScoringConfig,
                    sentence_end: str | None = None) -> dict[str, float]:
    """Acoustic-derived fluency / prosody / audio_quality (each 0..1).

    A component is the weighted mean of the sub-measures that were actually measured;
    measures that are unavailable (e.g. no word timestamps, no voiced frames) are left out
    and the remaining weights are renormalised instead of being scored as zero.
    """
    values = dict(acoustic)
    values["intonation_score"] = intonation_score(
        acoustic.get("terminal_pitch_delta_st"), sentence_end, config)

    out: dict[str, float] = {}
    for component, parts in config.delivery.items():
        total = weight_sum = 0.0
        for metric, weight in parts.items():
            value = values.get(metric)
            if value is None:
                continue
            total += weight * (value if metric == "intonation_score"
                               else normalize(metric, value, config))
            weight_sum += weight
        out[component] = min(1.0, max(0.0, total / weight_sum)) if weight_sum > 0 else 0.0
    out["audio_quality"] = min(1.0, max(0.0, float(acoustic.get("audio_quality_score", 0.0))))
    return out
