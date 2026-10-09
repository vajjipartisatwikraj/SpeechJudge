"""Audio Quality Gate (Requirement 6). Pure function of (probe, thresholds): deterministic."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import Settings
from app.models.domain import Flag, QualityProbe


@dataclass
class GateDecision:
    rejected: bool
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def evaluate_gate(probe: QualityProbe, settings: Settings) -> GateDecision:
    if probe.duration_s < settings.min_duration_s:
        return GateDecision(True, "Audio too short")
    if probe.speech_duration_s < settings.min_speech_s:
        return GateDecision(True, "Insufficient speech signal")

    flags: list[str] = []
    if probe.snr_db < settings.snr_threshold_db:
        flags.append(Flag.EXCESSIVE_NOISE.value)
    if probe.clipping_ratio > settings.clipping_threshold:
        flags.append(Flag.CLIPPING.value)
    return GateDecision(False, None, flags)
