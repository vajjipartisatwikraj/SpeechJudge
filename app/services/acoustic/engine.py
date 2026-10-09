"""Acoustic Engine: objective, deterministic speech-signal analysis (Requirement 11)."""
from __future__ import annotations

from typing import Any

import numpy as np

from app.models.domain import AudioData, QualityProbe
from app.services.acoustic import vad
from app.services.acoustic.energy import energy_metrics, loudness_lufs
from app.services.acoustic.pauses import pause_metrics
from app.services.acoustic.pitch import pitch_metrics
from app.services.acoustic.speech_rate import (SYLLABLES_PER_WORD, rates_from_word_count,
                                               syllable_energy_std_db, syllable_peaks)
from app.services.interfaces import AcousticEngine


def _mask_from_segments(segments: list[tuple[float, float]], n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for s, e in segments:
        mask[int(round(s / vad.HOP_S)): int(round(e / vad.HOP_S))] = True
    return mask


def audio_quality_score(snr_db: float, clipping: float, loudness: float) -> float:
    """0..1 composite: SNR 50%, clipping 30%, loudness-in-range 20%."""
    snr_score = min(1.0, max(0.0, (snr_db - 10.0) / 20.0))          # 10 dB -> 0, 30 dB -> 1
    clip_score = 1.0 - min(1.0, clipping / 0.05)                    # 5% clipped -> 0
    off = 0.0 if -30.0 <= loudness <= -10.0 else min(abs(loudness + 20.0) - 10.0, 20.0)
    loud_score = 1.0 - off / 20.0
    return round(0.5 * snr_score + 0.3 * clip_score + 0.2 * loud_score, 4)


class NumpyAcousticEngine(AcousticEngine):
    """Baseline engine. Uses parselmouth / pyloudnorm automatically when installed."""

    def load(self) -> None:
        # Warm up so optional libraries (parselmouth, pyloudnorm, scipy) are imported now,
        # not on the first student's request (observed ~12 s cold start on CPU).
        t = np.arange(16_000 * 2) / 16_000
        tone = (0.3 * np.sin(2 * np.pi * 150 * t) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
                ).astype(np.float32)
        self.analyze(AudioData(samples=tone, original_duration_s=2.0))
        self._loaded = True

    def probe(self, audio: AudioData) -> QualityProbe:
        x, sr = audio.samples, audio.sample_rate
        mask = vad.speech_mask(x, sr)
        segments = vad.mask_to_segments(mask)
        return QualityProbe(
            duration_s=audio.duration_s,
            speech_duration_s=float(sum(e - s for s, e in segments)),
            snr_db=vad.estimate_snr_db(x, sr, mask),
            clipping_ratio=vad.clipping_ratio(x),
            speech_segments=segments,
        )

    def analyze(self, audio: AudioData, probe: QualityProbe | None = None) -> dict[str, Any]:
        x, sr = audio.samples, audio.sample_rate
        if probe is None:
            probe = self.probe(audio)
        n_frames = len(vad.frame_rms_db(x, sr))
        mask = _mask_from_segments(probe.speech_segments, n_frames)

        metrics: dict[str, Any] = {
            "original_duration_s": round(audio.original_duration_s, 4),
            "duration_s": round(audio.duration_s, 4),
        }
        metrics.update(pause_metrics(probe.speech_segments, audio.duration_s))
        metrics.update(pitch_metrics(x, sr, mask, vad.HOP_S, int(mask.sum())))
        metrics.update(energy_metrics(x, sr, mask))
        lufs, lufs_source = loudness_lufs(x, sr)
        metrics.update({
            "loudness_lufs": lufs,
            "loudness_source": lufs_source,
            "snr_db": round(probe.snr_db, 3),
            "clipping_ratio": round(probe.clipping_ratio, 6),
        })

        peaks = syllable_peaks(x, sr, mask)
        syllables = len(peaks)
        metrics["syllable_energy_std_db"] = syllable_energy_std_db(peaks)
        est_words = syllables / SYLLABLES_PER_WORD
        rate, artic = rates_from_word_count(est_words, metrics["speech_duration_s"],
                                            metrics["speech_span_s"])
        metrics.update({
            "syllable_count_estimate": syllables,
            "speech_rate_wpm": rate,
            "articulation_rate_wpm": artic,
            "rate_source": "estimated",
        })
        metrics["audio_quality_score"] = audio_quality_score(probe.snr_db, probe.clipping_ratio, lufs)
        return metrics

    def refine_rates(self, metrics: dict[str, Any], word_count: int) -> dict[str, Any]:
        refined = dict(metrics)
        rate, artic = rates_from_word_count(word_count, metrics["speech_duration_s"],
                                            metrics["speech_span_s"])
        refined.update({"speech_rate_wpm": rate, "articulation_rate_wpm": artic,
                        "rate_source": "transcript"})
        return refined
