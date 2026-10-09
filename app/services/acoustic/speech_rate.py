"""Speech/articulation rate and syllable-level energy variation.

When no transcript is available the rate is *estimated* from syllable nuclei
(energy-envelope peaks) and converted to words with ~1.4 syllables/word
(Requirement 11.3). When a transcript word count is known, that is used instead.
"""
from __future__ import annotations

import numpy as np

from app.services.acoustic.vad import HOP_S, frame_rms_db

SYLLABLES_PER_WORD = 1.4
MIN_SYLLABLE_GAP_S = 0.08
MIN_PROMINENCE_DB = 3.0


def syllable_peaks(x: np.ndarray, sr: int, speech_mask: np.ndarray) -> list[tuple[int, float]]:
    """Syllable-nucleus candidates as (frame_index, smoothed_level_db)."""
    db = frame_rms_db(x, sr)
    n = min(len(db), len(speech_mask))
    if n < 3 or not speech_mask[:n].any():
        return []
    db = db[:n]
    smooth = np.convolve(db, np.ones(3) / 3.0, mode="same")  # suppress jitter
    min_gap = max(1, int(round(MIN_SYLLABLE_GAP_S / HOP_S)))
    peaks: list[int] = []
    for i in range(1, n - 1):
        if not speech_mask[i] or not (smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]):
            continue
        lo = max(0, i - min_gap * 2)
        hi = min(n, i + min_gap * 2 + 1)
        prominence = smooth[i] - max(smooth[lo:i + 1].min(), smooth[i:hi].min())
        if prominence < MIN_PROMINENCE_DB:
            continue
        if peaks and i - peaks[-1] < min_gap:
            if smooth[i] > smooth[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return [(i, float(smooth[i])) for i in peaks]


def estimate_syllable_count(x: np.ndarray, sr: int, speech_mask: np.ndarray) -> int:
    return len(syllable_peaks(x, sr, speech_mask))


def syllable_energy_std_db(peaks: list[tuple[int, float]]) -> float | None:
    """Spread of syllable loudness (stress/emphasis variation). None if too few syllables."""
    if len(peaks) < 4:
        return None
    return round(float(np.std([level for _, level in peaks])), 3)


def rates_from_word_count(word_count: float, speech_duration_s: float, speech_span_s: float
                          ) -> tuple[float, float]:
    """(speech_rate_wpm over speaking span, articulation_rate_wpm over voiced time)."""
    speech_rate = word_count / (speech_span_s / 60.0) if speech_span_s > 0 else 0.0
    artic = word_count / (speech_duration_s / 60.0) if speech_duration_s > 0 else 0.0
    return round(speech_rate, 2), round(artic, 2)
