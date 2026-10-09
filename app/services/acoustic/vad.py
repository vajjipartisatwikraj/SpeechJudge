"""Deterministic energy-based voice activity detection + quality probes.

This is the dependency-free baseline. It can be swapped for Silero VAD behind the
same ``detect_speech`` signature without touching callers.
"""
from __future__ import annotations

import numpy as np

FRAME_S = 0.025
HOP_S = 0.010
MIN_SPEECH_S = 0.10   # drop speech blips shorter than this
MERGE_GAP_S = 0.10    # bridge very short dips inside a word/phrase
_EPS = 1e-12


def frame_rms_db(x: np.ndarray, sr: int) -> np.ndarray:
    frame = int(FRAME_S * sr)
    hop = int(HOP_S * sr)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), frame)[::hop]
    power = np.mean(frames ** 2, axis=1)
    return 10.0 * np.log10(power + _EPS)


def frame_power(x: np.ndarray, sr: int) -> np.ndarray:
    return 10.0 ** (frame_rms_db(x, sr) / 10.0)


def speech_mask(x: np.ndarray, sr: int) -> np.ndarray:
    """Boolean per-frame speech decision (hop = HOP_S)."""
    db = frame_rms_db(x, sr)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    # Need a clear dynamic range and audible level; otherwise it's silence/stationary noise.
    if peak - floor < 8.0 or peak < -55.0:
        return np.zeros(len(db), dtype=bool)
    threshold = floor + 0.35 * (peak - floor)
    mask = db > threshold
    return _smooth(mask)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index runs where mask is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _smooth(mask: np.ndarray) -> np.ndarray:
    gap = int(round(MERGE_GAP_S / HOP_S))
    minlen = int(round(MIN_SPEECH_S / HOP_S))
    out = mask.copy()
    # bridge short silent gaps between speech runs
    runs = _runs(mask)
    for (_, e1), (s2, _) in zip(runs, runs[1:]):
        if s2 - e1 <= gap:
            out[e1:s2] = True
    # remove short speech blips
    for s, e in _runs(out):
        if e - s < minlen:
            out[s:e] = False
    return out


def mask_to_segments(mask: np.ndarray) -> list[tuple[float, float]]:
    return [(s * HOP_S, e * HOP_S) for s, e in _runs(mask)]


def detect_speech(x: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Speech segments as (start_s, end_s)."""
    return mask_to_segments(speech_mask(x, sr))


def estimate_snr_db(x: np.ndarray, sr: int, mask: np.ndarray | None = None) -> float:
    """Speech-frame power over non-speech-frame power, in dB. 0.0 if there is no speech."""
    if mask is None:
        mask = speech_mask(x, sr)
    if not mask.any():
        return 0.0
    power = frame_power(x, sr)
    n = min(len(power), len(mask))
    power, mask = power[:n], mask[:n]
    speech_p = float(np.mean(power[mask]))
    if (~mask).any():
        noise_p = float(np.mean(power[~mask]))
    else:
        noise_p = float(np.percentile(power, 10))
    noise_p = max(noise_p, 1e-10)
    return float(min(60.0, max(0.0, 10.0 * np.log10(max(speech_p, _EPS) / noise_p))))


def clipping_ratio(x: np.ndarray, level: float = 0.99) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= level))
