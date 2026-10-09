"""Pitch (F0) metrics. Parselmouth/Praat when available, autocorrelation fallback otherwise.

Prosody-relevant measures are expressed in semitones relative to the speaker's own median
pitch, so a deep and a high voice with the same expressiveness get the same values.
"""
from __future__ import annotations

import numpy as np

FMIN, FMAX = 75.0, 400.0
FRAME_S, HOP_S = 0.040, 0.020
VOICING_THRESHOLD = 0.35
MIN_VOICED_FRAMES = 8


def _f0_autocorr(x: np.ndarray, sr: int, speech_mask: np.ndarray, mask_hop_s: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, f0_hz) of voiced frames inside VAD speech."""
    empty = (np.zeros(0), np.zeros(0))
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < frame:
        return empty
    windows = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), frame)[::hop]
    centers = (np.arange(len(windows)) * hop + frame // 2) / sr
    idx = np.minimum((centers / mask_hop_s).astype(int), max(len(speech_mask) - 1, 0))
    keep = speech_mask[idx] if len(speech_mask) else np.zeros(len(windows), dtype=bool)
    windows, centers = windows[keep], centers[keep]
    if len(windows) == 0:
        return empty

    lag_min, lag_max = int(sr / FMAX), int(sr / FMIN)
    nfft = 1
    while nfft < 2 * frame:
        nfft *= 2
    hann = np.hanning(frame)
    times, f0s = [], []
    for start in range(0, len(windows), 512):
        chunk = windows[start:start + 512]
        chunk = (chunk - chunk.mean(axis=1, keepdims=True)) * hann
        spec = np.fft.rfft(chunk, n=nfft, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, : lag_max + 1]
        r0 = np.maximum(ac[:, 0], 1e-12)
        region = (ac / r0[:, None])[:, lag_min: lag_max + 1]
        peak_idx = np.argmax(region, axis=1)
        peak_val = region[np.arange(len(region)), peak_idx]
        voiced = (peak_val > VOICING_THRESHOLD) & (r0 > 1e-8)
        f0 = sr / (peak_idx + lag_min)
        f0s.append(f0[voiced])
        times.append(centers[start:start + 512][voiced])
    return np.concatenate(times), np.concatenate(f0s)


def _f0_parselmouth(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import parselmouth  # type: ignore

        pitch = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr).to_pitch(
            pitch_floor=FMIN, pitch_ceiling=FMAX)
        values = pitch.selected_array["frequency"]
        times = pitch.xs()
        ok = values > 0
        return times[ok], values[ok]
    except Exception:  # noqa: BLE001 - optional dependency
        return None


_NONE = {"pitch_mean_hz": None, "pitch_min_hz": None, "pitch_max_hz": None, "pitch_range_hz": None,
         "pitch_std_hz": None, "pitch_median_hz": None, "pitch_std_st": None,
         "pitch_range_st": None, "terminal_pitch_delta_st": None, "voiced_ratio": 0.0}


def pitch_metrics(x: np.ndarray, sr: int, speech_mask: np.ndarray, mask_hop_s: float,
                  total_speech_frames: int) -> dict:
    found = _f0_parselmouth(x, sr)
    source = "parselmouth"
    if found is None:
        found = _f0_autocorr(x, sr, speech_mask, mask_hop_s)
        source = "autocorrelation"
    times, f0 = found

    if len(f0) < 3:
        return {**_NONE, "pitch_source": source}

    lo, hi = np.percentile(f0, [2, 98])  # trim octave-error outliers
    keep = (f0 >= lo) & (f0 <= hi)
    times, f0 = times[keep], f0[keep]
    voiced_ratio = min(1.0, len(f0) * HOP_S / max(total_speech_frames * mask_hop_s, 1e-9))

    median = float(np.median(f0))
    semitones = 12.0 * np.log2(f0 / median)
    p5, p95 = np.percentile(semitones, [5, 95])

    terminal = None
    if len(f0) >= MIN_VOICED_FRAMES:
        order = np.argsort(times)
        tail = semitones[order][-max(3, len(f0) // 4):]
        terminal = round(float(np.median(tail)), 3)   # end of utterance vs. speaker median

    return {
        "pitch_mean_hz": round(float(np.mean(f0)), 2),
        "pitch_min_hz": round(float(np.min(f0)), 2),
        "pitch_max_hz": round(float(np.max(f0)), 2),
        "pitch_range_hz": round(float(np.max(f0) - np.min(f0)), 2),
        "pitch_std_hz": round(float(np.std(f0)), 2),
        "pitch_median_hz": round(median, 2),
        "pitch_std_st": round(float(np.std(semitones)), 3),
        "pitch_range_st": round(float(p95 - p5), 3),
        "terminal_pitch_delta_st": terminal,
        "voiced_ratio": round(float(voiced_ratio), 4),
        "pitch_source": source,
    }
