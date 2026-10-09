"""Energy and loudness metrics."""
from __future__ import annotations

import numpy as np

from app.services.acoustic.vad import frame_rms_db

_EPS = 1e-12


def energy_metrics(x: np.ndarray, sr: int, speech_mask: np.ndarray) -> dict:
    db = frame_rms_db(x, sr)
    n = min(len(db), len(speech_mask))
    voiced_db = db[:n][speech_mask[:n]] if n and speech_mask[:n].any() else db
    return {
        "energy_mean_db": round(float(np.mean(voiced_db)), 3),
        "energy_std_db": round(float(np.std(voiced_db)), 3),
        "peak_amplitude": round(float(np.max(np.abs(x))) if len(x) else 0.0, 5),
    }


def loudness_lufs(x: np.ndarray, sr: int) -> tuple[float, str]:
    """Integrated loudness. Uses pyloudnorm (BS.1770) when installed, else an unweighted
    mean-square approximation (reported via the returned source label)."""
    if len(x) == 0:
        return -70.0, "approximation"
    try:
        import pyloudnorm as pyln  # type: ignore

        if len(x) >= int(0.4 * sr):  # BS.1770 needs at least one 400 ms block
            value = float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
            if np.isfinite(value):
                return round(value, 3), "pyloudnorm"
    except Exception:  # noqa: BLE001 - optional dependency, fall back deterministically
        pass
    ms = float(np.mean(x.astype(np.float64) ** 2))
    return round(max(-70.0, -0.691 + 10.0 * np.log10(ms + _EPS)), 3), "approximation"
