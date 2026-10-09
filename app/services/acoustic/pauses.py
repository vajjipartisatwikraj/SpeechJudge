"""Pause metrics derived from VAD speech segments."""
from __future__ import annotations

MIN_PAUSE_S = 0.25  # internal silences shorter than this are not counted as pauses


def pause_metrics(segments: list[tuple[float, float]], total_duration_s: float) -> dict:
    if not segments:
        return {
            "speech_duration_s": 0.0,
            "silence_duration_s": round(total_duration_s, 4),
            "pause_count": 0,
            "pause_duration_s": 0.0,
            "pause_mean_s": 0.0,
            "pause_max_s": 0.0,
            "pause_ratio": 0.0,
            "leading_silence_s": round(total_duration_s, 4),
            "trailing_silence_s": 0.0,
            "speech_span_s": 0.0,
        }

    speech = sum(e - s for s, e in segments)
    gaps = [s2 - e1 for (_, e1), (s2, _) in zip(segments, segments[1:])]
    pauses = [g for g in gaps if g >= MIN_PAUSE_S]
    first, last = segments[0][0], segments[-1][1]
    span = max(last - first, 1e-9)
    pause_total = sum(pauses)
    return {
        "speech_duration_s": round(speech, 4),
        "silence_duration_s": round(max(0.0, total_duration_s - speech), 4),
        "pause_count": len(pauses),
        "pause_duration_s": round(pause_total, 4),
        "pause_mean_s": round(pause_total / len(pauses), 4) if pauses else 0.0,
        "pause_max_s": round(max(pauses), 4) if pauses else 0.0,
        "pause_ratio": round(min(1.0, pause_total / span), 4),
        "leading_silence_s": round(first, 4),
        "trailing_silence_s": round(max(0.0, total_duration_s - last), 4),
        "speech_span_s": round(span, 4),
    }
