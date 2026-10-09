"""Fluency measures derived from word timestamps + transcript.

Based on the usual spoken-fluency indicators: where pauses fall (between clauses is natural,
inside a phrase signals hesitation), mean length of run between pauses, long pauses, and
disfluencies (fillers, immediate repetitions).
"""
from __future__ import annotations

import re
from typing import Any

PAUSE_S = 0.25
LONG_PAUSE_S = 1.0
MIN_WORDS = 4
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mmm"}
_BOUNDARY_END = (".", ",", ";", ":", "?", "!")
_STRIP = re.compile(r"[^\w']+")


def _norm(word: str) -> str:
    return _STRIP.sub("", word.lower())


def word_fluency_metrics(word_timestamps: list[dict[str, Any]]) -> dict[str, Any]:
    """Empty dict when there isn't enough data (callers then skip these components)."""
    words = [w for w in word_timestamps if _norm(w.get("word", ""))]
    n = len(words)
    if n < MIN_WORDS:
        return {}

    mid_pauses = boundary_pauses = long_pauses = 0
    runs: list[int] = []
    run = 1
    for prev, cur in zip(words, words[1:]):
        gap = max(0.0, float(cur["start"]) - float(prev["end"]))
        if gap >= PAUSE_S:
            if str(prev["word"]).rstrip().endswith(_BOUNDARY_END):
                boundary_pauses += 1
            else:
                mid_pauses += 1
            if gap >= LONG_PAUSE_S:
                long_pauses += 1
            runs.append(run)
            run = 1
        else:
            run += 1
    runs.append(run)

    normed = [_norm(w["word"]) for w in words]
    fillers = sum(1 for w in normed if w in FILLERS)
    repeats = sum(1 for a, b in zip(normed, normed[1:]) if a == b and a not in FILLERS)

    per10 = 10.0 / n
    return {
        "mid_clause_pause_count": mid_pauses,
        "boundary_pause_count": boundary_pauses,
        "long_pause_count": long_pauses,
        "mid_clause_pause_rate": round(mid_pauses * per10, 3),
        "mean_run_length": round(sum(runs) / len(runs), 3),
        "hesitation_count": fillers + repeats,
        "hesitation_rate": round((fillers + repeats) * per10, 3),
    }
