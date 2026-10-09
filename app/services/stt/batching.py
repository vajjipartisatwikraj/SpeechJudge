"""Pack several short clips into one waveform so Whisper decodes them as ONE GPU batch.

faster-whisper's ``BatchedInferencePipeline`` batches the *clips of a single recording*. To batch
different recordings (10 students' answers, 10 questions of one section) we place them one after the
other in a single array, separated by silence, and tell the pipeline where each clip starts and ends
(``clip_timestamps``). The pipeline then decodes the clips as one batch and returns segments and word
timestamps on the packed timeline; ``unpack_segments`` maps them back to each clip (times relative to
the start of that clip, exactly like single-clip transcription).

Only the public faster-whisper API is used.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Any, Iterable, Sequence

import numpy as np

from app.models.domain import Transcription

SAMPLE_RATE = 16_000
GAP_S = 1.0               # silence between clips: keeps every clip's segments unambiguous
MAX_CLIP_S = 29.5         # Whisper decodes a 30 s window; longer clips use the normal long-form decoder


def pack_clips(clips: Sequence[np.ndarray], sample_rate: int = SAMPLE_RATE, gap_s: float = GAP_S
               ) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Return (packed waveform, [{"start": s, "end": s}, ...] in seconds, one per clip)."""
    grid = sample_rate // 100                    # Whisper positions chunks on a 10 ms grid (seek)
    gap = np.zeros(int(gap_s * sample_rate) // grid * grid, dtype=np.float32)
    parts: list[np.ndarray] = []
    spans: list[dict[str, float]] = []
    cursor = 0
    for clip in clips:
        clip = np.asarray(clip, dtype=np.float32)
        spans.append({"start": cursor / sample_rate, "end": (cursor + len(clip)) / sample_rate})
        tail = np.zeros(-len(clip) % grid, dtype=np.float32)       # next clip starts on the grid too
        parts.extend([clip, tail, gap])
        cursor += len(clip) + len(tail) + len(gap)
    return np.concatenate(parts), spans


def unpack_segments(segments: Iterable[Any], spans: Sequence[dict[str, float]]) -> list[Transcription]:
    """Distribute faster-whisper segments (packed timeline) back to their clips."""
    starts = [s["start"] for s in spans]
    seg_out: list[list[dict]] = [[] for _ in spans]
    words_out: list[list[dict]] = [[] for _ in spans]
    for seg in segments:
        # tolerance: Whisper rounds chunk offsets down to 10 ms; clips are >= GAP_S apart
        idx = max(0, bisect_right(starts, seg.start + 0.05) - 1)
        base = spans[idx]["start"]
        seg_out[idx].append({"start": round(max(seg.start - base, 0.0), 3),
                             "end": round(max(seg.end - base, 0.0), 3), "text": seg.text.strip()})
        for w in seg.words or []:
            words_out[idx].append({"word": w.word.strip(), "start": round(max(w.start - base, 0.0), 3),
                                   "end": round(max(w.end - base, 0.0), 3),
                                   "probability": round(w.probability, 4)})
    return [Transcription(text=" ".join(s["text"] for s in segs).strip(), segments=segs,
                          word_timestamps=words)
            for segs, words in zip(seg_out, words_out)]
