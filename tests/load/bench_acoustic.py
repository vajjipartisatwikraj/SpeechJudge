"""Acoustic engine benchmark: time per clip for 3-15 s of real speech, plus 10 clips sequential vs threaded.

    python tests/load/bench_acoustic.py
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _common import SR, load_samples


def main() -> None:
    from app.models.domain import AudioData
    from app.services.acoustic.engine import NumpyAcousticEngine

    speech = np.concatenate([load_samples("story"), load_samples("qa_good")])     # ~33 s of real speech
    engine = NumpyAcousticEngine()
    engine.load()                                                   # includes library warm-up

    def clip(seconds: float, offset_s: float = 0.0) -> AudioData:
        start = int(offset_s * SR)
        return AudioData(speech[start:start + int(seconds * SR)], float(seconds))

    print(f"{'length':>7} {'probe ms':>9} {'analyze ms':>11} {'total ms':>9} {'p95 total':>9} {'x realtime':>10}")
    for secs in (3, 5, 7, 10, 12, 15):
        audios = [clip(secs, o) for o in (0, 2, 4, 6, 8, 10)]
        probe_ms, total_ms = [], []
        for a in audios * 4:                                        # 24 runs on 6 different windows
            t0 = time.perf_counter()
            p = engine.probe(a)
            t1 = time.perf_counter()
            m = engine.analyze(a, p)
            engine.refine_rates(m, int(secs * 2.5))
            t2 = time.perf_counter()
            probe_ms.append((t1 - t0) * 1000)
            total_ms.append((t2 - t0) * 1000)
        tot, pr = statistics.median(total_ms), statistics.median(probe_ms)
        print(f"{secs:>5} s {pr:>9.0f} {tot - pr:>11.0f} {tot:>9.0f} {np.percentile(total_ms, 95):>9.0f} "
              f"{secs * 1000 / tot:>9.0f}x")

    ten = [clip(10, o) for o in range(10)]

    def full(a):
        engine.analyze(a, engine.probe(a))

    t = time.perf_counter()
    for a in ten:
        full(a)
    seq = time.perf_counter() - t
    t = time.perf_counter()
    with ThreadPoolExecutor(10) as pool:
        list(pool.map(full, ten))
    par = time.perf_counter() - t
    print(f"\n10 clips x 10 s: one by one {seq * 1000:.0f} ms ({seq * 100:.0f} ms/clip) | 10 threads {par * 1000:.0f} ms")


if __name__ == "__main__":
    main()
