"""Phoneme (pronunciation) benchmark: 10 audio files one by one vs the same 10 files as ONE batch.

    python benchmarks/phoneme_benchmark.py [--repeat 3] [--clips 10] [--no-save]

Uses the real PhonemePronunciationService (wav2vec2 phoneme recogniser on SJ_DEVICE) with the settings
from speech-judge/.env. Model load and warm-up are not timed. Each mode runs --repeat times; the
median is reported. Writes components.phoneme in benchmarks/benchmark_results.json.

This process loads its OWN copy of the model: if the VRAM is tight, stop the worker while it runs.
"""
from __future__ import annotations

import argparse
import time

from _common import (GpuMonitor, closest_to_median, load_clips, median, now_iso, preflight, say,
                     update_results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=3, help="timed repetitions per mode (median reported)")
    ap.add_argument("--clips", type=int, default=10, help="number of benchmark clips (default 10 = one batch of 10)")
    ap.add_argument("--no-save", action="store_true", help="do not write benchmark_results.json")
    args = ap.parse_args()

    from app.core.config import get_settings
    from app.services.pronunciation.phoneme_service import PhonemePronunciationService

    settings = get_settings()
    settings.phoneme_batch_size = max(settings.phoneme_batch_size, args.clips)   # one batch holds every clip
    preflight(settings, torch_needed=True)
    clips = load_clips(args.clips)
    items = [(c.audio, c.text) for c in clips]
    say(f"{len(clips)} clips, {sum(c.audio.duration_s for c in clips):.1f} s of audio in total")

    service = PhonemePronunciationService(settings)
    t0 = time.perf_counter()
    service.load()
    say(f"model loaded in {time.perf_counter() - t0:.1f} s on {service._device}")

    say("warm-up ...")
    service.assess(*items[0])
    service.assess_batch(items)

    seq_times, batch_times, seq_gpu, batch_gpu = [], [], [], []
    seq_res = batch_res = None
    for i in range(1, args.repeat + 1):
        with GpuMonitor() as gpu:
            t = time.perf_counter()
            seq_res = [service.assess(a, text) for a, text in items]
            seq_times.append(time.perf_counter() - t)
        seq_gpu.append(gpu.summary())
        with GpuMonitor() as gpu:
            t = time.perf_counter()
            batch_res = service.assess_batch(items)
            batch_times.append(time.perf_counter() - t)
        batch_gpu.append(gpu.summary())
        say(f"run {i}/{args.repeat}: sequential {seq_times[-1]:.2f} s   batch of {len(items)} {batch_times[-1]:.2f} s")

    diff = max(abs(a.pronunciation - b.pronunciation) for a, b in zip(seq_res, batch_res))
    seq, batch = median(seq_times), median(batch_times)
    seq_g, batch_g = closest_to_median(seq_times, seq_gpu), closest_to_median(batch_times, batch_gpu)
    result = {
        "sequential_10_audio_seconds": round(seq, 3),
        "batch_10_audio_seconds": round(batch, 3),
        "speedup": round(seq / batch, 2) if batch else None,
        "clips": len(clips), "repeats": args.repeat,
        "audio_seconds_total": round(sum(c.audio.duration_s for c in clips), 1),
        "max_score_difference_batch_vs_sequential": round(diff, 5),
        "sequential_gpu": seq_g, "batch_gpu": batch_g,
        "model": settings.phoneme_model or "accent profile model", "device": service._device,
        "measured_at": now_iso(),
    }
    say("\n=== Phoneme model ===")
    n = len(clips)
    say(f"{n} files one by one   : {seq:7.2f} s   {seq_g}")
    say(f"{n} files, one batch   : {batch:7.2f} s   {batch_g}")
    say(f"speed-up {result['speedup']}x   largest pronunciation-score difference {diff:.5f}")
    if not args.no_save:
        def save(data: dict) -> None:
            data.setdefault("components", {})["phoneme"] = result
        update_results(save)
        say("saved -> benchmarks/benchmark_results.json (components.phoneme)")


if __name__ == "__main__":
    main()
