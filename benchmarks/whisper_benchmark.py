"""Whisper benchmark: 10 audio files one by one vs the same 10 files as ONE batch.

    python benchmarks/whisper_benchmark.py [--repeat 3] [--clips 10] [--no-save]

Uses the real WhisperService with the settings from speech-judge/.env (SJ_GPU_PRESENT=true gives
cuda, large-v3-turbo and float16 on the GPU server). The model is loaded and
warmed up first, so only inference is timed. Each mode runs --repeat times; the median is reported.
Writes components.whisper in benchmarks/benchmark_results.json.

This process loads its OWN copy of the model: if the VRAM is tight, stop the worker while it runs.
"""
from __future__ import annotations

import argparse
import re
import time

from _common import (GpuMonitor, closest_to_median, load_clips, median, now_iso, preflight, say,
                     update_results)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=3, help="timed repetitions per mode (median reported)")
    ap.add_argument("--clips", type=int, default=10, help="number of benchmark clips (default 10 = one batch of 10)")
    ap.add_argument("--no-save", action="store_true", help="do not write benchmark_results.json")
    args = ap.parse_args()

    from app.core.config import get_settings
    from app.services.stt.whisper_service import WhisperService

    settings = get_settings()
    preflight(settings, ctranslate2_needed=True)
    clips = load_clips(args.clips)
    audios = [c.audio for c in clips]
    say(f"{len(clips)} clips, {sum(a.duration_s for a in audios):.1f} s of audio in total")

    service = WhisperService(settings)
    t0 = time.perf_counter()
    service.load()
    say(f"model loaded in {time.perf_counter() - t0:.1f} s (batched pipeline: {service._batched is not None})")

    say("warm-up ...")
    service.transcribe(audios[0])
    service.transcribe_batch(audios)

    seq_times, batch_times, seq_gpu, batch_gpu = [], [], [], []
    seq_texts = batch_texts = None
    for i in range(1, args.repeat + 1):
        with GpuMonitor() as gpu:
            t = time.perf_counter()
            seq_texts = [service.transcribe(a) for a in audios]
            seq_times.append(time.perf_counter() - t)
        seq_gpu.append(gpu.summary())
        with GpuMonitor() as gpu:
            t = time.perf_counter()
            batch_texts = service.transcribe_batch(audios)
            batch_times.append(time.perf_counter() - t)
        batch_gpu.append(gpu.summary())
        say(f"run {i}/{args.repeat}: sequential {seq_times[-1]:.2f} s   batch of {len(audios)} {batch_times[-1]:.2f} s")

    # the batch must give the same words as one-by-one (a quality check on the batching itself)
    same = sum(words(a.text) == words(b.text) for a, b in zip(seq_texts, batch_texts))
    seq, batch = median(seq_times), median(batch_times)
    seq_g, batch_g = closest_to_median(seq_times, seq_gpu), closest_to_median(batch_times, batch_gpu)
    result = {
        "sequential_10_audio_seconds": round(seq, 3),
        "batch_10_audio_seconds": round(batch, 3),
        "speedup": round(seq / batch, 2) if batch else None,
        "clips": len(clips), "repeats": args.repeat,
        "audio_seconds_total": round(sum(a.duration_s for a in audios), 1),
        "transcripts_identical": f"{same}/{len(clips)}",
        "sequential_gpu": seq_g, "batch_gpu": batch_g,
        "model": settings.whisper_model, "compute_type": settings.whisper_compute_type,
        "device": settings.device, "measured_at": now_iso(),
    }
    say("\n=== Whisper ===")
    n = len(clips)
    say(f"{n} files one by one   : {seq:7.2f} s   {seq_g}")
    say(f"{n} files, one batch   : {batch:7.2f} s   {batch_g}")
    say(f"speed-up {result['speedup']}x   identical transcripts {result['transcripts_identical']}")
    if not args.no_save:
        def save(data: dict) -> None:
            data.setdefault("components", {})["whisper"] = result
        update_results(save)
        say("saved -> benchmarks/benchmark_results.json (components.whisper)")


if __name__ == "__main__":
    main()
