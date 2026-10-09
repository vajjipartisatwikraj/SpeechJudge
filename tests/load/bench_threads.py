"""How model speed changes with the CPU thread budget (use it to pick SJ_*_THREADS for a server).

    python tests/load/bench_threads.py                 # all three models
    python tests/load/bench_threads.py --only whisper  # whisper | phoneme | qwen
    python tests/load/bench_threads.py --threads 2 3 4 --qwen-threads 2 4 6

Measures, per thread count: Whisper (small.en int8, word timestamps) and the phoneme model on 10 s clips,
and Qwen via Ollama (full output with justifications vs compact scores-only output). Ollama must be running.
Run on the server for server numbers; a laptop run only shows the scaling shape.
"""
from __future__ import annotations

import argparse
import statistics
import time

from _common import ten_second_clips

QA_TEXT = ("On weekends I usually wake up late. Then I go to the park with my friends and we play "
           "football. In the evening I watch a movie with my family.")
QUESTION = "What do you usually do on weekends?"


def bench_whisper(threads: list[int], model_name: str) -> None:
    from faster_whisper import WhisperModel

    clips = ten_second_clips(3)
    print(f"\nWhisper {model_name} int8, 3 clips x 10 s, word timestamps (as the API runs it)")
    for t in threads:
        m = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=t)
        list(m.transcribe(clips[0], language="en", beam_size=5)[0])        # warm-up
        times = []
        for c in clips:
            t0 = time.perf_counter()
            segs, _ = m.transcribe(c, language="en", beam_size=5, temperature=0.0, word_timestamps=True,
                                   condition_on_previous_text=False, vad_filter=False)
            list(segs)
            times.append(time.perf_counter() - t0)
        print(f"  cpu_threads={t}: {statistics.mean(times):5.2f} s per 10 s clip", flush=True)
        del m


def bench_phoneme(threads: list[int]) -> None:
    import numpy as np
    import torch

    from app.core.config import Settings
    from app.services.pronunciation.phoneme_service import PhonemePronunciationService

    clips = ten_second_clips(3)
    svc = PhonemePronunciationService(Settings(_env_file=None, pronunciation_backend="phoneme", device="cpu"))
    svc.load()
    print("\nPhoneme model, 3 clips x 10 s (model forward)")
    for t in threads:
        torch.set_num_threads(t)
        svc._recognize(np.zeros(16000, dtype=np.float32))                   # warm-up for this setting
        times = []
        for c in clips:
            t0 = time.perf_counter()
            svc._recognize(c)
            times.append(time.perf_counter() - t0)
        print(f"  torch threads={t}: {statistics.mean(times):5.2f} s per 10 s clip", flush=True)


def bench_qwen(threads: list[int]) -> None:
    from app.core.config import Settings
    from app.models.domain import QuestionType
    from app.services.language.ollama_service import OllamaLanguageEvaluator

    print("\nQwen3 via Ollama, one Q&A answer (30 words)")
    cases = [(t, True) for t in threads] + [(threads[len(threads) // 2], False)]
    for t, just in cases:
        s = Settings(_env_file=None, qwen_backend="ollama", ollama_num_thread=t, llm_justifications=just)
        svc = OllamaLanguageEvaluator(s)
        svc.load()
        svc._generate([{"role": "user", "content": "Reply with {}"}])        # settle the runner at this thread count
        t0 = time.perf_counter()
        res = svc.evaluate(QuestionType.QUESTION_ANSWER, QUESTION, QA_TEXT)
        secs = time.perf_counter() - t0
        label = "with justifications (default)" if just else "compact scores only"
        print(f"  num_thread={t}, {label:30s}: {secs:5.1f} s   relevance={res.scores()['relevance']:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["whisper", "phoneme", "qwen"])
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 3, 4], help="thread counts for Whisper / phoneme")
    ap.add_argument("--qwen-threads", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--whisper-model", default="small.en")
    args = ap.parse_args()
    if args.only in (None, "whisper"):
        bench_whisper(args.threads, args.whisper_model)
    if args.only in (None, "phoneme"):
        bench_phoneme(args.threads)
    if args.only in (None, "qwen"):
        bench_qwen(args.qwen_threads)


if __name__ == "__main__":
    main()
