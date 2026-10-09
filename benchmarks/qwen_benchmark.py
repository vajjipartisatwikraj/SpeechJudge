"""Qwen benchmark: real production Q&A and Storytelling evaluation prompts, one call at a time.

    python benchmarks/qwen_benchmark.py [--repeat 1] [--clips 10] [--types qa,story] [--no-save]

Uses the real language evaluator from speech-judge/.env (SJ_QWEN_BACKEND=ollama or transformers) with
the exact prompt, rubric and JSON validation of the service. For every benchmark clip it evaluates the
spoken text as a Q&A answer and as a Storytelling answer. Model load and one warm-up call are not timed.
Writes components.qwen in benchmarks/benchmark_results.json.

With the ollama backend Ollama runs the model, so make sure it is running and that
SJ_OLLAMA_MODEL is pulled. GPU utilisation / VRAM are read system-wide, which includes Ollama.
This process loads its own copy of a transformers model: if the VRAM is tight, stop the worker first.
"""
from __future__ import annotations

import argparse
import statistics
import time

from _common import GpuMonitor, load_clips, now_iso, preflight, say, update_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1, help="rounds over all clips")
    ap.add_argument("--clips", type=int, default=10, help="number of benchmark clips")
    ap.add_argument("--types", default="qa,story", help="comma list: qa, story")
    ap.add_argument("--no-save", action="store_true", help="do not write benchmark_results.json")
    args = ap.parse_args()

    from app.core.config import get_settings
    from app.models.domain import QuestionType
    from app.services.registry import _build_language

    settings = get_settings()
    wanted = {t.strip() for t in args.types.split(",") if t.strip()}
    cases = []                                   # (label, QuestionType, prompt attribute)
    if "qa" in wanted:
        cases.append(("qa", QuestionType.QUESTION_ANSWER, "question"))
    if "story" in wanted:
        cases.append(("story", QuestionType.STORYTELLING, "story_prompt"))
    if not cases:
        raise SystemExit("--types must contain qa and/or story")

    preflight(settings, torch_needed=settings.qwen_backend == "transformers")
    if not settings.qwen_enabled:
        raise SystemExit("SJ_QWEN_ENABLED=false: the language evaluator is mocked, nothing to benchmark")
    clips = load_clips(args.clips, decode=False)

    service = _build_language(settings)
    t0 = time.perf_counter()
    service.load()
    say(f"{settings.qwen_backend} model loaded in {time.perf_counter() - t0:.1f} s")

    say("warm-up ...")
    for _, qtype, attr in cases:
        service.evaluate(qtype, getattr(clips[0], attr), clips[0].text)

    latencies: dict[str, list[float]] = {label: [] for label, _, _ in cases}
    with GpuMonitor() as gpu:
        for round_no in range(1, args.repeat + 1):
            for clip in clips:
                for label, qtype, attr in cases:
                    t = time.perf_counter()
                    result = service.evaluate(qtype, getattr(clip, attr), clip.text)
                    latencies[label].append(time.perf_counter() - t)
                    assert result.dimensions, "empty evaluation"
            say(f"round {round_no}/{args.repeat} done ({sum(len(v) for v in latencies.values())} calls)")
    everything = [x for v in latencies.values() for x in v]
    summary = gpu.summary()

    result = {
        "average_latency_seconds": round(statistics.mean(everything), 3),
        "median_latency_seconds": round(statistics.median(everything), 3),
        "max_latency_seconds": round(max(everything), 3),
        "calls": len(everything),
        **{f"{label}_average_seconds": round(statistics.mean(v), 3) for label, v in latencies.items()},
        **summary,
        "backend": settings.qwen_backend,
        "model": settings.ollama_model if settings.qwen_backend == "ollama" else settings.qwen_model_path,
        "justifications": settings.llm_justifications, "device": settings.device, "measured_at": now_iso(),
    }
    say("\n=== Qwen ===")
    say(f"{len(everything)} calls: average {result['average_latency_seconds']:.2f} s, median "
        f"{result['median_latency_seconds']:.2f} s, max {result['max_latency_seconds']:.2f} s")
    for label, values in latencies.items():
        say(f"  {label:5s} average {statistics.mean(values):.2f} s")
    say(f"GPU {summary}")
    if not args.no_save:
        def save(data: dict) -> None:
            data.setdefault("components", {})["qwen"] = result
        update_results(save)
        say("saved -> benchmarks/benchmark_results.json (components.qwen)")


if __name__ == "__main__":
    main()
