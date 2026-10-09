# Load and benchmark scripts

Reusable performance checks. They are **not** part of the pytest run (names don't start with `test_`);
run them by hand, ideally on the machine you are sizing. All need the generated voice clips:

```powershell
pwsh tests/fixtures/make_audio.ps1
```

| Script | What it measures | Needs |
|---|---|---|
| `concurrency_load.py` | N simultaneous requests (N = 1..max) against a **running API**: wall time, per-request latency, jobs/min, HTTP 429 count, CPU, free RAM. Types: `reading`, `repeat`, `jumbled`, `qa`, `story`, or `exam` (mix 8 / 16 / 10 / 24 / 2). | API running + credential |
| `bench_whisper.py` | Whisper: 10 clips x 10 s, sequential vs batched, with and without word timestamps. | `faster-whisper` |
| `bench_phoneme.py` | Pronunciation (wav2vec2) model: 10 clips x 10 s, batch 1 / 2 / 5 / 10, output equality check. | `torch`, `transformers`, `phonemizer` |
| `bench_acoustic.py` | Acoustic engine: 3-15 s clips, ms per clip, sequential vs threads. | nothing extra |
| `bench_threads.py` | Speed of Whisper, the pronunciation model and Qwen (Ollama) at different CPU thread counts, plus Qwen full vs compact output. Use it to choose `SJ_WHISPER_CPU_THREADS`, `SJ_PHONEME_THREADS`, `SJ_OLLAMA_NUM_THREAD`. | models, Ollama running |

```powershell
# from speech-judge/
python tests\load\concurrency_load.py --max 15 --out tests\load\results\my_run.json
python tests\load\concurrency_load.py --type exam --max 20
python tests\load\concurrency_load.py --url http://<server>:8000 --token <credential> --type repeat
python tests\load\bench_whisper.py --threads 4
python tests\load\bench_phoneme.py --threads 4
python tests\load\bench_acoustic.py
```

Tips: run `concurrency_load.py` **on the server** (CPU/RAM columns describe the machine running the script).
With `SJ_MAX_CONCURRENT_EVALUATIONS` set, levels above the limit show HTTP 429 instead of growing latency;
that is the back-pressure working. Qwen-backed types (`qa`, `story`, `exam`) are slow on CPU: use a small `--max`.
With `SJ_QWEN_ENABLED=false` / `SJ_PRONUNCIATION_ENABLED=false` those services are mocked, which isolates
the cost of the remaining pipeline.

## GPU server benchmarks (`benchmarks/`, RTX 5070)

Simple scripts for the GPU server; results go to `benchmarks/benchmark_results.json`. The 10 audio files
(about 10 s each) are in `benchmark_audio/` (`manifest.json` holds the sentences, questions and 10 Jumbled items).
Set up the server first: `deploy/env/rtx5070.env.example` and `requirements-gpu.txt`.

| Script | What it measures |
|---|---|
| `whisper_benchmark.py` | Whisper: 10 files one by one vs ONE batch of 10; time, GPU utilisation, VRAM; checks the batch gives the same words. |
| `phoneme_benchmark.py` | Phoneme model: same comparison; checks the scores are the same. |
| `qwen_benchmark.py` | Qwen on real Q&A and Story evaluation prompts: average latency, GPU utilisation, VRAM. |
| `section_benchmark.py` | The real pipeline: 10 questions per section through `POST /api/v1/evaluate/section` + polling `GET /api/v1/jobs/<id>`; total time and average per question. `--section reading\|repeat\|jumbled\|question_answer\|story_telling\|all`. |

```bash
# on the GPU server, from speech-judge/ (API + worker + Redis + Ollama running for the section benchmark)
python benchmarks/whisper_benchmark.py           # --repeat 3 (median), --no-save
python benchmarks/phoneme_benchmark.py
python benchmarks/qwen_benchmark.py
python benchmarks/section_benchmark.py --section all          # --repeat N, --jumbled-mode spoken, --base-url, --token
```

* The three model scripts load their **own** copy of the model on the GPU (Whisper large-v3 about 3 GB, phoneme about
  1.5 GB, Qwen via Ollama about 3 GB). With the worker also running, 12 GB is enough for one copy of everything plus one
  extra model; stop the worker if the memory is tight.
* `section_benchmark.py` times from the moment it sends the request until the job is COMPLETED (so it includes upload,
  queue and polling every 0.25 s) and also prints the worker's own `processing_s`. Run it on the server (127.0.0.1) to
  leave the network out. A 2-question warm-up per section is sent first and not counted.
* The scripts check the GPU first and say what is wrong when it cannot be used (RTX 50 series needs a CUDA 12.8 build of
  PyTorch and CTranslate2 >= 4.6). With `SJ_GPU_PRESENT=false` they still run but warn that the numbers are CPU numbers.
* From the test console: set `SJ_BENCHMARKS_ENABLED=true` on the server and use the **Benchmarks** tab; it starts the same
  scripts through `POST /api/v1/benchmarks/run`. `deploy.sh` keeps `benchmark_results.json` across deployments.
* The audio is synthetic text-to-speech: right for timing, not for judging scoring accuracy.

## Baseline (saved for comparison)

Machine: Windows laptop, Intel i5-1235U (10 cores / 12 threads), 16 GB RAM, **CPU only**, Whisper `small.en`
int8, phoneme model on CPU, backend running with `SJ_JOB_BACKEND=inline`. Measured 2026-10-06.

**Concurrency, Repeat question (3.8 s audio, no Qwen):**

| N at once | wall s | fastest s | slowest s | jobs/min |
|---|---|---|---|---|
| 1 | 3.6 | 3.6 | 3.6 | 16.9 |
| 2 | 6.8 | 3.9 | 6.8 | 17.7 |
| 4 | 16.6 | 5.9 | 16.6 | 14.4 |
| 8 | 41.4 | 7.8 | 41.4 | 11.6 |
| 10 | 52.6 | 8.6 | 52.6 | 11.4 |
| 15 | 95.7 | 8.4 | 95.7 | 9.4 |

All requests succeeded; CPU at 100 % even for N = 1; free RAM stayed above 5.8 GB. Requests queue behind
the model locks: only 1 runs without waiting. Throughput peaks around 17 jobs/min and falls with more threads.

**Whisper `small.en` int8, 4 threads, 10 clips x 10 s:** sequential with word timestamps 57.1 s (5.7 s/clip);
sequential without 50.1 s; sequential greedy 40.4 s; **one batch of 10: 38.0 s (3.8 s/clip)**; batches of 5: 40.3 s.
Batching gives about 1.5x and drops word timestamps.

**Phoneme model, 4 threads, 10 clips x 10 s:** batch 1: 47.1 s (4.71 s/clip); batch 2: 46.3 s; batch 5: 48.2 s;
batch 10: 47.3 s. No speed-up from batching on CPU; outputs identical. The model is 99 % of `assess()`.

**Acoustic engine:** 3 s: 6 ms, 5 s: 11 ms, 7 s: 15 ms, 10 s: 21 ms, 12 s: 25 ms, 15 s: 31 ms per clip
(about 470x real time). 10 clips x 10 s: 216 ms sequential, 185 ms with 10 threads.

**Open answers (Qwen3-4B via Ollama, CPU):** about 40-50 s for a Q&A answer, about 57 s for a story.

**Thread scaling (`bench_threads.py`, same laptop, machine otherwise idle, measured later the same day):**

| Model | 2 threads | 3 threads | 4 threads | 6 threads |
|---|---|---|---|---|
| Whisper `small.en`, s per 10 s clip (word timestamps) | 9.6 | 8.5 | 8.0 | – |
| Pronunciation model, s per 10 s clip | 5.2 | 4.9 | 4.4 | – |
| Qwen3-4B Q&A answer, full output, s | 72.8 | – | 57.1 | 50.2 |
| Qwen3-4B Q&A answer, **compact scores-only**, s | – | – | **29.4** | – |

Reading of the table: Whisper and the pronunciation model barely speed up from 2 to 4 threads (about 1.2x), while
Qwen gains the most from extra threads (2 to 6 threads: 1.45x), and compact output halves Qwen's time at any thread count.
Whisper on this laptop varies run to run (5.7 s/clip in the earlier batching test vs 8.0 s here), so treat these as a shape, not exact values.

Re-run on the target server and compare; GPU numbers for Qwen, Whisper and the phoneme model have not been
measured yet.
