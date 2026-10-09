# Speech Judge (backend)

Stateless evaluation service: it receives audio plus question metadata, runs only the analysis each
question type needs, converts the metrics into a score with a deterministic rule set, and returns JSON.
Architecture: `../ReadMe.md`. Requirements: `../.kiro/specs/speech-judge-service/requirements.md`.

## Quick start (Windows, CPU)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt -r requirements-ml.txt
copy .env.example .env          # then edit .env (credential, models, ports)
ollama pull qwen3:4b            # language model used on CPU (Ollama must be running)

.\scripts\run_api.ps1           # API from .env  (add -Mock for instant mocked services)
.\scripts\run_tests.ps1         # unit + integration tests
```

Frontend test console (reads the same `.env`, so no credentials to copy): `python ..\frontend\server.py`, then
open http://127.0.0.1:8080.

## Folder layout

```
speech-judge/
├── app/
│   ├── main.py                 FastAPI factory (wires middleware, error handlers, routers)
│   ├── __main__.py             `python -m app` -> uvicorn on SJ_HOST:SJ_PORT
│   ├── runtime.py              composition root: settings, services, pipeline, job store
│   ├── api/                    HTTP layer only
│   │   ├── evaluate.py health.py jobs.py     routers
│   │   ├── middleware.py                      auth, body-size and readiness gates
│   │   ├── error_handlers.py validation.py deps.py
│   ├── models/                 domain.py (enums, dataclasses) · schemas.py (API contract)
│   ├── core/                   config.py (env settings) · paths.py · errors.py · logging.py
│   ├── pipeline/               request -> audio -> quality gate -> evaluator -> score -> result
│   ├── audio/                  decoder.py (WAV/MP3/OGG/WebM/M4A -> 16 kHz mono) · quality_gate.py
│   ├── evaluators/             one orchestrator per question type + router.py
│   ├── services/               shared, load-once model services behind interfaces.py
│   │   ├── stt/                faster-whisper
│   │   ├── pronunciation/      phoneme analysis (align, accent profile) · GOPT adapter
│   │   ├── language/           Qwen3 (Ollama or transformers), rubric prompts
│   │   ├── acoustic/           VAD, pitch, pauses, speech rate, fluency, loudness
│   │   ├── text_comparison/    deterministic WER / order comparison
│   │   ├── registry.py         builds the one instance of each service
│   │   └── mocks.py            fixed-output doubles (SJ_MOCK_MODE, tests)
│   ├── scoring/                engine.py (weighted sum) · config.py · normalizers.py · metrics/ (per type)
│   ├── jobs/                   store.py (Redis / in-memory, TTL) · runner.py (retries) · dispatch.py
│   └── workers/                celery_app.py · heartbeat.py
├── config/
│   ├── scoring.yaml            weights, target ranges, intonation (tune here, no code change)
│   └── accents/indian_english.yaml   tolerated sound differences + pronunciation thresholds
├── tests/
│   ├── load/                   concurrency test + Whisper / phoneme / acoustic benchmarks (run by hand), saved baseline
│   ├── unit/                   scoring · audio · jobs · core · services/{acoustic,pronunciation,...}
│   ├── integration/            FastAPI app on mock services (auth, validation, flows, async jobs)
│   ├── e2e/                    real models / running stack; skipped unless `--real`
│   ├── fixtures/               generated voice clips (git-ignored) + make_audio.ps1
│   ├── conftest.py support.py
├── scripts/                    run_api.ps1 · run_worker.ps1 · run_tests.ps1
├── deploy/                     server deployment files: pm2/ecosystem.config.js · scripts/{deploy,wait_ready}.sh · env/m7i-cpu.env.example
├── .env.example  .env          configuration (see below); .env is git-ignored
├── Dockerfile  docker-compose.yml  .dockerignore  Makefile  pyproject.toml
└── requirements.txt  requirements-ml.txt  requirements-dev.txt
```

Dependency direction: `api -> pipeline -> evaluators -> services`, with `models` and `core` usable by all.
Services never import from `api`.

## Configuration

Everything is an environment variable with the `SJ_` prefix. Values come from `speech-judge/.env`
(override the location with `SJ_ENV_FILE`); real environment variables win over the file.
`.env.example` documents every variable. The frontend console reads `SJ_HOST`, `SJ_PORT` and
`SJ_SERVICE_CREDENTIALS` from the same file (plus `FRONTEND_PORT`, optional `BACKEND_URL` / `FRONTEND_API_KEY`).

**CPU or GPU: one switch.** `SJ_GPU_PRESENT=false` (default) selects the CPU architecture, `SJ_GPU_PRESENT=true` the GPU
one. It only chooses defaults (`app/core/hardware.py`); any variable you set explicitly still wins.

| Setting | `false` (CPU) | `true` (GPU) |
|---|---|---|
| `SJ_DEVICE` | `cpu` | `cuda` |
| Whisper model / compute type | `small.en` / `int8` | `large-v3-turbo` / `float16` |
| Whisper and phoneme batch size | 1 / 1 (no batching) | 10 / 10 |
| `SJ_SECTION_PROCESSING_TIMEOUT_S` | 3600 | 900 |
| PM2 worker concurrency (`deploy/pm2/ecosystem.config.js`) | 3 | 2 |

Do not keep an old `SJ_DEVICE` line in `.env`: a value that contradicts the switch (`cuda` with `false`, `cpu`
with `true`) stops the service at start-up. The startup log prints the active profile. Thread budgets, the
process layout (inline vs Celery) and Redis are ordinary settings, see `deploy/env/m7i-cpu.env.example` and
`deploy/env/rtx5070.env.example`.

Production guardrails: with `SJ_ENVIRONMENT=production` the service refuses to start unless credentials are
set and at least 24 characters, `SJ_MOCK_MODE=false` and `SJ_ENABLE_DOCS=false`. `/docs` is off by default.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/health` | no | liveness |
| GET | `/api/v1/ready` | no | per-model readiness (200 / 503) |
| POST | `/api/v1/evaluate` | Bearer | synchronous evaluation |
| POST | `/api/v1/evaluate/async` | Bearer | enqueue job (202) |
| GET | `/api/v1/jobs/{job_id}` | Bearer | job state and result |

Send `Authorization: Bearer <one of SJ_SERVICE_CREDENTIALS>`. Audio is base64 (WAV works without FFmpeg;
other formats need FFmpeg). `question_config.submitted_text` carries a UI-based Jumbled answer.
With `SJ_ENABLE_DOCS=true` the OpenAPI UI is at `/docs`.

## What is measured

- **Pronunciation / completeness** (`SJ_PRONUNCIATION_BACKEND=phoneme`): a wav2vec2 phoneme recogniser
  (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, about 1.2 GB, downloaded on first start) hears what was said; it is
  aligned with the espeak phonemes of the expected sentence using accent-aware costs. Indian-English features
  (th->t, v/w, retroflex t/d, less vowel reduction, variable r, diphthong simplification) are tolerated; the list
  is in `config/accents/indian_english.yaml` and should be reviewed against your speakers.
- **Fluency**: speech rate, pause ratio, hesitation pauses (from Whisper word timings), mean run length, long
  pauses, fillers and repetitions. **Prosody**: pitch variation/range in semitones around the speaker's own
  median, syllable loudness variation, end-of-sentence pitch movement for `.` / `?`. Weights and targets:
  `config/scoring.yaml`.
- **Service switches:** `SJ_PRONUNCIATION_ENABLED=false` and `SJ_QWEN_ENABLED=false` replace that service with a
  mock returning fixed placeholder values (no model is loaded). Reading/Repeat then get the mock pronunciation and
  completeness, Q&A/Storytelling the mock language scores (0.8 per dimension). Every affected result carries a
  `MOCKED_PRONUNCIATION` / `MOCKED_LANGUAGE` flag, and `/api/v1/ready` lists the `mocked` services. Production mode
  refuses to start with either switch off. Useful for demos, CPU-only test servers and isolating latency.
- `gopt` and `mock` are the alternative pronunciation backends (GOPT needs an inference backend you provide,
  contract in `app/services/pronunciation/gopt_service.py`).

Known limits: one wrong sound in a long sentence can be lost in recogniser noise (the per-word table still shows
it); Q&A and Storytelling have no reference text, so no pronunciation score; thresholds are engineering values
until compared with human ratings.

## Tests

```powershell
.\scripts\run_tests.ps1              # unit + integration: fast, no models (109 tests)
.\scripts\run_tests.ps1 -Real        # + e2e: real Whisper / phoneme model / Ollama, and the running stack
pwsh tests\fixtures\make_audio.ps1   # (re)generate the voice clips the e2e tests use
```

The stack tests need the API and `frontend/server.py` running; they skip themselves otherwise.

Performance checks live in `tests/load/` (run by hand, not part of pytest): `concurrency_load.py` against a
running API, and `bench_whisper.py` / `bench_phoneme.py` / `bench_acoustic.py`. Its README holds the saved
baseline numbers from the CPU laptop so you can compare after deploying.

Back-pressure: set `SJ_MAX_CONCURRENT_EVALUATIONS` (for example 12) and synchronous `/evaluate` calls beyond that
return HTTP 429 with `Retry-After`, instead of queueing until they time out. The async endpoint is unaffected.

## Deployment

Two step-by-step AWS EC2 guides (server sizing, IAM, GitHub Actions, start/stop, connecting the local frontend):

- [`docs/EC2_M7I_PM2_DEPLOYMENT.md`](docs/EC2_M7I_PM2_DEPLOYMENT.md): CPU-only `m7i.2xlarge`, no Docker, **PM2**,
  Ollama + Redis, CPU thread budget (`SJ_WHISPER_CPU_THREADS`, `SJ_PHONEME_THREADS`, `SJ_OLLAMA_NUM_THREAD`).
  Process files in `deploy/` (`pm2/ecosystem.config.js`, `scripts/deploy.sh`, `env/m7i-cpu.env.example`).
- [`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md): GPU server with Docker and ECR.

`docker-compose.yml` runs Redis, an API gateway (no models) and a GPU Celery worker that loads Whisper, the
phoneme model and Qwen once. Compose sets production mode, so put a long random `SJ_SERVICE_CREDENTIALS` in `.env`.
Keep port 8000 on a private network. Scale with `docker compose up --scale worker=N`. Jobs and results live in
Redis for `SJ_JOB_RETENTION_S` (default 24 h); nothing permanent is stored.
