"""Environment-driven configuration (Requirement 20.3).

Every setting is an environment variable with the ``SJ_`` prefix and can be placed in the
backend ``.env`` file (see ``.env.example``). Real environment variables override the file.
Set ``SJ_ENV_FILE`` to load a different file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.hardware import profile_for
from app.core.paths import (BACKEND_DIR, DEFAULT_ACCENT_PROFILE, DEFAULT_ENV_FILE,
                            DEFAULT_SCORING_CONFIG)

MIN_PRODUCTION_CREDENTIAL_LEN = 24


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SJ_", env_file=os.environ.get("SJ_ENV_FILE", str(DEFAULT_ENV_FILE)),
        env_file_encoding="utf-8", extra="ignore")

    # --- Server ----------------------------------------------------------------------------
    environment: str = "local"           # "local" | "production"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    enable_docs: bool = False            # serve /docs and /openapi.json (keep off in production)

    # --- Access control (Requirement 3) ----------------------------------------------------
    # Comma-separated list of credentials the Main Application may present as a Bearer token.
    service_credentials: str = ""

    # --- Mode and backends -----------------------------------------------------------------
    mock_mode: bool = False
    job_backend: str = "celery"          # "celery" (Redis workers) | "inline" (in-process threads)
    # True: this API process loads the models (serves sync /evaluate, reports its own readiness).
    # False ("gateway"): the API only enqueues; readiness comes from worker heartbeats in Redis.
    load_models_in_api: bool = True

    # --- Hardware ----------------------------------------------------------------------------
    # THE switch between the CPU and the GPU architecture. It selects the defaults of the settings
    # marked "(profile)" below, see app/core/hardware.py. Anything set explicitly still wins.
    gpu_present: bool = False
    device: str = "cpu"                  # (profile) "cpu" | "cuda"; leave unset, it follows gpu_present

    # --- Speech-to-text (Whisper) ----------------------------------------------------------
    whisper_model: str = "small.en"      # (profile)
    whisper_compute_type: str = "int8"   # (profile)
    whisper_language: str = "en"
    whisper_beam_size: int = 3           # decoding beam: lower = faster (1 = greedy), higher = slightly more accurate

    # --- CPU thread budget (0 = each library's default) --------------------------------------
    # On a CPU-only server give every model its own share so they don't fight over cores.
    # See docs/EC2_M7I_PM2_DEPLOYMENT.md for recommended splits.
    whisper_cpu_threads: int = 0     # faster-whisper / CTranslate2
    phoneme_threads: int = 0         # PyTorch (pronunciation model)
    ollama_num_thread: int = 0       # Ollama (Qwen); sent as the num_thread option

    # --- Service switches ------------------------------------------------------------------
    # false -> that service is replaced by a mock with fixed placeholder values (no model is
    # loaded) and every result that used it carries a MOCKED_* flag. Not allowed in production.
    pronunciation_enabled: bool = True
    qwen_enabled: bool = True

    # --- Pronunciation ---------------------------------------------------------------------
    # "phoneme" (wav2vec2 phoneme recogniser + accent profile) | "gopt" | "mock"
    pronunciation_backend: str = "phoneme"
    phoneme_model: str = ""              # empty -> the model named in the accent profile
    accent_profile_path: str = str(DEFAULT_ACCENT_PROFILE)
    gopt_backend: str = ""               # "package.module:Class" when pronunciation_backend=gopt
    gopt_checkpoint: str = ""

    # --- Language evaluator (Qwen3) --------------------------------------------------------
    qwen_backend: str = "ollama"         # (profile) "ollama" (quantised, CPU or GPU) | "transformers" (GPU, ~8 GB)
    qwen_model_path: str = "Qwen/Qwen3-4B"
    qwen_max_new_tokens: int = 700
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_s: float = 300.0
    llm_retry_limit: int = 2
    prompt_version: str = "v1"
    # false -> Qwen returns bare scores without a sentence of justification per dimension: roughly a
    # quarter of the generated tokens, so much faster on CPU. The UI then shows empty justifications.
    llm_justifications: bool = True

    # --- Audio limits and quality gate (Requirements 4.7, 6) -------------------------------
    max_audio_bytes: int = 15 * 1024 * 1024
    max_audio_seconds: float = 180.0
    ffmpeg_path: str = "ffmpeg"
    min_duration_s: float = 1.0
    min_speech_s: float = 0.5
    snr_threshold_db: float = 10.0
    clipping_threshold: float = 0.01

    # --- Sections: whole exam sections in one request ----------------------------------------
    section_max_questions: int = 50
    section_max_total_audio_bytes: int = 120 * 1024 * 1024    # all audio of one section together
    section_processing_timeout_s: float = 900.0               # (profile) a section runs much longer than one question
    # Questions of one section evaluated at the same time. Keep >= the batch sizes below, otherwise a
    # batch can never fill up.
    section_question_parallelism: int = 16
    # Threads that wait on the shared stage queues for the section pipeline (cheap: they only block).
    section_pipeline_threads: int = 96

    # --- Shared stage queues / micro-batching (section worker) -------------------------------
    # One persistent Whisper, one phoneme model and one Qwen serve every student. Requests wait in
    # an in-process queue per model; the Whisper and phoneme workers take up to *_batch_size items
    # at once (1 = no batching). A partial batch is dispatched after batch_max_wait_ms.
    whisper_batch_size: int = 1          # (profile)
    phoneme_batch_size: int = 1          # (profile)
    batch_max_wait_ms: int = 300
    qwen_stage_workers: int = 1          # concurrent Qwen requests (>1 only helps with OLLAMA_NUM_PARALLEL)

    # --- Benchmarks (GPU server) -------------------------------------------------------------
    # Lets the authenticated API start the scripts in benchmarks/ (used by the test console).
    # Off by default: a benchmark loads models on the GPU and sends real load through the service.
    benchmarks_enabled: bool = False

    # --- Scoring ---------------------------------------------------------------------------
    scoring_config_path: str = str(DEFAULT_SCORING_CONFIG)

    # --- Timeouts and jobs -----------------------------------------------------------------
    request_timeout_s: float = 60.0
    # Back-pressure: synchronous evaluations allowed in flight at once. Beyond this the API answers
    # 429 + Retry-After instead of letting requests pile up and time out. 0 = unlimited.
    max_concurrent_evaluations: int = 0
    processing_timeout_s: float = 300.0
    job_retry_limit: int = 2
    job_retention_s: int = 24 * 3600
    redis_url: str = "redis://localhost:6379/0"

    @property
    def credential_set(self) -> frozenset[str]:
        return frozenset(c.strip() for c in self.service_credentials.split(",") if c.strip())

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def hardware_profile(self) -> str:
        return profile_for(self.gpu_present).name

    @model_validator(mode="after")
    def _apply_hardware_profile(self) -> "Settings":
        """Fill every setting the operator did not give explicitly from the CPU or GPU profile."""
        explicit = set(self.model_fields_set)          # snapshot: setattr below adds to the set
        profile = profile_for(self.gpu_present)
        for name, value in profile.defaults().items():
            if name not in explicit:
                setattr(self, name, value)

        # An explicit device that contradicts the switch is a configuration mistake (typically an old
        # SJ_DEVICE line left in .env): refuse to start instead of silently running on the wrong hardware.
        device = self.device.strip().lower()
        if self.gpu_present and device != "cuda":
            raise ValueError(f"SJ_GPU_PRESENT=true but SJ_DEVICE={self.device!r}: remove the SJ_DEVICE line "
                             "(the GPU profile uses cuda) or set SJ_GPU_PRESENT=false")
        if not self.gpu_present and device == "cuda":
            raise ValueError("SJ_DEVICE=cuda but SJ_GPU_PRESENT=false: set SJ_GPU_PRESENT=true on a GPU "
                             "server and remove the SJ_DEVICE line")
        return self

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        """Relative config paths are relative to the backend folder, not the current directory."""
        for name in ("scoring_config_path", "accent_profile_path"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                setattr(self, name, str(BACKEND_DIR / path))
        return self

    @model_validator(mode="after")
    def _production_guardrails(self) -> "Settings":
        """Fail fast on unsafe production configuration instead of running exposed."""
        if not self.is_production:
            return self
        problems = []
        creds = self.credential_set
        if not creds:
            problems.append("SJ_SERVICE_CREDENTIALS is empty")
        elif any(len(c) < MIN_PRODUCTION_CREDENTIAL_LEN for c in creds):
            problems.append(f"every credential must be at least {MIN_PRODUCTION_CREDENTIAL_LEN} characters")
        if self.mock_mode:
            problems.append("SJ_MOCK_MODE must be false in production")
        if not self.pronunciation_enabled:
            problems.append("SJ_PRONUNCIATION_ENABLED must be true in production (it would return fake scores)")
        if not self.qwen_enabled:
            problems.append("SJ_QWEN_ENABLED must be true in production (it would return fake scores)")
        if self.enable_docs:
            problems.append("SJ_ENABLE_DOCS must be false in production")
        if problems:
            raise ValueError("Unsafe production configuration: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
