"""Hardware profiles: one switch (``SJ_GPU_PRESENT``) picks sensible defaults for the whole stack.

    SJ_GPU_PRESENT=false   CPU-only server (laptop, EC2 m7i/c7i ...)
    SJ_GPU_PRESENT=true    server with an NVIDIA GPU (RTX 5070, L4, A10G ...)

A profile only supplies *defaults*. Any setting that is given explicitly (environment variable or ``.env``,
for example ``SJ_WHISPER_MODEL``) always wins, so a profile never takes control away from the operator.
Everything else that differs between the two architectures (thread budgets, process layout, Redis, ...)
stays an ordinary setting; see ``deploy/env/*.env.example``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    device: str                        # "cpu" | "cuda" for Whisper and the phoneme model
    whisper_model: str
    whisper_compute_type: str
    qwen_backend: str                  # Ollama uses the GPU on its own when there is one
    whisper_batch_size: int            # clips decoded together by the shared Whisper stage
    phoneme_batch_size: int            # clips recognised together by the shared phoneme stage
    section_processing_timeout_s: float  # a section job running longer than this is reported "Worker lost"

    def defaults(self) -> dict[str, object]:
        """Settings fields this profile provides (``name`` is a label, not a setting)."""
        values = asdict(self)
        values.pop("name")
        return values


CPU_PROFILE = HardwareProfile(
    name="cpu",
    device="cpu",
    whisper_model="small.en",
    whisper_compute_type="int8",
    qwen_backend="ollama",
    # Batching brought no speed-up for the phoneme model on CPU and the models are serialised by locks
    # anyway, so every clip is handled on its own.
    whisper_batch_size=1,
    phoneme_batch_size=1,
    # Qwen answers take about a minute each on CPU: a 24-question Q&A section needs ~25 minutes.
    section_processing_timeout_s=3600.0,
)

GPU_PROFILE = HardwareProfile(
    name="gpu",
    device="cuda",
    whisper_model="large-v3-turbo",
    whisper_compute_type="float16",
    qwen_backend="ollama",
    whisper_batch_size=10,
    phoneme_batch_size=10,
    section_processing_timeout_s=900.0,
)


def profile_for(gpu_present: bool) -> HardwareProfile:
    return GPU_PROFILE if gpu_present else CPU_PROFILE
