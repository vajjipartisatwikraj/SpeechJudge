"""Qwen3-4B language evaluator (Requirement 13).

Deterministic decoding (no sampling), schema-validated JSON output, bounded retries.
It receives only the transcript and question context - never acoustic data.
"""
from __future__ import annotations

import json
import math
import threading
from typing import Any

from app.core.config import Settings
from app.core.errors import ServiceLoadError, StageError
from app.core.logging import get_logger
from app.models.domain import LanguageResult, QuestionType
from app.services.interfaces import LanguageEvaluator
from app.services.language.prompts import build_messages, rubric_for

log = get_logger(__name__)

INVALID_OUTPUT_REASON = "Language evaluation output invalid"


def parse_language_output(text: str, rubric: tuple[str, ...], justifications: bool = True
                          ) -> dict[str, dict[str, Any]] | None:
    """Return validated {dimension: {score, justification}} or None if schema-invalid.

    With ``justifications=False`` (compact mode) each dimension may simply be a number."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, dict[str, Any]] = {}
    for dim in rubric:
        item = data.get(dim)
        why: Any = ""
        if isinstance(item, dict):
            score, why = item.get("score"), item.get("justification", "" if not justifications else None)
        elif not justifications:
            score = item                      # compact: {"relevance": 0.8, ...}
        else:
            return None
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            return None
        if not 0.0 <= float(score) <= 1.0 or not isinstance(why, str):
            return None
        out[dim] = {"score": float(score), "justification": why.strip()[:300]}
    return out

class QwenLanguageEvaluator(LanguageEvaluator):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ServiceLoadError(self.name, "torch/transformers are not installed "
                                   "(pip install -r requirements-ml.txt)") from exc
        s = self._settings
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(s.qwen_model_path)
            dtype = torch.bfloat16 if s.device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                s.qwen_model_path, torch_dtype=dtype,
                device_map=s.device if s.device == "cuda" else None)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise ServiceLoadError(self.name, f"could not load model: {exc}") from exc
        self._loaded = True

    # Single seam for generation so retry/validation logic is testable without a model.
    def _generate(self, messages: list[dict]) -> str:
        import torch  # type: ignore

        assert self._model is not None and self._tokenizer is not None
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        with self._lock, torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self._settings.qwen_max_new_tokens,
                                          do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def evaluate(self, question_type: QuestionType, question: str, transcript: str
                 ) -> LanguageResult:
        rubric = rubric_for(question_type)
        messages = build_messages(question_type, question, transcript, self._settings.llm_justifications)
        attempts = 1 + max(0, self._settings.llm_retry_limit)
        for attempt in range(1, attempts + 1):
            parsed = parse_language_output(self._generate(messages), rubric, self._settings.llm_justifications)
            if parsed is not None:
                return LanguageResult(self._settings.prompt_version, parsed)
            log.warning("Language evaluator output failed schema validation (attempt %d/%d)",
                        attempt, attempts)
        raise StageError("language", INVALID_OUTPUT_REASON)
