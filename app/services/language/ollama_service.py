"""Qwen3 served by Ollama (quantized GGUF, runs on CPU). Same validation/retry logic as the
transformers backend; only model loading and generation differ."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from app.core.config import Settings
from app.core.errors import ServiceLoadError
from app.services.language.qwen_service import QwenLanguageEvaluator


class OllamaLanguageEvaluator(QwenLanguageEvaluator):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base = settings.ollama_url.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(self._base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured URL
            return json.loads(resp.read())

    def load(self) -> None:
        s = self._settings
        try:
            # Loads the model into memory and keeps it resident.
            self._post("/api/chat", {"model": s.ollama_model, "messages": [], "keep_alive": -1,
                                     "stream": False}, timeout=s.ollama_timeout_s)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ServiceLoadError(self.name, f"Ollama model '{s.ollama_model}' unavailable at "
                                   f"{s.ollama_url}: {exc}") from exc
        self._loaded = True

    def _options(self) -> dict:
        s = self._settings
        options = {"temperature": 0, "seed": 0, "num_predict": s.qwen_max_new_tokens}
        if s.ollama_num_thread > 0:        # CPU thread budget for this model (0 = Ollama decides)
            options["num_thread"] = s.ollama_num_thread
        return options

    def _generate(self, messages: list[dict]) -> str:
        s = self._settings
        data = self._post("/api/chat", {
            "model": s.ollama_model, "messages": messages, "stream": False, "think": False,
            "format": "json", "keep_alive": -1,
            "options": self._options(),
        }, timeout=s.ollama_timeout_s)
        return data.get("message", {}).get("content", "")
