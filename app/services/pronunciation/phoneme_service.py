"""Phoneme-based pronunciation assessment (replaces GOPT; same GoptService interface).

1. A wav2vec2 CTC phoneme recogniser (espeak-style IPA output) transcribes what was *said*
   at phoneme level, without a language model that could "correct" mispronunciations.
2. The expected sentence is converted to phonemes with the same espeak system.
3. The two sequences are aligned with accent-aware costs (config/accents/indian_english.yaml).

Scores: pronunciation (weighted phoneme error rate mapped to 0..1) and completeness
(share of expected words actually produced). Fluency and prosody are NOT measured here;
they come from the acoustic engine.

The model runs on ``settings.device`` (cuda when ``SJ_GPU_PRESENT=true``, else cpu) and can recognise several clips in one padded batch
(``assess_batch``), which is what the section worker uses on the GPU.
"""
from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np

from app.core.config import Settings
from app.core.errors import ServiceLoadError
from app.core.logging import get_logger
from app.models.domain import AudioData, GoptResult
from app.services.interfaces import GoptService
from app.services.pronunciation.align import (ExpectedWord, align_phonemes, score_alignment)
from app.services.pronunciation.phonemes import AccentProfile, load_profile, split_token
from app.services.text_comparison.comparison import normalize_words

log = get_logger(__name__)
CHUNK_S = 30.0          # longer audio is recognised in chunks to bound memory
SAMPLE_RATE = 16000


class PhonemePronunciationService(GoptService):
    def __init__(self, settings: Settings, profile: AccentProfile | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._profile = profile
        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._can_pad = False                   # padded batches need a layer-norm wav2vec2 + attention mask
        self._id_to_token: dict[int, str] = {}
        self._lock = threading.Lock()           # torch model + espeak are not thread-safe
        self._word_cache: dict[str, list[str]] = {}

    # ---- loading ----------------------------------------------------------------------
    def load(self) -> None:
        try:
            if self._profile is None:
                self._profile = load_profile(self._settings.accent_profile_path)
        except Exception as exc:  # noqa: BLE001
            raise ServiceLoadError(self.name, f"cannot read accent profile: {exc}") from exc
        try:
            import espeakng_loader  # type: ignore
            import torch  # type: ignore
            from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
        except ImportError as exc:
            raise ServiceLoadError(self.name, f"missing dependency ({exc}); "
                                   "pip install torch transformers phonemizer espeakng-loader") from exc
        if self._settings.device == "cuda":     # fail loudly instead of silently running on the CPU
            from app.core.gpu import torch_cuda_problem
            problem = torch_cuda_problem(torch)
            if problem:
                raise ServiceLoadError(self.name, problem)
            self._device = "cuda"
        try:
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            name = self._settings.phoneme_model or self._profile.model
            self._processor = Wav2Vec2Processor.from_pretrained(name)
            self._model = Wav2Vec2ForCTC.from_pretrained(name).eval().to(self._device)
            vocab = self._processor.tokenizer.get_vocab()
            self._id_to_token = {i: t for t, i in vocab.items()}
            self._torch = torch
            self._can_pad = getattr(self._model.config, "feat_extract_norm", "group") == "layer"
            if self._settings.phoneme_threads > 0:       # CPU thread budget (0 = PyTorch default)
                torch.set_num_threads(self._settings.phoneme_threads)
            self._phonemize_word("warmup")       # initialises espeak
            self._recognize(np.zeros(SAMPLE_RATE, dtype=np.float32))   # warm up the model
            if self._can_pad:                    # and the padded-batch path (two different lengths)
                self._recognize_batch([np.zeros(SAMPLE_RATE, dtype=np.float32),
                                       np.zeros(SAMPLE_RATE * 2, dtype=np.float32)])
        except Exception as exc:  # noqa: BLE001
            raise ServiceLoadError(self.name, f"could not load phoneme model: {exc}") from exc
        self._loaded = True

    # ---- seams (overridden in unit tests) ---------------------------------------------
    def _phonemize_word(self, word: str) -> list[str]:
        if word not in self._word_cache:
            with self._lock:
                text = self._processor.tokenizer.phonemize(word)
            # the tokenizer appends its word-separator "|" to each phonemized word; drop it
            self._word_cache[word] = [t for t in text.split() if t != "|"]
        return self._word_cache[word]

    def _ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        """Collapse CTC repeats and drop blank / special tokens."""
        tokens: list[str] = []
        prev = None
        for i in ids:
            if i != prev:
                tok = self._id_to_token.get(i, "")
                if tok and not tok.startswith("<") and tok != "|":
                    tokens.append(tok)
            prev = i
        return tokens

    def _recognize(self, samples: np.ndarray) -> list[str]:
        """Greedy CTC decode of ONE clip -> list of phoneme tokens, as the model heard them."""
        torch = self._torch
        step = int(CHUNK_S * SAMPLE_RATE)
        tokens: list[str] = []
        for start in range(0, max(len(samples), 1), step):
            chunk = samples[start:start + step]
            if len(chunk) < 1600:        # <0.1 s tail: nothing to recognise
                continue
            inputs = self._processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with self._lock, torch.inference_mode():
                logits = self._model(inputs.input_values.to(self._device)).logits
            tokens.extend(self._ids_to_tokens(torch.argmax(logits, dim=-1)[0].tolist()))
        return tokens

    def _recognize_group(self, chunks: Sequence[np.ndarray]) -> list[list[str]]:
        """Recognise several chunks (each <= 30 s) in ONE padded forward pass."""
        torch = self._torch
        feats = self._processor.feature_extractor(
            [np.asarray(c, dtype=np.float32) for c in chunks], sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True, return_attention_mask=True)
        mask = feats.get("attention_mask")
        if mask is None:            # cannot tell padding from speech: stay exact, go one by one
            return [self._recognize(c) for c in chunks]
        with self._lock, torch.inference_mode():
            mask = mask.to(self._device)
            logits = self._model(feats["input_values"].to(self._device), attention_mask=mask).logits
            ids = torch.argmax(logits, dim=-1).cpu().tolist()
            lengths = self._model._get_feat_extract_output_lengths(mask.sum(-1)).cpu().tolist()
        return [self._ids_to_tokens(ids[b][:int(lengths[b])]) for b in range(len(chunks))]

    def _recognize_batch(self, clips: Sequence[np.ndarray]) -> list[list[str]]:
        """Phoneme tokens for every clip, recognising up to ``phoneme_batch_size`` chunks per pass."""
        if not self._can_pad:
            return [self._recognize(c) for c in clips]
        step = int(CHUNK_S * SAMPLE_RATE)
        chunks: list[tuple[int, np.ndarray]] = []
        for owner, samples in enumerate(clips):
            for start in range(0, max(len(samples), 1), step):
                chunk = samples[start:start + step]
                if len(chunk) >= 1600:
                    chunks.append((owner, chunk))
        out: list[list[str]] = [[] for _ in clips]
        size = max(1, self._settings.phoneme_batch_size)
        for g in range(0, len(chunks), size):
            group = chunks[g:g + size]
            for (owner, _), tokens in zip(group, self._recognize_group([c for _, c in group])):
                out[owner].extend(tokens)
        return out

    # ---- assessment -------------------------------------------------------------------
    def _expected_words(self, expected_text: str) -> list[ExpectedWord]:
        assert self._profile is not None, "service not loaded"
        words: list[ExpectedWord] = []
        for w in normalize_words(expected_text):
            units = [u for tok in self._phonemize_word(w) for u in split_token(tok, self._profile)]
            words.append(ExpectedWord(w, units))
        return words

    def _score(self, expected_text: str, heard_tokens: list[str], inference_s: float,
               batch_size: int = 1) -> GoptResult:
        profile = self._profile
        assert profile is not None, "service not loaded"
        words = self._expected_words(expected_text)
        heard = [u.symbol for tok in heard_tokens for u in split_token(tok, profile)]

        ops = align_phonemes(words, heard, profile)
        scores = score_alignment(words, ops, len(heard), profile)

        return GoptResult(
            pronunciation=scores.pronunciation, fluency=None, prosody=None,
            completeness=scores.completeness, words=scores.words,
            phonemes=[{"word_index": op.word_index, "expected": op.expected, "heard": op.heard,
                       "status": op.status, "cost": round(op.cost, 3)} for op in scores.ops],
            method="phoneme_ctc",
            details={
                **scores.extra,
                "accent_profile": profile.name,
                "model": self._settings.phoneme_model or profile.model,
                "weighted_per": scores.weighted_per, "raw_per": scores.raw_per,
                "expected_phonemes": " | ".join(" ".join(u.symbol for u in w.units) for w in words),
                "heard_phonemes": " ".join(heard_tokens),
                "inference_s": round(inference_s, 2),
                "batch_size": batch_size,
            })

    def assess(self, audio: AudioData, expected_text: str) -> GoptResult:
        t0 = time.perf_counter()
        heard_tokens = self._recognize(audio.samples)
        return self._score(expected_text, heard_tokens, time.perf_counter() - t0)

    def assess_batch(self, items: Sequence[tuple[AudioData, str]]) -> list[GoptResult]:
        """Assess many clips with ONE padded forward pass per ``phoneme_batch_size`` chunks."""
        if len(items) == 1:
            return [self.assess(*items[0])]
        t0 = time.perf_counter()
        heard = self._recognize_batch([audio.samples for audio, _ in items])
        per_item = (time.perf_counter() - t0) / max(len(items), 1)
        return [self._score(text, tokens, per_item, batch_size=len(items))
                for (_, text), tokens in zip(items, heard)]
