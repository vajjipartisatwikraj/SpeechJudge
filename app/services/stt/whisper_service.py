"""faster-whisper STT backend (Requirement 9). Model is loaded once per process."""
from __future__ import annotations

import threading
from typing import Sequence

from app.core.config import Settings
from app.core.errors import ServiceLoadError
from app.core.logging import get_logger
from app.models.domain import AudioData, Transcription
from app.services.interfaces import STTService
from app.services.stt.batching import MAX_CLIP_S, pack_clips, unpack_segments

log = get_logger(__name__)


class WhisperService(STTService):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._model = None
        self._batched = None
        self._lock = threading.Lock()

    def load(self) -> None:
        s = self._settings
        if s.device == "cuda":
            from app.core.gpu import ctranslate2_cuda_problem, prepare_cuda_runtime
            prepare_cuda_runtime()                  # PyTorch first: its CUDA libraries are then found
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ServiceLoadError(self.name, "faster-whisper is not installed "
                                   "(pip install -r requirements-ml.txt)") from exc
        if s.device == "cuda":
            problem = ctranslate2_cuda_problem()
            if problem:
                raise ServiceLoadError(self.name, problem)
        try:
            extra = {"cpu_threads": s.whisper_cpu_threads} \
                if s.whisper_cpu_threads > 0 else {}     # CPU thread budget (0 = library default)
            self._model = WhisperModel(s.whisper_model, device=s.device,
                                       compute_type=s.whisper_compute_type, **extra)
        except Exception as exc:  # noqa: BLE001
            raise ServiceLoadError(self.name, f"could not load model: {exc}") from exc
        try:        # batched decoding (faster-whisper >= 1.1); without it batches run one by one
            from faster_whisper import BatchedInferencePipeline  # type: ignore
            self._batched = BatchedInferencePipeline(model=self._model)
        except Exception:  # noqa: BLE001
            log.warning("faster-whisper has no BatchedInferencePipeline: batches will run sequentially")
            self._batched = None
        self._loaded = True

    # ---- single clip -------------------------------------------------------------------
    def transcribe(self, audio: AudioData) -> Transcription:
        assert self._model is not None, "WhisperService not loaded"
        with self._lock:
            segments_iter, _info = self._model.transcribe(
                audio.samples, language=self._settings.whisper_language,
                beam_size=self._settings.whisper_beam_size,
                temperature=0.0, word_timestamps=True, condition_on_previous_text=False,
                vad_filter=False)
            segments = list(segments_iter)  # generator: must be consumed inside the lock

        seg_out, words_out = [], []
        for seg in segments:
            seg_out.append({"start": round(seg.start, 3), "end": round(seg.end, 3),
                            "text": seg.text.strip()})
            for w in seg.words or []:
                words_out.append({"word": w.word.strip(), "start": round(w.start, 3),
                                  "end": round(w.end, 3), "probability": round(w.probability, 4)})
        text = " ".join(s["text"] for s in seg_out).strip()
        return Transcription(text=text, segments=seg_out, word_timestamps=words_out)

    # ---- several clips as one GPU batch ----------------------------------------------------
    def transcribe_batch(self, audios: Sequence[AudioData]) -> list[Transcription]:
        """Decode all clips of ``audios`` in one batch (word timestamps included).

        Clips longer than ~30 s do not fit Whisper's window; they use the normal long-form decoder one
        at a time. Everything else is packed into one waveform and decoded together."""
        assert self._model is not None, "WhisperService not loaded"
        if self._batched is None or len(audios) < 2:
            return [self.transcribe(a) for a in audios]

        results: list[Transcription | None] = [None] * len(audios)
        short = [i for i, a in enumerate(audios) if a.duration_s <= MAX_CLIP_S]
        for i in range(len(audios)):
            if i not in short:
                results[i] = self.transcribe(audios[i])
        if not short:
            return results  # type: ignore[return-value]

        packed, spans = pack_clips([audios[i].samples for i in short])
        with self._lock:
            segments_iter, _info = self._batched.transcribe(
                packed, language=self._settings.whisper_language,
                beam_size=self._settings.whisper_beam_size, temperature=0.0,
                word_timestamps=True, vad_filter=False, clip_timestamps=spans,
                batch_size=max(len(spans), 1))
            segments = list(segments_iter)      # generator: consume inside the lock
        for i, transcription in zip(short, unpack_segments(segments, spans)):
            results[i] = transcription
        return results  # type: ignore[return-value]
