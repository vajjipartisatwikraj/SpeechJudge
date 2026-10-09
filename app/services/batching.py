"""Shared stage queues with micro-batching: ONE persistent worker per model, shared by every student.

    question threads ──► [Whisper queue ] ──► Whisper worker   (batches up to N clips on the GPU)
                    ├──► [Phoneme queue ] ──► Phoneme worker   (batches up to N clips on the GPU)
                    ├──► [Acoustic queue] ──► Acoustic worker
                    └──► [Qwen queue    ] ──► Qwen worker

The evaluators are unchanged: they call ``services.stt.transcribe(audio)`` etc. The ``Batched*``
services below look exactly like the real ones, but put the call on the stage queue and wait for the
result. While a worker is busy, new requests wait in the queue; the next time the worker is free it
takes up to ``batch_size`` of them and runs them as one batch. So 10 questions arriving together are one
GPU batch of 10, and requests of different students share batches too.

The queues live inside the worker process (the models and the audio arrays are there, so nothing is
copied through Redis). Redis carries the *section jobs* from the API to the worker.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from concurrent.futures import Future
from typing import Any, Callable, Sequence

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.domain import AudioData, GoptResult, LanguageResult, QualityProbe, QuestionType, Transcription
from app.services.interfaces import AcousticEngine, GoptService, LanguageEvaluator, STTService
from app.services.registry import ServiceRegistry

log = get_logger(__name__)


class StageWorker:
    """Queue + worker thread(s) that run ``handler(list_of_items) -> list_of_results`` in batches."""

    def __init__(self, name: str, handler: Callable[[list[Any]], list[Any]], *, batch_size: int = 1,
                 max_wait_s: float = 0.0, threads: int = 1) -> None:
        self.name = name
        self._handler = handler
        self.batch_size = max(1, batch_size)
        self.max_wait_s = max(0.0, max_wait_s)
        self._thread_count = max(1, threads)
        self._queue: queue.Queue[tuple[Any, Future]] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._start_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.batch_sizes: Counter[int] = Counter()          # batch size -> how many batches had it

    # ---- public --------------------------------------------------------------------------
    def submit(self, item: Any) -> Future:
        self._ensure_started()
        fut: Future = Future()
        self._queue.put((item, fut))
        return fut

    def shutdown(self) -> None:
        self._stop.set()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            sizes = dict(sorted(self.batch_sizes.items()))
        return {"batches": sum(sizes.values()), "items": sum(k * v for k, v in sizes.items()),
                "max_batch": max(sizes) if sizes else 0, "batch_sizes": sizes}

    # ---- worker --------------------------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._threads:
            return
        with self._start_lock:
            if self._threads:
                return
            for i in range(self._thread_count):
                t = threading.Thread(target=self._loop, name=f"sj-stage-{self.name}-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def _collect(self) -> list[tuple[Any, Future]] | None:
        try:
            batch = [self._queue.get(timeout=0.2)]
        except queue.Empty:
            return None
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            try:
                batch.append(self._queue.get(timeout=remaining) if remaining > 0
                             else self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if batch:
                self._run(batch)

    def _run(self, batch: list[tuple[Any, Future]]) -> None:
        items = [item for item, _ in batch]
        t0 = time.perf_counter()
        try:
            outputs = self._handler(items)
            if len(outputs) != len(items):
                raise RuntimeError(f"{self.name}: handler returned {len(outputs)} results for {len(items)} items")
        except Exception as exc:  # noqa: BLE001
            if len(batch) == 1:
                batch[0][1].set_exception(exc)
                return
            log.warning("Stage '%s': batch of %d failed (%s); retrying the items one by one",
                        self.name, len(batch), exc)
            for item, fut in batch:           # one bad item must not fail its neighbours
                try:
                    fut.set_result(self._handler([item])[0])
                except Exception as item_exc:  # noqa: BLE001
                    fut.set_exception(item_exc)
            return
        with self._stats_lock:
            self.batch_sizes[len(batch)] += 1
        log.info("Stage '%s': batch of %d done in %.2f s", self.name, len(batch), time.perf_counter() - t0)
        for (_, fut), out in zip(batch, outputs):
            fut.set_result(out)


# ---- services that route their calls through a stage queue ---------------------------------
class BatchedSTT(STTService):
    def __init__(self, inner: STTService, worker: StageWorker) -> None:
        super().__init__()
        self._inner, self._worker = inner, worker

    @property
    def mocked(self) -> bool:  # type: ignore[override]
        return self._inner.mocked

    @property
    def is_loaded(self) -> bool:
        return self._inner.is_loaded

    def load(self) -> None:
        self._inner.load()

    def transcribe(self, audio: AudioData) -> Transcription:
        return self._worker.submit(audio).result()

    def transcribe_batch(self, audios: Sequence[AudioData]) -> list[Transcription]:
        return self._inner.transcribe_batch(audios)


class BatchedGopt(GoptService):
    def __init__(self, inner: GoptService, worker: StageWorker) -> None:
        super().__init__()
        self._inner, self._worker = inner, worker

    @property
    def mocked(self) -> bool:  # type: ignore[override]
        return self._inner.mocked

    @property
    def is_loaded(self) -> bool:
        return self._inner.is_loaded

    def load(self) -> None:
        self._inner.load()

    def assess(self, audio: AudioData, expected_text: str) -> GoptResult:
        return self._worker.submit((audio, expected_text)).result()

    def assess_batch(self, items: Sequence[tuple[AudioData, str]]) -> list[GoptResult]:
        return self._inner.assess_batch(items)


class BatchedAcoustic(AcousticEngine):
    """``probe`` and ``refine_rates`` are cheap and stay direct; the full ``analyze`` is queued."""

    def __init__(self, inner: AcousticEngine, worker: StageWorker) -> None:
        super().__init__()
        self._inner, self._worker = inner, worker

    @property
    def mocked(self) -> bool:  # type: ignore[override]
        return self._inner.mocked

    @property
    def is_loaded(self) -> bool:
        return self._inner.is_loaded

    def load(self) -> None:
        self._inner.load()

    def probe(self, audio: AudioData) -> QualityProbe:
        return self._inner.probe(audio)

    def analyze(self, audio: AudioData, probe: QualityProbe | None = None) -> dict[str, Any]:
        return self._worker.submit((audio, probe)).result()

    def refine_rates(self, metrics: dict[str, Any], word_count: int) -> dict[str, Any]:
        return self._inner.refine_rates(metrics, word_count)


class BatchedLanguage(LanguageEvaluator):
    def __init__(self, inner: LanguageEvaluator, worker: StageWorker) -> None:
        super().__init__()
        self._inner, self._worker = inner, worker

    @property
    def mocked(self) -> bool:  # type: ignore[override]
        return self._inner.mocked

    @property
    def is_loaded(self) -> bool:
        return self._inner.is_loaded

    def load(self) -> None:
        self._inner.load()

    def evaluate(self, question_type: QuestionType, question: str, transcript: str) -> LanguageResult:
        return self._worker.submit((question_type, question, transcript)).result()


class SharedStages:
    """The four stage workers plus a ServiceRegistry whose services use them."""

    def __init__(self, services: ServiceRegistry, settings: Settings) -> None:
        wait = settings.batch_max_wait_ms / 1000.0
        inner = services
        self.whisper = StageWorker("whisper", lambda items: list(inner.stt.transcribe_batch(items)),
                                   batch_size=settings.whisper_batch_size, max_wait_s=wait)
        self.phoneme = StageWorker("phoneme", lambda items: list(inner.gopt.assess_batch(items)),
                                   batch_size=settings.phoneme_batch_size, max_wait_s=wait)
        self.acoustic = StageWorker("acoustic",
                                    lambda items: [inner.acoustic.analyze(a, p) for a, p in items])
        self.qwen = StageWorker("qwen",
                                lambda items: [inner.language.evaluate(*item) for item in items],
                                threads=settings.qwen_stage_workers)
        self.registry = ServiceRegistry(
            stt=BatchedSTT(services.stt, self.whisper), gopt=BatchedGopt(services.gopt, self.phoneme),
            acoustic=BatchedAcoustic(services.acoustic, self.acoustic),
            language=BatchedLanguage(services.language, self.qwen))

    def workers(self) -> list[StageWorker]:
        return [self.whisper, self.phoneme, self.acoustic, self.qwen]

    def stats(self) -> dict[str, dict[str, Any]]:
        return {w.name: w.stats() for w in self.workers()}

    def shutdown(self) -> None:
        for w in self.workers():
            w.shutdown()
