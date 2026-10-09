import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.batching import StageWorker
from app.services.stt.batching import pack_clips, unpack_segments


def collector():
    sizes = []

    def handler(items):
        sizes.append(len(items))
        return [i * 10 for i in items]
    return sizes, handler


def test_items_submitted_together_form_one_batch():
    sizes, handler = collector()
    worker = StageWorker("t", handler, batch_size=10, max_wait_s=1.0)
    futures = [worker.submit(i) for i in range(10)]
    assert [f.result(timeout=5) for f in futures] == [i * 10 for i in range(10)]
    assert sizes == [10] and worker.stats()["max_batch"] == 10
    worker.shutdown()


def test_batch_never_exceeds_batch_size():
    sizes, handler = collector()
    worker = StageWorker("t", handler, batch_size=4, max_wait_s=0.5)
    futures = [worker.submit(i) for i in range(10)]
    assert [f.result(timeout=5) for f in futures] == [i * 10 for i in range(10)]
    assert max(sizes) == 4 and sum(sizes) == 10
    worker.shutdown()


def test_partial_batch_is_dispatched_after_the_wait():
    sizes, handler = collector()
    worker = StageWorker("t", handler, batch_size=10, max_wait_s=0.05)
    t0 = time.monotonic()
    assert worker.submit(7).result(timeout=5) == 70
    assert sizes == [1] and time.monotonic() - t0 < 2
    worker.shutdown()


def test_batch_size_one_runs_items_individually():
    sizes, handler = collector()
    worker = StageWorker("t", handler, batch_size=1)
    for f in [worker.submit(i) for i in range(3)]:
        f.result(timeout=5)
    assert sizes == [1, 1, 1]
    worker.shutdown()


def test_one_bad_item_does_not_fail_the_batch():
    calls = []

    def handler(items):
        calls.append(list(items))
        if "bad" in items:
            raise ValueError("boom")
        return [str(i).upper() for i in items]
    worker = StageWorker("t", handler, batch_size=3, max_wait_s=1.0)
    good1, bad, good2 = worker.submit("a"), worker.submit("bad"), worker.submit("b")
    assert good1.result(timeout=5) == "A" and good2.result(timeout=5) == "B"
    with pytest.raises(ValueError):
        bad.result(timeout=5)
    assert calls[0] == ["a", "bad", "b"]          # tried as a batch first, then one by one
    worker.shutdown()


def test_wrong_result_count_is_an_error_for_every_item():
    worker = StageWorker("t", lambda items: [], batch_size=1)
    with pytest.raises(RuntimeError):
        worker.submit("x").result(timeout=5)
    worker.shutdown()


def test_many_threads_share_the_single_worker():
    sizes, handler = collector()
    worker = StageWorker("t", handler, batch_size=5, max_wait_s=0.2)
    out = {}

    def student(n):
        out[n] = worker.submit(n).result(timeout=10)
    threads = [threading.Thread(target=student, args=(n,)) for n in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert out == {n: n * 10 for n in range(12)} and sum(sizes) == 12 and max(sizes) <= 5
    worker.shutdown()


# ---- packing clips for one Whisper batch --------------------------------------------------------
def seg(start, end, text, words=()):
    return SimpleNamespace(start=start, end=end, text=f" {text} ",
                           words=[SimpleNamespace(word=f" {w} ", start=s, end=e, probability=0.9)
                                  for w, s, e in words])


def test_pack_clips_places_clips_after_each_other_with_silence():
    a, b = np.ones(16000, dtype=np.float32), np.full(8000, 0.5, dtype=np.float32)
    packed, spans = pack_clips([a, b], sample_rate=16000, gap_s=1.0)
    assert spans == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 2.5}]
    assert len(packed) == 16000 + 16000 + 8000 + 16000
    assert packed[:16000].min() == 1.0 and packed[16000:32000].max() == 0.0
    assert packed[32000:40000].max() == 0.5


def test_unpack_segments_maps_text_and_word_times_back_to_each_clip():
    spans = [{"start": 0.0, "end": 3.0}, {"start": 4.0, "end": 7.0}, {"start": 8.0, "end": 9.0}]
    segments = [seg(0.2, 2.5, "hello world", [("hello", 0.2, 0.9), ("world", 1.0, 2.5)]),
                seg(4.5, 6.0, "second clip", [("second", 4.5, 5.0), ("clip", 5.2, 6.0)])]
    out = unpack_segments(segments, spans)
    assert [t.text for t in out] == ["hello world", "second clip", ""]
    assert out[1].word_timestamps[0] == {"word": "second", "start": 0.5, "end": 1.0, "probability": 0.9}
    assert out[1].segments == [{"start": 0.5, "end": 2.0, "text": "second clip"}]
    assert out[2].word_timestamps == [] and out[2].segments == []
