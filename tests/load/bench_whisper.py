"""Whisper benchmark on CPU: 10 clips x 10 s, one at a time vs batched.

    python tests/load/bench_whisper.py [--threads 4] [--model small.en]

Batched mode calls CTranslate2 directly with several clips at once (no word timestamps).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from _common import ten_second_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for CTranslate2 (0 = library default of 4)")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer

    clips = ten_second_clips(10)
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type, cpu_threads=args.threads)
    print(f"model={args.model} {args.compute_type} cpu_threads={args.threads or 'default(4)'}", flush=True)
    list(model.transcribe(clips[0], language="en", beam_size=5)[0])          # warm-up

    def sequential(word_ts: bool, beam: int):
        t = time.perf_counter()
        texts = []
        for c in clips:
            segs, _ = model.transcribe(c, language="en", beam_size=beam, temperature=0.0,
                                       word_timestamps=word_ts, condition_on_previous_text=False, vad_filter=False)
            texts.append(" ".join(s.text.strip() for s in segs))
        return time.perf_counter() - t, texts

    tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language="en")

    def batched(batch: int, beam: int):
        t = time.perf_counter()
        texts = []
        for i in range(0, len(clips), batch):
            feats = np.stack([pad_or_trim(model.feature_extractor(c)[..., :-1]) for c in clips[i:i + batch]])
            enc = model.encode(feats)
            prompt = model.get_prompt(tok, previous_tokens=[], without_timestamps=True)
            res = model.model.generate(enc, [prompt.copy() for _ in range(feats.shape[0])], beam_size=beam,
                                       max_length=448, suppress_blank=True, sampling_temperature=0.0)
            texts += [tok.decode([x for x in r.sequences_ids[0] if x < tok.eot]).strip() for r in res]
        return time.perf_counter() - t, texts

    print("\n10 clips x 10 s each\n")
    for label, fn in [("sequential, beam5, word timestamps (what the API does)", lambda: sequential(True, 5)),
                      ("sequential, beam5, no word timestamps", lambda: sequential(False, 5)),
                      ("sequential, beam1 (greedy), no word timestamps", lambda: sequential(False, 1)),
                      ("ONE batch of 10, beam5", lambda: batched(10, 5)),
                      ("ONE batch of 10, beam1 (greedy)", lambda: batched(10, 1)),
                      ("two batches of 5, beam5", lambda: batched(5, 5))]:
        secs, texts = fn()
        print(f"{label:58s} total {secs:6.1f} s   per clip {secs / 10:5.2f} s   nonempty {sum(bool(x) for x in texts)}/10",
              flush=True)


if __name__ == "__main__":
    main()
