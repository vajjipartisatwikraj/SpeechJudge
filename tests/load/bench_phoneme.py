"""Phoneme (pronunciation) model benchmark on CPU: 10 clips x 10 s, batch size 1 vs 2 / 5 / 10.

    python tests/load/bench_phoneme.py [--threads 4]

Downloads the ~1.2 GB wav2vec2 model on first run. Also checks that batched output equals the
one-at-a-time output.
"""
from __future__ import annotations

import argparse
import statistics
import time

from _common import SR, ten_second_clips

TEXT = "Last summer my family went to a small village near the mountains."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch

    from app.core.config import Settings
    from app.models.domain import AudioData
    from app.services.pronunciation.phoneme_service import PhonemePronunciationService

    torch.set_num_threads(args.threads)
    clips = ten_second_clips(10)
    svc = PhonemePronunciationService(Settings(_env_file=None, pronunciation_backend="phoneme", device="cpu"))
    svc.load()
    proc, model, id2tok = svc._processor, svc._model, svc._id_to_token
    print(f"torch threads = {torch.get_num_threads()}", flush=True)

    def decode(ids):
        out, prev = [], None
        for i in ids:
            if i != prev:
                tok = id2tok.get(i, "")
                if tok and not tok.startswith("<") and tok != "|":
                    out.append(tok)
            prev = i
        return out

    def forward(batch_clips):
        inputs = proc(batch_clips, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(inputs.input_values).logits
        return [decode(row) for row in torch.argmax(logits, dim=-1).tolist()]

    def run(batch, reps=2):
        times, outs = [], []
        for _ in range(reps):
            t = time.perf_counter()
            outs = []
            for i in range(0, 10, batch):
                outs += forward(clips[i:i + batch])
            times.append(time.perf_counter() - t)
        return statistics.mean(times), outs

    forward(clips[:1])                                             # warm-up
    print("\n10 clips x 10 s each (model forward + decode, mean of 2 runs)\n")
    base_total, base_out = run(1)
    print(f"{'one at a time (batch 1)':32s} total {base_total:6.1f} s   per clip {base_total / 10:5.2f} s")
    for b in (2, 5, 10):
        total, outs = run(b)
        same = sum(o == r for o, r in zip(outs, base_out))
        print(f"{'batch of ' + str(b):32s} total {total:6.1f} s   per clip {total / 10:5.2f} s   "
              f"speed-up {base_total / total:4.2f}x   identical output {same}/10")
    t = time.perf_counter()
    for c in clips:
        svc.assess(AudioData(c, 10.0), TEXT)
    e2e = time.perf_counter() - t
    print(f"\nfull assess() per clip (phonemize + model + alignment): {e2e / 10:5.2f} s "
          f"(model forward is about {base_total / e2e * 100:.0f}% of it)")


if __name__ == "__main__":
    main()
