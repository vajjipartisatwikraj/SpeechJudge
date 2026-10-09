"""Deterministic Expected-vs-Hypothesis comparison (Requirement 12).

WER is computed from a minimum-edit-distance word alignment. The reported
missing / extra / substituted lists describe *content* differences: words that
were merely moved (appear in both the unmatched reference and unmatched
hypothesis words) are cancelled out, so a pure reordering reports nothing
missing or extra and shows up in ``order_correctness`` instead.
"""
from __future__ import annotations

import re
from collections import Counter

from app.core.logging import get_logger
from app.models.domain import TextComparisonResult

log = get_logger(__name__)

_APOSTROPHES = re.compile(r"['\u2019\u2018`]")
_NON_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalize_words(text: str | None) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace, split into words."""
    if not text:
        return []
    text = _APOSTROPHES.sub("", text.lower())
    text = _NON_WORD.sub(" ", text)
    return text.split()


def _align(ref: list[str], hyp: list[str]):
    """Return (substitutions, deletions, insertions) as index lists.

    Tie-break order on backtrace is match/substitute, then delete, then insert,
    which makes the alignment deterministic.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    subs: list[tuple[int, int]] = []
    dels: list[int] = []
    ins: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                if cost:
                    subs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels.append(i - 1)
            i -= 1
        else:
            ins.append(j - 1)
            j -= 1
    subs.reverse()
    dels.reverse()
    ins.reverse()
    return subs, dels, ins


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def compare_text(expected_text: str | None, hypothesis_text: str | None) -> TextComparisonResult:
    ref = normalize_words(expected_text)
    hyp = normalize_words(hypothesis_text)

    if not ref:
        log.error("Text comparison received an empty reference (expected_text has zero words)")
        return TextComparisonResult(
            word_accuracy=0.0, wer=0.0, missing_words=[], extra_words=[], substituted_words=[],
            order_correctness=0.0, expected_word_count=0, hypothesis_word_count=len(hyp))

    subs, dels, ins = _align(ref, hyp)
    wer = (len(subs) + len(dels) + len(ins)) / len(ref)

    # Unmatched words in positional order, from both sides.
    unmatched_ref = sorted([i for i, _ in subs] + dels)
    unmatched_hyp = sorted([j for _, j in subs] + ins)
    ref_bag = Counter(ref[i] for i in unmatched_ref)
    hyp_bag = Counter(hyp[j] for j in unmatched_hyp)
    moved = ref_bag & hyp_bag  # words present on both sides: reordered, not missing/extra

    def _kept_indices(indices: list[int], words: list[str], cancel: Counter) -> set[int]:
        """Indices NOT cancelled by the moved-word budget (earliest occurrences cancel first)."""
        budget = Counter(cancel)
        kept: set[int] = set()
        for idx in indices:
            w = words[idx]
            if budget[w] > 0:
                budget[w] -= 1
            else:
                kept.add(idx)
        return kept

    kept_ref = _kept_indices(unmatched_ref, ref, moved)
    kept_hyp = _kept_indices(unmatched_hyp, hyp, moved)

    substituted: list[dict[str, str]] = []
    paired_ref: set[int] = set()
    paired_hyp: set[int] = set()
    for i, j in subs:
        if i in kept_ref and j in kept_hyp:
            substituted.append({"expected": ref[i], "spoken": hyp[j]})
            paired_ref.add(i)
            paired_hyp.add(j)
    # Everything else that survived cancellation is a genuinely missing / extra word.
    missing = [ref[i] for i in unmatched_ref if i in kept_ref and i not in paired_ref]
    extra = [hyp[j] for j in unmatched_hyp if j in kept_hyp and j not in paired_hyp]

    common = sum((Counter(ref) & Counter(hyp)).values())
    if common == 0:
        order = 0.0
    else:
        order = min(1.0, _lcs_length(ref, hyp) / common)

    return TextComparisonResult(
        word_accuracy=max(0.0, 1.0 - wer),
        wer=wer,
        missing_words=missing,
        extra_words=extra,
        substituted_words=substituted,
        order_correctness=order,
        expected_word_count=len(ref),
        hypothesis_word_count=len(hyp),
        substitutions=len(subs),
        deletions=len(dels),
        insertions=len(ins),
    )
