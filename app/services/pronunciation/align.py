"""Weighted phoneme alignment between expected and recognised phonemes, and scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.pronunciation.phonemes import AccentProfile, Unit

_EPS = 1e-9


@dataclass
class ExpectedWord:
    word: str
    units: list[Unit]


@dataclass
class PhonemeOp:
    word_index: int
    expected: str | None      # None for an insertion
    heard: str | None         # None for a deletion
    status: str               # match | accepted | near | substitution | deletion | insertion
    cost: float


@dataclass
class PronunciationScores:
    ops: list[PhonemeOp]
    words: list[dict]
    weighted_per: float
    raw_per: float
    pronunciation: float
    completeness: float
    expected_count: int
    heard_count: int = 0
    extra: dict = field(default_factory=dict)


def _ramp(value: float, zero_at: float, full_at: float) -> float:
    if zero_at == full_at:
        return 0.0
    return min(1.0, max(0.0, (value - zero_at) / (full_at - zero_at)))


def align_phonemes(words: list[ExpectedWord], heard: list[str], profile: AccentProfile
                   ) -> list[PhonemeOp]:
    exp: list[tuple[int, Unit]] = [(wi, u) for wi, w in enumerate(words) for u in w.units]
    n, m = len(exp), len(heard)
    # d[i][j] = cheapest alignment of exp[i:] with heard[j:]. Tracing forward from (0, 0) and
    # preferring the diagonal makes ties resolve in favour of the EARLIEST expected sound, so a
    # single heard schwa is credited to the first word that expects one (not a later word).
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        d[i][m] = d[i + 1][m] + profile.deletion_cost(exp[i][1])
    for j in range(m - 1, -1, -1):
        d[n][j] = d[n][j + 1] + profile.insertion_cost(heard[j])
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            d[i][j] = min(
                d[i + 1][j + 1] + profile.sub_cost(exp[i][1].symbol, heard[j]),
                d[i + 1][j] + profile.deletion_cost(exp[i][1]),
                d[i][j + 1] + profile.insertion_cost(heard[j]))

    ops: list[PhonemeOp] = []
    i = j = 0
    while i < n or j < m:
        if i < n and j < m:
            c = profile.sub_cost(exp[i][1].symbol, heard[j])
            if abs(d[i][j] - (d[i + 1][j + 1] + c)) < _EPS:
                wi, u = exp[i]
                if u.symbol == heard[j]:
                    status = "match"
                elif c == 0.0:
                    status = "accepted"
                elif c == profile.costs["near_substitution"]:
                    status = "near"
                else:
                    status = "substitution"
                ops.append(PhonemeOp(wi, u.symbol, heard[j], status, c))
                i, j = i + 1, j + 1
                continue
        if i < n:
            c = profile.deletion_cost(exp[i][1])
            if j == m or abs(d[i][j] - (d[i + 1][j] + c)) < _EPS:
                wi, u = exp[i]
                ops.append(PhonemeOp(wi, u.symbol, None, "deletion", c))
                i += 1
                continue
        wi = exp[i - 1][0] if i > 0 else 0   # an extra sound belongs to the preceding word
        ops.append(PhonemeOp(wi, None, heard[j], "insertion", profile.insertion_cost(heard[j])))
        j += 1
    return ops


def score_alignment(words: list[ExpectedWord], ops: list[PhonemeOp], heard_count: int,
                    profile: AccentProfile) -> PronunciationScores:
    n_exp = sum(len(w.units) for w in words)
    total_cost = sum(op.cost for op in ops)
    weighted_per = total_cost / n_exp if n_exp else 0.0
    raw_errors = sum(1 for op in ops if op.status in ("substitution", "near", "deletion", "insertion"))
    raw_per = raw_errors / n_exp if n_exp else 0.0

    word_results: list[dict] = []
    present = scored = 0
    present_cost = 0.0
    present_units = 0
    bad_words = 0.0
    for wi, w in enumerate(words):
        w_ops = [op for op in ops if op.word_index == wi]
        n = len(w.units)
        if n == 0:
            continue
        scored += 1
        cost = sum(op.cost for op in w_ops)
        deleted = sum(1 for op in w_ops if op.status == "deletion")
        coverage = 1.0 - deleted / n
        accuracy = min(1.0, max(0.0, 1.0 - cost / n))
        if coverage >= profile.word_present_min_coverage:
            present += 1
            present_cost += cost
            present_units += n
            bad_words += 1.0 - _ramp(accuracy, profile.word_bad_below, profile.word_ok_above)
        word_results.append({
            "word": w.word,
            "accuracy": round(accuracy, 4),
            "expected": " ".join(u.symbol for u in w.units),
            "heard": " ".join(op.heard for op in w_ops if op.heard),
            "errors": [{"expected": op.expected, "heard": op.heard, "type": op.status,
                        "cost": round(op.cost, 3)}
                       for op in w_ops if op.status in ("substitution", "near", "deletion", "insertion")],
        })

    # Pronunciation is judged on the words that were actually produced; words that are missing
    # are already penalised by `completeness`, so they are not counted twice.
    # The error blends the sound-level rate with the share of mispronounced words: listeners
    # notice a wrong word more than a wrong sound, and a single wrong sound in a 40-sound
    # sentence would otherwise disappear into recogniser noise.
    if present_units:
        sound_err = present_cost / present_units
        word_err = bad_words / present
        blended = (1 - profile.word_error_weight) * sound_err + profile.word_error_weight * word_err
        pronunciation = _ramp(blended, profile.per_zero_at, profile.per_full_at)
    else:
        sound_err = word_err = blended = 1.0
        pronunciation = 0.0
    completeness = (present / scored) if scored else 0.0
    pronunciation_present = pronunciation
    # Skipping words must not be rewarded with full pronunciation marks for the rest.
    pronunciation *= completeness

    return PronunciationScores(
        ops=ops, words=word_results, weighted_per=round(weighted_per, 4), raw_per=round(raw_per, 4),
        pronunciation=pronunciation,
        completeness=completeness,
        expected_count=n_exp, heard_count=heard_count,
        extra={"pronunciation_of_spoken_words": round(pronunciation_present, 4),
               "sound_error_present": round(sound_err, 4),
               "mispronounced_word_rate": round(word_err, 4), "blended_error": round(blended, 4)})
