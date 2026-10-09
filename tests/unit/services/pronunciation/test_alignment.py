import pytest

from app.core.config import get_settings
from app.services.pronunciation.align import ExpectedWord, align_phonemes, score_alignment
from app.services.pronunciation.phonemes import load_profile, split_token

PROFILE = load_profile(get_settings().accent_profile_path)


def word(w, *tokens):
    return ExpectedWord(w, [u for t in tokens for u in split_token(t, PROFILE)])


def run(words, heard_tokens):
    heard = [u.symbol for t in heard_tokens for u in split_token(t, PROFILE)]
    ops = align_phonemes(words, heard, PROFILE)
    return score_alignment(words, ops, len(heard), PROFILE)


# ---- token normalisation ------------------------------------------------------------------
def test_split_token_rules():
    sym = lambda t: [(u.symbol, u.role) for u in split_token(t, PROFILE)]  # noqa: E731
    assert sym("tʃ") == [("t", None), ("ʃ", None)]
    assert sym("eɪ") == [("e", None), ("ɪ", "glide")]
    assert sym("ɔːɹ") == [("ɔ", None), ("ɹ", "rcolor")]
    assert sym("ɚ") == [("ə", None), ("ɹ", "rcolor")]
    assert sym("ɑː") == [("ɑ", None)]            # length ignored
    assert sym("tʰ") == [("t", None)] and sym("th") == [("t", None)]
    assert sym("ɹ") == [("ɹ", None)]             # consonant r stays a consonant


# ---- alignment + accent tolerance ----------------------------------------------------------
def test_perfect_match_scores_full():
    w = [word("think", "θ", "ɪ", "ŋ", "k")]
    r = run(w, ["θ", "ɪ", "ŋ", "k"])
    assert r.weighted_per == 0 and r.pronunciation == 1.0 and r.completeness == 1.0


def test_indian_english_features_are_tolerated():
    # th->t, v->w, retroflex t, flapped t, no vowel reduction
    assert run([word("think", "θ", "ɪ", "ŋ", "k")], ["t", "ɪ", "ŋ", "k"]).weighted_per == 0
    assert run([word("this", "ð", "ɪ", "s")], ["d", "ɪ", "s"]).weighted_per == 0
    assert run([word("very", "v", "ɛ", "ɹ", "i")], ["w", "ɛ", "r", "i"]).weighted_per == 0
    assert run([word("tea", "t", "iː")], ["ʈ", "iː"]).weighted_per == 0
    assert run([word("water", "w", "ɔː", "ɾ", "ɚ")], ["w", "ɔ", "t", "ɐ", "ɹ"]).weighted_per == 0
    assert run([word("about", "ə", "b", "aʊ", "t")], ["a", "b", "aʊ", "t"]).weighted_per == 0


def test_real_errors_are_penalised():
    # a 6-word sentence of 3-sound words; one word gets one wrong / dropped / extra sound
    ws = [word(f"w{i}", "k", "æ", "t") for i in range(6)]
    good = ["k", "æ", "t"] * 6
    base = run(ws, good)
    sub = run(ws, ["k", "æ", "m"] + good[3:])                 # wrong consonant in word 0
    dele = run(ws, ["k", "æ"] + good[3:])                     # dropped consonant
    ins = run(ws, good[:3] + ["s"] + good[3:])                # extra sound after word 0
    assert base.pronunciation == 1.0
    for r in (sub, dele, ins):
        assert 0.0 < r.pronunciation < base.pronunciation     # noticed, but not catastrophic
    assert sub.words[0]["errors"][0]["type"] == "substitution"
    assert dele.words[0]["errors"][0]["type"] == "deletion"
    assert ins.words[0]["errors"][0]["type"] == "insertion"
    assert 0.1 < sub.extra["mispronounced_word_rate"] < 0.17   # one of six words clearly off


def test_diphthong_simplification_is_cheap_not_free():
    r = run([word("day", "d", "eɪ")], ["d", "e"])
    assert 0 < r.weighted_per < 0.2


def test_missing_word_lowers_completeness():
    words = [word("the", "ð", "ə"), word("doctor", "d", "ɑː", "k", "t", "ɚ")]
    r = run(words, ["ð", "ə"])        # second word not said at all
    assert r.completeness == 0.5
    # The lone heard schwa may be credited to "the" or (as a reduced vowel) to "doctor", which
    # costs the same order of magnitude; what matters is that "doctor" is clearly not produced.
    assert r.words[0]["accuracy"] >= 0.75
    assert r.words[1]["accuracy"] < 0.4


def test_pronunciation_decreases_with_more_errors_and_is_deterministic():
    w = [word("sentence", "s", "ɛ", "n", "t", "ə", "n", "s")]
    scores = [run(w, h).pronunciation for h in (
        ["s", "ɛ", "n", "t", "ə", "n", "s"], ["s", "ɛ", "n", "t", "ə", "n", "k"],
        ["m", "ɛ", "n", "p", "ə", "n", "k"], ["k", "k", "k", "k"])]
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0 and scores[-1] == 0.0
    assert run(w, ["s", "ɛ", "n"]).weighted_per == run(w, ["s", "ɛ", "n"]).weighted_per
