from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.text_comparison.comparison import compare_text, normalize_words

words = st.lists(st.sampled_from("the a doctor apple is good health for your cat dog runs".split()),
                 min_size=1, max_size=12)


def test_normalization():
    assert normalize_words("  Hello,   WORLD! Don't-stop. ") == ["hello", "world", "dont", "stop"]


def test_identical():
    r = compare_text("The weather is beautiful today.", "the weather is beautiful today")
    assert r.wer == 0.0 and r.word_accuracy == 1.0 and r.order_correctness == 1.0
    assert r.missing_words == [] and r.extra_words == [] and r.substituted_words == []


def test_mixed_errors():
    r = compare_text("the cat sat on the mat", "the dog sat the mat now")
    assert r.substituted_words == [{"expected": "cat", "spoken": "dog"}]
    assert r.missing_words == ["on"] and r.extra_words == ["now"]
    assert abs(r.wer - 3 / 6) < 1e-9


def test_empty_reference_is_degenerate():
    r = compare_text("...", "anything")
    assert r.wer == 0.0 and r.word_accuracy == 0.0


def test_empty_hypothesis():
    r = compare_text("a b c", "")
    assert r.wer == 1.0 and r.word_accuracy == 0.0 and r.missing_words == ["a", "b", "c"]


def test_reordered_words_not_missing():
    r = compare_text("one two three", "three one two")
    assert r.missing_words == [] and r.extra_words == []
    assert 0.0 <= r.order_correctness < 1.0


@given(words)
def test_self_comparison(ws):
    r = compare_text(" ".join(ws), " ".join(ws))
    assert r.wer == 0.0 and r.word_accuracy == 1.0 and r.order_correctness == 1.0
    assert not (r.missing_words or r.extra_words or r.substituted_words)


@given(words, words)
@settings(max_examples=200)
def test_count_bounds_and_determinism(ref, hyp):
    e, h = " ".join(ref), " ".join(hyp)
    r = compare_text(e, h)
    assert len(r.substituted_words) + len(r.missing_words) <= len(ref)
    assert len(r.substituted_words) + len(r.extra_words) <= len(hyp)
    assert 0.0 <= r.word_accuracy <= 1.0 and 0.0 <= r.order_correctness <= 1.0
    assert r == compare_text(e, h)


@given(words, st.randoms(use_true_random=False))
def test_permutation(ref, rnd):
    shuffled = list(ref)
    rnd.shuffle(shuffled)
    r = compare_text(" ".join(ref), " ".join(shuffled))
    assert r.missing_words == [] and r.extra_words == []
    assert 0.0 <= r.order_correctness <= 1.0
