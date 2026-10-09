import pytest

from app.core.config import get_settings
from app.scoring.config import load_scoring_config
from app.scoring.normalizers import delivery_scores
from app.services.acoustic.fluency import word_fluency_metrics

SCORING = load_scoring_config(get_settings().scoring_config_path)


# ---- fluency from word timing -----------------------------------------------------------------
def words_at(spec):
    """spec: list of (word, start, end)."""
    return [{"word": w, "start": s, "end": e, "probability": 0.9} for w, s, e in spec]


def test_fluent_speech_has_no_hesitation_metrics():
    ws = words_at([(f"w{i}", i * 0.4, i * 0.4 + 0.35) for i in range(10)])
    m = word_fluency_metrics(ws)
    assert m["mid_clause_pause_rate"] == 0 and m["hesitation_count"] == 0 and m["mean_run_length"] == 10


def test_pause_placement_distinguishes_hesitation_from_sentence_breaks():
    ws = words_at([("I", 0, .2), ("went", .25, .5), ("home.", .55, .9),       # boundary pause follows
                   ("Then", 1.6, 1.8), ("I", 1.85, 2.0), ("um", 2.9, 3.1), ("ate", 3.15, 3.4)])
    m = word_fluency_metrics(ws)
    assert m["boundary_pause_count"] == 1
    assert m["mid_clause_pause_count"] == 1 and m["long_pause_count"] == 0
    assert m["hesitation_count"] == 1                                            # the "um"


def test_repetitions_counted_and_short_input_skipped():
    ws = words_at([("I", 0, .2), ("I", .25, .4), ("want", .45, .7), ("tea", .75, 1.0)])
    assert word_fluency_metrics(ws)["hesitation_count"] == 1
    assert word_fluency_metrics(ws[:2]) == {}


def test_delivery_skips_unmeasured_components_instead_of_scoring_zero():
    base = {"speech_rate_wpm": 140, "pause_ratio": 0.05, "audio_quality_score": 1.0}
    only_basic = delivery_scores(base, SCORING)
    assert only_basic["fluency"] == pytest.approx(1.0)       # not dragged down by absent metrics
    hesitant = delivery_scores({**base, "mid_clause_pause_rate": 4.0, "mean_run_length": 2.0,
                                "hesitation_rate": 3.0, "long_pause_count": 3}, SCORING)
    assert hesitant["fluency"] < 0.55
    assert delivery_scores({"audio_quality_score": 1.0}, SCORING)["prosody"] == 0.0
