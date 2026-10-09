import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.core.config import get_settings
from app.core.errors import ScoreError, ScoringConfigError
from app.scoring.config import load_scoring_config, parse_scoring_config
from app.scoring.engine import ScoreEngine, compute_score

W = {"a": 0.5, "b": 0.3, "c": 0.2}
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def test_default_config_loads_and_matches_spec():
    cfg = load_scoring_config(get_settings().scoring_config_path)
    assert cfg.weights["READING"] == {"word_accuracy": 0.30, "pronunciation": 0.35, "fluency": 0.15,
                                      "prosody": 0.10, "audio_quality": 0.10}
    assert cfg.weights["QUESTION_ANSWER"]["relevance"] == 0.30
    for key in ("REPEAT", "JUMBLED_UI", "JUMBLED_SPOKEN", "STORYTELLING"):
        assert key in cfg.weights


def test_weighted_sum_and_components():
    out = compute_score({"a": 1.0, "b": 0.5, "c": 0.0}, W)
    assert out.score == pytest.approx(0.65)
    assert sum(c["weighted_value"] for c in out.components) == pytest.approx(out.score, abs=1e-4)


def test_missing_metric_named():
    with pytest.raises(ScoreError, match="b"):
        compute_score({"a": 1.0, "c": 1.0}, W)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), None])
def test_out_of_range_rejected(bad):
    with pytest.raises(ScoreError, match="a"):
        compute_score({"a": bad, "b": 0.5, "c": 0.5}, W)


@pytest.mark.parametrize("weights", [{"a": 0.5, "b": 0.6}, {"a": 1.2, "b": -0.2}])
def test_invalid_weights_refuse_start(weights):
    data = {"scoring": {k: weights for k in
                        ("READING", "REPEAT", "JUMBLED_UI", "JUMBLED_SPOKEN", "QUESTION_ANSWER",
                         "STORYTELLING")}}
    with pytest.raises(ScoringConfigError):
        parse_scoring_config(data)


@given(unit, unit, unit)
def test_deterministic_bounded(a, b, c):
    m = {"a": a, "b": b, "c": c}
    assert compute_score(m, W).score == compute_score(m, W).score
    assert 0.0 <= compute_score(m, W).score <= 1.0


@given(unit, unit, unit, unit)
def test_monotonic(a, b, c, bump):
    base = compute_score({"a": a, "b": b, "c": c}, W).score
    higher = compute_score({"a": max(a, bump), "b": b, "c": c}, W).score
    assert higher >= base - 1e-12


@given(unit)
def test_constant_metrics_give_constant_score(c):
    assert compute_score({"a": c, "b": c, "c": c}, W).score == pytest.approx(c, abs=1e-4)


def test_recompute_without_services():
    cfg = load_scoring_config(get_settings().scoring_config_path)
    engine = ScoreEngine(cfg)
    assert engine.recompute({"x": 1.0, "y": 0.0}, {"x": 0.25, "y": 0.75}).score == 0.25
