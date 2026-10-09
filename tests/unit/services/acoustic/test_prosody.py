import numpy as np
import pytest

from app.core.config import get_settings
from app.models.domain import AudioData
from app.scoring.config import load_scoring_config
from app.scoring.normalizers import delivery_scores
from app.services.acoustic.engine import NumpyAcousticEngine
from tests.support import SR, make_speech

SCORING = load_scoring_config(get_settings().scoring_config_path)


# ---- prosody from pitch (semitones, speaker-normalised) --------------------------------------
def tone(f_start, f_end, seconds=2.5):
    n = int(seconds * SR)
    freq = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 5)) * 0.15
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(n) / SR)
    return (x * env).astype(np.float32)


def analyze(x):
    return NumpyAcousticEngine().analyze(AudioData(x, len(x) / SR))


def test_pitch_metrics_in_semitones():
    flat = analyze(tone(150, 150))
    moving = analyze(tone(110, 220))
    assert flat["pitch_std_st"] < 0.8 and flat["pitch_range_st"] < 2.5
    assert moving["pitch_range_st"] > 6 and moving["pitch_std_st"] > flat["pitch_std_st"] * 2
    # speaker-normalised: the same contour shape one octave higher gives the same measures
    higher = analyze(tone(220, 440))
    assert higher["pitch_range_st"] == pytest.approx(moving["pitch_range_st"], abs=2.0)


def test_terminal_pitch_direction_and_intonation_score():
    falling, rising = analyze(tone(220, 120)), analyze(tone(120, 220))
    assert falling["terminal_pitch_delta_st"] < -1 < 1 < rising["terminal_pitch_delta_st"]
    stat = lambda a: delivery_scores(a, SCORING, ".")  # noqa: E731
    ques = lambda a: delivery_scores(a, SCORING, "?")  # noqa: E731
    # a falling contour suits a statement better than a rising one, and vice versa for a question
    assert stat(falling)["prosody"] > stat(rising)["prosody"]
    assert ques(rising)["prosody"] > ques(falling)["prosody"]


def test_speech_like_fixture_still_analyses():
    m = analyze(make_speech())
    assert m["pitch_std_st"] is not None and m["syllable_energy_std_db"] is not None
