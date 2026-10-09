from app.models.domain import AudioData
from app.services.acoustic.engine import NumpyAcousticEngine
from tests.support import SR, make_speech

engine = NumpyAcousticEngine()


def audio(x) -> AudioData:
    return AudioData(samples=x, original_duration_s=len(x) / SR)


def test_acoustic_metrics_complete_and_sane():
    m = engine.analyze(audio(make_speech(4.0, pause_at=2.0, pause_len=0.6)))
    for key in ("speech_duration_s", "silence_duration_s", "pause_count", "pause_duration_s",
                "speech_rate_wpm", "articulation_rate_wpm", "pitch_mean_hz", "pitch_min_hz",
                "pitch_max_hz", "pitch_range_hz", "energy_mean_db", "loudness_lufs",
                "leading_silence_s", "trailing_silence_s", "snr_db", "clipping_ratio",
                "audio_quality_score", "original_duration_s", "duration_s"):
        assert key in m, key
    assert m["pause_count"] == 1
    assert 0.4 < m["pause_duration_s"] < 0.8
    assert 130 < m["pitch_mean_hz"] < 170
    assert 0.0 <= m["audio_quality_score"] <= 1.0
    assert m["rate_source"] == "estimated"


def test_acoustic_deterministic_and_refine():
    a = audio(make_speech())
    m1, m2 = engine.analyze(a), engine.analyze(a)
    assert m1 == m2
    refined = engine.refine_rates(m1, 10)
    assert refined["rate_source"] == "transcript" and refined["speech_rate_wpm"] > 0
