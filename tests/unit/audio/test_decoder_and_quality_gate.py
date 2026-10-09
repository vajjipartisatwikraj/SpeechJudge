import base64

import numpy as np
import pytest

from app.audio.decoder import decode_base64_audio, preprocess_audio
from app.audio.quality_gate import evaluate_gate
from app.core.errors import AudioDecodeError, AudioLimitError
from app.models.domain import AudioData
from app.services.acoustic.engine import NumpyAcousticEngine
from tests.support import SR, make_settings, make_speech, wav_bytes

S = make_settings()
engine = NumpyAcousticEngine()


def audio(x) -> AudioData:
    return AudioData(samples=x, original_duration_s=len(x) / SR)


def test_preprocess_wav_resamples_to_canonical():
    x = make_speech(2.0)
    # 8 kHz stereo-ish source: build at 8k by decimating
    raw = wav_bytes(x[::2], sr=8000)
    out = preprocess_audio(raw, S)
    assert out.sample_rate == 16000 and out.samples.ndim == 1
    assert abs(out.duration_s - 2.0) < 0.01


def test_rejects_non_audio_and_bad_base64():
    with pytest.raises(AudioDecodeError):
        preprocess_audio(b"not audio at all", S)
    with pytest.raises(AudioDecodeError):
        preprocess_audio(b"#EXTM3U\nhttp://evil/x", S)  # text playlists never reach FFmpeg
    with pytest.raises(AudioDecodeError):
        decode_base64_audio("", 1000)


def test_size_and_duration_limits():
    with pytest.raises(AudioLimitError):
        decode_base64_audio(base64.b64encode(b"x" * 5000).decode(), 1000)
    with pytest.raises(AudioLimitError):
        preprocess_audio(wav_bytes(make_speech(3.0)), make_settings(max_audio_seconds=2.0))


def test_gate_too_short():
    probe = engine.probe(audio(make_speech(0.5, pause_at=None)))
    d = evaluate_gate(probe, S)
    assert d.rejected and d.reason == "Audio too short"


def test_gate_no_speech():
    x = (0.0005 * np.random.default_rng(1).standard_normal(3 * SR)).astype(np.float32)
    d = evaluate_gate(engine.probe(audio(x)), S)
    assert d.rejected and d.reason == "Insufficient speech signal"


def test_gate_passes_clean_speech():
    d = evaluate_gate(engine.probe(audio(make_speech())), S)
    assert not d.rejected and d.flags == []


def test_gate_flags_noise_and_clipping_but_continues():
    noisy = make_speech(noise=0.04)  # ~6 dB SNR: speech still detectable, below the 10 dB threshold
    d = evaluate_gate(engine.probe(audio(noisy)), S)
    assert not d.rejected and "EXCESSIVE_NOISE" in d.flags
    clipped = np.clip(make_speech(amplitude=3.0), -1, 1).astype(np.float32)
    d = evaluate_gate(engine.probe(audio(clipped)), S)
    assert "CLIPPING" in d.flags


def test_gate_is_deterministic():
    a = audio(make_speech())
    assert evaluate_gate(engine.probe(a), S) == evaluate_gate(engine.probe(a), S)
