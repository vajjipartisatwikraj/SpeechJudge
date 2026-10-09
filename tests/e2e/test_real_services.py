"""Each real shared service on real audio: Whisper, acoustic engine, Qwen (via Ollama)."""
from __future__ import annotations

import pytest

from app.models.domain import QuestionType
from app.services.acoustic.engine import NumpyAcousticEngine
from app.services.text_comparison.comparison import compare_text

TRUTH = {
    "reading": "The doctor suggested that the apple is good for your health.",
    "qa_good": "On weekends I usually wake up late. Then I go to the park with my friends and we play "
               "football. In the evening I watch a movie with my family.",
    "qa_offtopic": "Bananas are yellow. Trains are very fast and I like the color blue.",
    "story": "Last summer my family went to a small village near the mountains. On the first day we "
             "walked to a river and saw a little boy who had lost his dog. We helped him search the "
             "forest, and after two hours we found the dog sleeping under a tree. The boy was very "
             "happy, and his mother invited us for dinner. It was the best holiday I have ever had.",
}


@pytest.fixture(scope="module")
def whisper(real_settings):
    from app.services.stt.whisper_service import WhisperService
    svc = WhisperService(real_settings)
    svc.load()
    return svc


@pytest.mark.parametrize("clip", list(TRUTH))
def test_whisper_transcribes_clean_speech(whisper, load_clip, clip):
    audio = load_clip(clip)
    result = whisper.transcribe(audio)
    assert compare_text(TRUTH[clip], result.text).wer <= 0.05
    assert len(result.word_timestamps) >= len(TRUTH[clip].split()) - 2


@pytest.mark.parametrize("clip", list(TRUTH))
def test_acoustic_engine_on_real_audio(load_clip, clip):
    engine = NumpyAcousticEngine()
    engine.load()
    audio = load_clip(clip)
    probe = engine.probe(audio)
    assert probe.speech_duration_s > 0.5
    metrics = engine.analyze(audio, probe)
    assert metrics["pitch_std_st"] is not None and metrics["audio_quality_score"] > 0.5


def test_language_evaluator_separates_on_and_off_topic(real_settings):
    if real_settings.qwen_backend != "ollama":
        pytest.skip("this test targets the Ollama backend")
    from app.services.language.ollama_service import OllamaLanguageEvaluator
    svc = OllamaLanguageEvaluator(real_settings)
    try:
        svc.load()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not available: {exc}")
    q = "What do you usually do on weekends?"
    good = svc.evaluate(QuestionType.QUESTION_ANSWER, q, TRUTH["qa_good"]).scores()
    bad = svc.evaluate(QuestionType.QUESTION_ANSWER, q, TRUTH["qa_offtopic"]).scores()
    assert good["relevance"] >= 0.7 and bad["relevance"] <= 0.3
