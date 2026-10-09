"""SJ_PRONUNCIATION_ENABLED / SJ_QWEN_ENABLED decide real vs mocked services."""
import pytest

from app.services.acoustic.engine import NumpyAcousticEngine
from app.services.language.ollama_service import OllamaLanguageEvaluator
from app.services.language.qwen_service import QwenLanguageEvaluator
from app.services.mocks import MockAcousticEngine, MockGopt, MockLanguageEvaluator, MockSTT
from app.services.pronunciation.phoneme_service import PhonemePronunciationService
from app.services.registry import build_registry
from app.services.stt.whisper_service import WhisperService
from tests.support import make_settings


def real_settings(**kw):
    """Non-mock settings; building the registry does not load any model."""
    return make_settings(mock_mode=False, qwen_backend="ollama", pronunciation_backend="phoneme", **kw)


def test_both_enabled_builds_real_services():
    reg = build_registry(real_settings())
    assert isinstance(reg.stt, WhisperService) and isinstance(reg.acoustic, NumpyAcousticEngine)
    assert isinstance(reg.gopt, PhonemePronunciationService)
    assert isinstance(reg.language, OllamaLanguageEvaluator)
    assert reg.mocked_names() == []


def test_pronunciation_disabled_mocks_only_pronunciation():
    reg = build_registry(real_settings(pronunciation_enabled=False))
    assert isinstance(reg.gopt, MockGopt) and isinstance(reg.language, OllamaLanguageEvaluator)
    assert reg.mocked_names() == ["gopt"]


def test_qwen_disabled_mocks_only_language():
    reg = build_registry(real_settings(qwen_enabled=False))
    assert isinstance(reg.language, MockLanguageEvaluator) and isinstance(reg.gopt, PhonemePronunciationService)
    assert reg.mocked_names() == ["qwen"]


def test_both_disabled_keeps_whisper_and_acoustic_real():
    reg = build_registry(real_settings(pronunciation_enabled=False, qwen_enabled=False))
    assert isinstance(reg.stt, WhisperService) and isinstance(reg.acoustic, NumpyAcousticEngine)
    assert sorted(reg.mocked_names()) == ["gopt", "qwen"]


def test_disabled_services_load_without_any_model():
    reg = build_registry(real_settings(pronunciation_enabled=False, qwen_enabled=False))
    reg.gopt.load()
    reg.language.load()           # no torch / transformers / Ollama involved
    assert reg.gopt.is_loaded and reg.language.is_loaded


def test_transformers_backend_selected_when_configured():
    reg = build_registry(make_settings(mock_mode=False, qwen_backend="transformers"))
    assert isinstance(reg.language, QwenLanguageEvaluator) and not isinstance(reg.language, OllamaLanguageEvaluator)


def test_mock_mode_mocks_everything():
    reg = build_registry(make_settings(mock_mode=True))
    assert isinstance(reg.stt, MockSTT) and isinstance(reg.acoustic, MockAcousticEngine)
    assert sorted(reg.mocked_names()) == ["acoustic", "gopt", "qwen", "whisper"]


@pytest.mark.parametrize("field,value", [("pronunciation_backend", "nope"), ("qwen_backend", "nope")])
def test_unknown_backend_is_rejected(field, value):
    with pytest.raises(ValueError, match="nope"):
        build_registry(make_settings(mock_mode=False, **{field: value}))
