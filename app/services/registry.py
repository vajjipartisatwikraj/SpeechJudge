"""One instance of each Shared_Service per process (Requirement 7)."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import ServiceLoadError
from app.core.logging import get_logger
from app.services.interfaces import (AcousticEngine, GoptService, LanguageEvaluator,
                                      SharedService, STTService)

log = get_logger(__name__)


@dataclass
class ServiceRegistry:
    stt: STTService
    gopt: GoptService
    acoustic: AcousticEngine
    language: LanguageEvaluator

    def all(self) -> list[SharedService]:
        return [self.stt, self.gopt, self.acoustic, self.language]

    def load_all(self) -> None:
        """Load every service once. Logs the failing service name, then re-raises."""
        for service in self.all():
            if service.is_loaded:
                continue
            try:
                log.info("Loading service %s%s", service.name, " (MOCK)" if service.mocked else "")
                service.load()
            except ServiceLoadError as exc:
                log.error("Service failed to load: %s (%s)", service.name, exc)
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("Service failed to load: %s (%s)", service.name, exc)
                raise ServiceLoadError(service.name, str(exc)) from exc

    def readiness(self) -> dict[str, bool]:
        return {s.name: s.is_loaded for s in self.all()}

    def not_loaded(self) -> list[str]:
        return [s.name for s in self.all() if not s.is_loaded]

    def mocked_names(self) -> list[str]:
        """Names (whisper | gopt | acoustic | qwen) of services that are placeholders."""
        return [s.name for s in self.all() if s.mocked]

    @property
    def ready(self) -> bool:
        return not self.not_loaded()


def _build_pronunciation(settings: Settings) -> GoptService:
    from app.services.mocks import MockGopt

    if not settings.pronunciation_enabled or settings.pronunciation_backend == "mock":
        log.warning("Pronunciation is MOCKED (fixed placeholder scores): results are NOT real")
        return MockGopt()
    if settings.pronunciation_backend == "phoneme":
        from app.services.pronunciation.phoneme_service import PhonemePronunciationService
        return PhonemePronunciationService(settings)
    if settings.pronunciation_backend == "gopt":
        from app.services.pronunciation.gopt_service import GoptPronunciationService
        return GoptPronunciationService(settings)
    raise ValueError(f"Unknown SJ_PRONUNCIATION_BACKEND '{settings.pronunciation_backend}' "
                     "(phoneme | gopt | mock)")


def _build_language(settings: Settings) -> LanguageEvaluator:
    from app.services.mocks import MockLanguageEvaluator

    if not settings.qwen_enabled:
        log.warning("Language evaluation (Qwen) is MOCKED (fixed placeholder scores): "
                    "Q&A / Storytelling results are NOT real")
        return MockLanguageEvaluator()
    if settings.qwen_backend == "ollama":
        from app.services.language.ollama_service import OllamaLanguageEvaluator
        return OllamaLanguageEvaluator(settings)
    if settings.qwen_backend == "transformers":
        from app.services.language.qwen_service import QwenLanguageEvaluator
        return QwenLanguageEvaluator(settings)
    raise ValueError(f"Unknown SJ_QWEN_BACKEND '{settings.qwen_backend}' (ollama | transformers)")


def build_registry(settings: Settings) -> ServiceRegistry:
    if settings.mock_mode:      # everything mocked (development / demos)
        from app.services.mocks import (MockAcousticEngine, MockGopt, MockLanguageEvaluator,
                                        MockSTT)
        return ServiceRegistry(stt=MockSTT(), gopt=MockGopt(), acoustic=MockAcousticEngine(),
                               language=MockLanguageEvaluator())

    # Heavy backends are imported lazily so mock mode and the API gateway stay light.
    from app.services.acoustic.engine import NumpyAcousticEngine
    from app.services.stt.whisper_service import WhisperService

    return ServiceRegistry(stt=WhisperService(settings), gopt=_build_pronunciation(settings),
                           acoustic=NumpyAcousticEngine(), language=_build_language(settings))
