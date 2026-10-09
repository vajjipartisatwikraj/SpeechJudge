import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.paths import BACKEND_DIR, DEFAULT_ACCENT_PROFILE, DEFAULT_SCORING_CONFIG

STRONG = "x" * 32


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults_point_at_existing_config_files():
    s = settings()
    assert DEFAULT_SCORING_CONFIG.exists() and DEFAULT_ACCENT_PROFILE.exists()
    assert s.scoring_config_path == str(DEFAULT_SCORING_CONFIG)
    assert s.accent_profile_path == str(DEFAULT_ACCENT_PROFILE)
    assert BACKEND_DIR.name == "speech-judge"


def test_credentials_are_split_and_trimmed():
    assert settings(service_credentials=" a , b,,c ").credential_set == frozenset({"a", "b", "c"})
    assert settings().credential_set == frozenset()


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("SJ_PORT", "9123")
    monkeypatch.setenv("SJ_QWEN_BACKEND", "ollama")
    s = settings()
    assert s.port == 9123 and s.qwen_backend == "ollama"


def test_local_environment_is_permissive():
    assert not settings().is_production


def test_production_accepts_safe_configuration():
    s = settings(environment="production", service_credentials=STRONG)
    assert s.is_production


@pytest.mark.parametrize("overrides,fragment", [
    ({"service_credentials": ""}, "SJ_SERVICE_CREDENTIALS is empty"),
    ({"service_credentials": "short"}, "at least"),
    ({"service_credentials": STRONG, "mock_mode": True}, "SJ_MOCK_MODE"),
    ({"service_credentials": STRONG, "enable_docs": True}, "SJ_ENABLE_DOCS"),
    ({"service_credentials": STRONG, "pronunciation_enabled": False}, "SJ_PRONUNCIATION_ENABLED"),
    ({"service_credentials": STRONG, "qwen_enabled": False}, "SJ_QWEN_ENABLED"),
])
def test_production_refuses_unsafe_configuration(overrides, fragment):
    with pytest.raises(ValidationError, match=fragment):
        settings(environment="production", **overrides)


def test_relative_config_paths_resolve_against_backend_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                       # a different working directory must not matter
    s = settings(scoring_config_path="config/scoring.yaml",
                 accent_profile_path="config/accents/indian_english.yaml")
    assert s.scoring_config_path == str(BACKEND_DIR / "config" / "scoring.yaml")
    assert s.accent_profile_path == str(BACKEND_DIR / "config" / "accents" / "indian_english.yaml")


def test_service_switches_default_to_enabled_and_read_from_env(monkeypatch):
    s = settings()
    assert s.pronunciation_enabled is True and s.qwen_enabled is True
    monkeypatch.setenv("SJ_PRONUNCIATION_ENABLED", "false")
    monkeypatch.setenv("SJ_QWEN_ENABLED", "false")
    s = settings()
    assert s.pronunciation_enabled is False and s.qwen_enabled is False


def test_gpu_present_defaults_to_the_cpu_profile():
    s = settings()
    assert s.gpu_present is False and s.hardware_profile == "cpu"
    assert (s.device, s.whisper_model, s.whisper_compute_type) == ("cpu", "small.en", "int8")
    assert (s.whisper_batch_size, s.phoneme_batch_size, s.section_processing_timeout_s) == (1, 1, 3600.0)


def test_gpu_present_selects_the_gpu_profile():
    s = settings(gpu_present=True)
    assert s.hardware_profile == "gpu"
    assert (s.device, s.whisper_model, s.whisper_compute_type) == ("cuda", "large-v3-turbo", "float16")
    assert (s.whisper_batch_size, s.phoneme_batch_size, s.section_processing_timeout_s) == (10, 10, 900.0)


def test_gpu_present_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SJ_GPU_PRESENT", "true")
    s = settings()
    assert s.gpu_present and s.device == "cuda" and s.whisper_batch_size == 10
    monkeypatch.setenv("SJ_GPU_PRESENT", "false")
    s = settings()
    assert not s.gpu_present and s.device == "cpu" and s.whisper_batch_size == 1


@pytest.mark.parametrize("gpu", [False, True])
def test_explicit_values_beat_the_profile(gpu):
    s = settings(gpu_present=gpu, whisper_model="medium.en", whisper_batch_size=4,
                 section_processing_timeout_s=123.0)
    assert (s.whisper_model, s.whisper_batch_size, s.section_processing_timeout_s) == ("medium.en", 4, 123.0)
    assert s.phoneme_batch_size == (10 if gpu else 1)            # untouched fields still follow the profile


def test_explicit_values_from_a_dotenv_file_beat_the_profile(tmp_path):
    env = tmp_path / "x.env"
    env.write_text("SJ_GPU_PRESENT=true\nSJ_WHISPER_COMPUTE_TYPE=int8_float16\n", encoding="utf-8")
    s = Settings(_env_file=str(env))
    assert s.device == "cuda" and s.whisper_compute_type == "int8_float16" and s.whisper_model == "large-v3-turbo"


@pytest.mark.parametrize("kw,fragment", [
    ({"gpu_present": True, "device": "cpu"}, "SJ_GPU_PRESENT=true"),
    ({"gpu_present": False, "device": "cuda"}, "SJ_GPU_PRESENT=false"),
])
def test_device_contradicting_the_switch_is_refused(kw, fragment):
    with pytest.raises(ValidationError, match=fragment):
        settings(**kw)


def test_matching_explicit_device_is_accepted():
    assert settings(gpu_present=True, device="cuda").device == "cuda"
    assert settings(gpu_present=False, device="cpu").device == "cpu"


def test_switches_may_be_off_outside_production():
    s = settings(environment="local", pronunciation_enabled=False, qwen_enabled=False)
    assert not s.pronunciation_enabled and not s.qwen_enabled