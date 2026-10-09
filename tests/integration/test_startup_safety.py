import time

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ScoringConfigError, ServiceLoadError
from app.main import create_app
from app.runtime import build_runtime
from tests.support import make_services, make_settings


def test_invalid_scoring_config_blocks_startup(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("scoring:\n  READING: {a: 0.5}\n")
    with pytest.raises(ScoringConfigError):
        build_runtime(make_settings(scoring_config_path=str(bad)))


def test_service_load_failure_calls_fatal():
    reg = make_services()
    reg.stt._loaded = False

    def fail():
        raise ServiceLoadError("whisper", "boom")
    reg.stt.load = fail
    fired = []
    rt = build_runtime(make_settings(), services=reg)
    with TestClient(create_app(runtime=rt, on_fatal=lambda: fired.append(1))):
        time.sleep(0.3)
    assert fired == [1]
    rt.shutdown()
