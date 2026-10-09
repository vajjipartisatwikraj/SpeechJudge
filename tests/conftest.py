"""Pytest fixtures and options.

Layout:  tests/unit         fast, isolated (no models, no network)
         tests/integration  the FastAPI app wired to mock services
         tests/e2e          real models / running stack; skipped unless --real is given
"""
from __future__ import annotations

import os

# Never let a developer's backend .env or shell variables leak into the test run (must be set before
# app imports). A leaked SJ_BENCHMARKS_ENABLED=true would let a test start the real GPU benchmarks.
for _name in [n for n in os.environ if n.startswith("SJ_") and not n.startswith("SJ_E2E_")]:
    del os.environ[_name]
os.environ["SJ_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "_no_such_env_file")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

collect_ignore = ["load"]        # tests/load holds manual load/benchmark scripts, not tests

from app.main import create_app  # noqa: E402
from app.runtime import build_runtime  # noqa: E402
from app.services.registry import ServiceRegistry  # noqa: E402
from tests.support import make_services, make_settings  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--real", action="store_true", default=False,
                     help="run end-to-end tests that need real models / a running stack")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--real"):
        return
    skip = pytest.mark.skip(reason="needs real models or a running stack; use --real")
    for item in items:
        if "tests/e2e" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.real)
        if "real" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def services() -> ServiceRegistry:
    return make_services()


@pytest.fixture
def runtime(services):
    rt = build_runtime(make_settings(), services=services)
    yield rt
    rt.shutdown()


@pytest.fixture
def client(runtime):
    with TestClient(create_app(runtime=runtime)) as c:
        yield c
