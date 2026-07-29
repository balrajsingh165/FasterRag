import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "test-openai-key",
    "QDRANT_API_KEY": "test-qdrant-key",
    "FASTERRAG_API_KEY": "test-control-plane-key",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Isolate os.environ so loading a .env file cannot leak between tests."""
    isolated = dict(os.environ)
    isolated.update(REQUIRED_ENV_VARS)
    monkeypatch.setattr(os, "environ", isolated)
    return isolated


@pytest.fixture(scope="session")
def canonical_config() -> Path:
    """Return the committed default config.yaml at the repository root."""
    return REPO_ROOT / "config.yaml"


@pytest.fixture(scope="session")
def config_reference() -> Path:
    """Return docs/config-reference.md, which carries the complete example block."""
    return REPO_ROOT / "docs" / "config-reference.md"
