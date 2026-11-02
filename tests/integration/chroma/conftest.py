"""Provisioning for the Chroma contract run, which is a directory and nothing else.

Chroma is the one backend fasterRag supports that needs no server, so there is no
container to start, no port to wait on, and no credential to plant: the whole fixture is
a temporary directory the embedded store persists into. That is also why this suite runs
everywhere the unit suite runs, Docker or no Docker.

The directory is session-scoped rather than per-test because opening a persistent Chroma
directory costs more than every case in the suite put together, and the cases already
isolate themselves with a unique collection name apiece.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fasterrag.config.schema import Settings


def chroma_settings(path: Path) -> Settings:
    """Return settings pointing the Chroma adapter at one persistent directory."""
    return Settings.model_validate(
        {"vector_db": {"provider": "chroma", "chroma": {"path": str(path)}}}
    )


@pytest.fixture(scope="session")
def chroma_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the directory the suite's embedded store lives in.

    Never torn down: Windows keeps the HNSW segment files open until the process exits,
    so pytest's own retention policy is what cleans up, one run later.
    """
    return tmp_path_factory.mktemp("chroma-store")


@pytest.fixture(scope="session")
def chroma(chroma_path: Path) -> Settings:
    """Return the settings every case in this package builds its adapter from."""
    return chroma_settings(chroma_path)


@pytest.fixture(scope="session")
def chroma_not_a_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a path that exists as a file, which the adapter must refuse to open."""
    blocked = tmp_path_factory.mktemp("chroma-blocked") / "store"
    blocked.write_text("not a directory", encoding="utf-8")
    return blocked
