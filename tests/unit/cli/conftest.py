"""Fixtures shared by the CLI wrapper tests.

The wrappers under test are thin fronts over services that have their own tests. What is
exercised here is the wrapper's own work: the exit code it chooses, whether ``--json`` stays
a single parseable document, and the flags it interprets itself rather than passing on.
"""

from pathlib import Path

import pytest

VALID_CONFIG = """
app:
  host: 127.0.0.1
  port: 8000
vector_db:
  provider: qdrant
  mode: external
  api_key_env: null
embeddings:
  provider: huggingface
llm:
  provider: ollama
  api_key_env: null
"""


def write_config(tmp_path: Path, extra: str = "") -> str:
    """Write a valid config, optionally with extra top-level sections appended."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(VALID_CONFIG + extra, encoding="utf-8")
    return str(path)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Return the path to a valid config.yaml with the referenced env var present."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    return write_config(tmp_path)


@pytest.fixture
def bad_config(tmp_path: Path) -> str:
    """Return the path to a config the schema refuses."""
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval:\n  top_k: 9999\n", encoding="utf-8")
    return str(path)


class Closeable:
    """Stands in for anything a command builds and is obliged to close.

    Every one of these wrappers closes its adapter, router, or service in a ``finally``, so
    the interesting assertion is that the close still happens on the error path.
    """

    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        """Record one close."""
        self.closed += 1


def corpus(root: Path, *, nested: bool = False) -> Path:
    """Write a two-file corpus, optionally with a third file one level down."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text(
        "Either party may terminate this agreement on thirty days written notice.",
        encoding="utf-8",
    )
    (root / "b.txt").write_text(
        "Invoices are payable within forty-five days of the invoice date.", encoding="utf-8"
    )
    if nested:
        deeper = root / "annex"
        deeper.mkdir()
        (deeper / "c.txt").write_text(
            "The annex records the agreed schedule of rates.", encoding="utf-8"
        )
    return root
