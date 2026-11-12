"""A query answered through the Python facade can be replayed through the Python facade.

Found by ``examples/replay_trace.py`` (TASK-0498): ``FasterRag.replay`` loads the trace from
the configured trace store, but ``FasterRag.query`` built its generation service without one,
so a facade query was never stored and replaying its ``trace_id`` failed with ``NOT_FOUND``.
Only queries answered over REST, whose route hands the service a store, could be replayed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fasterrag.config.loader import load_settings
from fasterrag.facade import FasterRag
from tests.support.plugins import install_distribution

PLUGIN = "tests.support.acme_plugin"

CONFIG = """\
vector_db:
  provider: acme
  api_key_env: null
embeddings:
  provider: acme
  model: acme-embed
llm:
  provider: acme
  model: acme-chat
  api_key_env: null
retrieval:
  rerank: false
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the acme providers and write a config selecting them in an empty directory."""
    install_distribution(
        tmp_path,
        monkeypatch,
        {
            "fasterrag.vectordb": {"acme": f"{PLUGIN}:AcmeVectorDB"},
            "fasterrag.embeddings": {"acme": f"{PLUGIN}:AcmeEmbedder"},
            "fasterrag.llm": {"acme": f"{PLUGIN}:AcmeLLM"},
        },
    )
    root = tmp_path / "deployment"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("FASTERRAG_API_KEY", "unit-test-control-plane-key")
    (root / "warranty.md").write_text(
        "The acme turbo widget warranty lasts thirty days.\n", encoding="utf-8"
    )
    path = root / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


async def test_a_facade_query_is_stored_and_replays_through_the_facade(config: Path) -> None:
    """The defect: the replay raised NOT_FOUND because the facade never stored the trace."""
    async with FasterRag.from_config(config) as rag:
        await rag.ingest([str(config.parent / "warranty.md")])
        answer = await rag.query("how long is the turbo widget warranty?")
        candidate = load_settings(config, overrides=["retrieval.top_k=1"])
        result = await rag.replay(answer.trace_id, candidate)

    assert result.trace_id == answer.trace_id
    assert result.original_answer == answer.answer
    assert result.config_changes


async def test_a_streamed_facade_query_is_stored_too(config: Path) -> None:
    """The streamed path builds its service through the same helper and must store alike."""
    async with FasterRag.from_config(config) as rag:
        await rag.ingest([str(config.parent / "warranty.md")])
        trace_ids = [
            str(event.data["trace_id"])
            async for event in rag.query_stream("turbo widget warranty")
            if "trace_id" in event.data
        ]
        assert trace_ids
        result = await rag.replay(trace_ids[0])

    assert result.trace_id == trace_ids[0]


async def test_a_query_stores_nothing_when_traces_are_off(config: Path) -> None:
    """``traces.store: false`` is still honoured: no trace, so a replay is refused."""
    config.write_text(CONFIG + "traces:\n  store: false\n", encoding="utf-8")

    async with FasterRag.from_config(config) as rag:
        await rag.ingest([str(config.parent / "warranty.md")])
        answer = await rag.query("how long is the turbo widget warranty?")
        with pytest.raises(Exception, match="no stored trace"):
            await rag.replay(answer.trace_id)
