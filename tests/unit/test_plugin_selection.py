"""A third-party provider, named in ``config.yaml``, ingests a document and answers a query.

This is the proof TASK-0439 asks for. ``tests/unit/adapters/test_plugin_entry_points.py``
replaces each factory's lookup function and calls ``resolve_adapter_class`` directly, so
it never touched ``Settings`` — which is how a closed ``Literal`` in the schema refused
every plugin for as long as the factories have supported them, with that suite green.

Nothing inside fasterRag is replaced here. A fake distribution is put on ``sys.path``
(``tests/support/plugins.py``), so ``importlib.metadata`` discovers its entry points the
way it discovers an installed package's; the configuration is a file, read by the real
loader through ``FasterRag.from_config``; and the adapters that end up serving the query
(``tests/support/acme_plugin.py``) are named by no factory and handed to no service — the
only route to them is ``provider: acme``.

The ingest runs the production pipeline, parse workers included. Selection is decided in
the parent process, but an ingest that only worked with the worker pool swapped out would
leave open whether a plugin survives the path a real deployment takes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fasterrag.config.loader import load_settings
from fasterrag.errors import ConfigError
from fasterrag.facade import FasterRag
from tests.support.acme_plugin import ANSWER_PREFIX, AcmeEmbedder, AcmeLLM, AcmeVectorDB
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
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the acme distribution and work from an empty directory holding no ``.env``."""
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
    return root


def write_config(root: Path, text: str = CONFIG) -> Path:
    """Write a ``config.yaml`` into the deployment directory."""
    path = root / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


async def test_a_plugin_named_in_config_yaml_ingests_and_answers_a_query(workspace: Path) -> None:
    """The defect: ``provider: acme`` was refused at load, so no plugin could ever be selected."""
    document = workspace / "warranty.md"
    document.write_text(
        "# Warranty\n\nThe acme turbo widget warranty lasts thirty days.\n", encoding="utf-8"
    )

    async with FasterRag.from_config(write_config(workspace)) as rag:
        assert type(rag.vector_db) is AcmeVectorDB
        assert type(rag.embeddings.default) is AcmeEmbedder

        job = await rag.ingest([str(document)])
        assert job.status == "completed"
        assert [info.name for info in await rag.collections()] == ["default"]

        chunks = await rag.retrieve("how long is the turbo widget warranty?")
        answer = await rag.query("how long is the turbo widget warranty?")

        generation = rag._generation_service()
        assert type(generation.llm) is AcmeLLM

    assert chunks
    assert "thirty days" in chunks[0].text
    assert answer.answer is not None
    assert answer.answer.startswith(ANSWER_PREFIX)
    assert "thirty days" in answer.answer
    assert answer.citations


async def test_a_plugin_named_in_config_yaml_streams_an_answer(workspace: Path) -> None:
    """The streamed path builds its adapters through the same factories as the whole one."""
    document = workspace / "warranty.md"
    document.write_text("The acme turbo widget warranty lasts thirty days.\n", encoding="utf-8")

    async with FasterRag.from_config(write_config(workspace)) as rag:
        await rag.ingest([str(document)])
        events = [event async for event in rag.query_stream("turbo widget warranty")]

    text = "".join(str(event.data.get("text", "")) for event in events if event.type == "token")
    assert text.startswith(ANSWER_PREFIX)


def test_an_unknown_provider_fails_at_load_listing_what_is_installed(workspace: Path) -> None:
    """Fail-fast did not weaken: a typo never reaches a factory, and the error says what would."""
    path = write_config(workspace, CONFIG.replace("provider: acme", "provider: acmee", 1))

    with pytest.raises(ConfigError) as caught:
        FasterRag.from_config(path)

    detail = caught.value.detail
    assert "vector_db.provider: 'acmee' is not an installed provider" in detail
    assert "built-in providers: chroma, milvus, pgvector, qdrant, weaviate" in detail
    assert "plugins installed under the 'fasterrag.vectordb' entry-point group: acme" in detail


def test_a_plugin_that_is_not_installed_fails_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same file, on a machine without the package: refused before anything is built."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as caught:
        load_settings(write_config(tmp_path), env_file=None, require_env=False)

    detail = caught.value.detail
    assert "vector_db.provider: 'acme' is not an installed provider" in detail
    assert "embeddings.provider: 'acme' is not an installed provider" in detail
    assert "llm.provider: 'acme' is not an installed provider" in detail
    assert "entry-point group: none" in detail
