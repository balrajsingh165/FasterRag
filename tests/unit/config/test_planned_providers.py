"""A specified-but-unbuilt provider is refused when configuration loads (TASK-0439).

``pinecone`` was a member of the schema's closed list, so ``vector_db.provider: pinecone``
validated and the refusal came later, from the factory — at startup for the API, at first
use for anything that builds its adapter lazily. A setting the schema accepts and nothing
can serve is the defect ``config/`` exists to stop at load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fasterrag.adapters.vectordb import factory as vectordb_factory
from fasterrag.config import plugins
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError
from tests.support.plugins import install_distribution


def test_the_planned_names_are_the_ones_the_factory_refuses() -> None:
    """Two lists, so ``config/`` need not import ``adapters/``; shipping an adapter moves both."""
    assert frozenset(plugins.PLANNED_VECTOR_DB_PROVIDERS) == vectordb_factory._PLANNED_ADAPTERS


def test_no_name_is_both_planned_and_built() -> None:
    assert not frozenset(plugins.PLANNED_VECTOR_DB_PROVIDERS) & plugins.BUILTIN_VECTOR_DB_PROVIDERS


@pytest.mark.parametrize("provider", sorted(plugins.PLANNED_VECTOR_DB_PROVIDERS))
def test_a_planned_provider_is_refused_at_load_naming_its_task(provider: str) -> None:
    """The defect: the name validated, and only the factory said it was unbuilt."""
    task = plugins.PLANNED_VECTOR_DB_PROVIDERS[provider]

    with pytest.raises(ValidationError) as caught:
        Settings.model_validate({"vector_db": {"provider": provider}})

    message = caught.value.errors()[0]["msg"]
    assert f"{provider!r} is specified but its adapter is not built yet ({task})" in message
    assert "chroma, milvus, pgvector, qdrant, weaviate" in message
    assert "'fasterrag.vectordb'" in message


def test_the_loader_reports_a_planned_provider_against_its_key(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("vector_db:\n  provider: pinecone\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None, require_env=False)

    assert (
        "vector_db.provider: 'pinecone' is specified but its adapter is not built yet (TASK-0049)"
        in caught.value.detail
    )


def test_a_plugin_registering_a_planned_name_is_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory resolves a plugin before it refuses a planned name; validation agrees."""
    install_distribution(
        tmp_path, monkeypatch, {plugins.VECTOR_DB_GROUP: {"pinecone": "acme_pkg:Pinecone"}}
    )

    settings = Settings.model_validate(
        {"vector_db": {"provider": "pinecone", "api_key_env": "PINECONE_API_KEY"}}
    )

    assert settings.vector_db.provider == "pinecone"


def test_a_planned_vector_database_is_only_an_unknown_name_elsewhere() -> None:
    """``pinecone`` is planned as a vector database, not as an embedder or an LLM."""
    with pytest.raises(ValidationError) as caught:
        Settings.model_validate({"embeddings": {"provider": "pinecone"}})

    assert "'pinecone' is not an installed provider" in caught.value.errors()[0]["msg"]
