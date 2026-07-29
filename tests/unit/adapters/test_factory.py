import pytest

from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.adapters.vectordb.factory import (
    ENTRY_POINT_GROUP,
    available_providers,
    create_vector_db_adapter,
    resolve_adapter_class,
)
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError


def test_qdrant_is_the_built_in_reference_adapter() -> None:
    assert resolve_adapter_class("qdrant") is QdrantAdapter
    assert available_providers()["qdrant"] == "built-in"


def test_factory_builds_the_configured_provider() -> None:
    adapter = create_vector_db_adapter(Settings())
    assert isinstance(adapter, QdrantAdapter)
    assert isinstance(adapter, VectorDBAdapter)


def test_construction_opens_no_connection() -> None:
    adapter = create_vector_db_adapter(Settings())
    assert isinstance(adapter, QdrantAdapter)
    assert adapter._client is None


@pytest.mark.parametrize("provider", ["milvus", "weaviate", "pinecone", "pgvector", "chroma"])
def test_documented_but_unbuilt_providers_fail_clearly(provider: str) -> None:
    with pytest.raises(ConfigError, match="not built") as caught:
        resolve_adapter_class(provider)
    assert ENTRY_POINT_GROUP in caught.value.detail


def test_unknown_provider_lists_what_is_available() -> None:
    with pytest.raises(ConfigError, match="not registered") as caught:
        resolve_adapter_class("nonexistent")
    assert "qdrant" in caught.value.detail


def test_plugins_cannot_override_a_built_in_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class Hijack:
        pass

    monkeypatch.setattr(
        "fasterrag.adapters.vectordb.factory._plugin_entry_points",
        lambda: {"qdrant": Hijack()},
    )
    assert resolve_adapter_class("qdrant") is QdrantAdapter


def test_a_plugin_that_is_not_an_adapter_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class NotAnAdapter:
        pass

    class Entry:
        value = "somewhere:NotAnAdapter"

        def load(self) -> type:
            return NotAnAdapter

    monkeypatch.setattr(
        "fasterrag.adapters.vectordb.factory._plugin_entry_points",
        lambda: {"custom": Entry()},
    )
    with pytest.raises(ConfigError, match="not a VectorDBAdapter subclass"):
        resolve_adapter_class("custom")


def test_a_conformant_plugin_is_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Custom(QdrantAdapter):
        pass

    class Entry:
        value = "somewhere:Custom"

        def load(self) -> type:
            return Custom

    monkeypatch.setattr(
        "fasterrag.adapters.vectordb.factory._plugin_entry_points",
        lambda: {"custom": Entry()},
    )
    assert resolve_adapter_class("custom") is Custom
    assert available_providers()["custom"] == "plugin (somewhere:Custom)"
