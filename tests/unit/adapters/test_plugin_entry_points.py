from importlib.metadata import EntryPoint

import pytest

from fasterrag.adapters.embeddings import factory as embeddings_factory
from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.llm import factory as llm_factory
from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.adapters.vectordb import factory as vectordb_factory
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.errors import ConfigError

GROUPS = [
    (vectordb_factory, VectorDBAdapter, "fasterrag.vectordb"),
    (embeddings_factory, EmbeddingAdapter, "fasterrag.embeddings"),
    (llm_factory, LLMAdapter, "fasterrag.llm"),
]


def register(monkeypatch: pytest.MonkeyPatch, factory: object, name: str, loaded: object) -> None:
    """Pretend a third-party distribution registered ``name`` under the factory's group."""
    entry = EntryPoint(name=name, value="acme.adapter:Adapter", group="ignored")
    monkeypatch.setattr(entry.__class__, "load", lambda self: loaded, raising=False)
    monkeypatch.setattr(factory, "_plugin_entry_points", lambda: {name: entry})


@pytest.mark.parametrize(("factory", "base", "group"), GROUPS)
def test_the_documented_group_name_is_the_one_used(factory: object, base: type, group: str) -> None:
    """The group name is a SemVer-stable extension contract; renaming one orphans plugins."""
    assert group == factory.ENTRY_POINT_GROUP  # type: ignore[attr-defined]


@pytest.mark.parametrize(("factory", "base", "group"), GROUPS)
def test_a_registered_plugin_is_resolvable(
    factory: object, base: type, group: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fork required: implementing the base class and registering it must be enough."""
    plugin = type("AcmeAdapter", (base,), {})
    register(monkeypatch, factory, "acme", plugin)

    assert factory.resolve_adapter_class("acme") is plugin  # type: ignore[attr-defined]


@pytest.mark.parametrize(("factory", "base", "group"), GROUPS)
def test_a_plugin_is_listed_with_its_origin(
    factory: object, base: type, group: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    register(monkeypatch, factory, "acme", type("AcmeAdapter", (base,), {}))

    providers = factory.available_providers()  # type: ignore[attr-defined]

    assert "plugin" in providers["acme"]


@pytest.mark.parametrize(("factory", "base", "group"), GROUPS)
def test_a_plugin_cannot_take_over_a_built_in_name(
    factory: object, base: type, group: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed package silently replacing the configured provider is a supply-chain hole."""
    builtin = next(iter(factory._BUILTIN_ADAPTERS))  # type: ignore[attr-defined]
    register(monkeypatch, factory, builtin, type("Impostor", (base,), {}))

    assert factory.available_providers()[builtin] == "built-in"  # type: ignore[attr-defined]


@pytest.mark.parametrize(("factory", "base", "group"), GROUPS)
def test_a_plugin_that_is_not_an_adapter_is_rejected(
    factory: object, base: type, group: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the failure surfaces later as a missing method on an unrelated call."""
    register(monkeypatch, factory, "acme", object)

    with pytest.raises(ConfigError, match=r"adapter contract|not a"):
        factory.resolve_adapter_class("acme")  # type: ignore[attr-defined]
