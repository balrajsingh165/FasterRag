"""A plugin built for another adapter API is refused when it is resolved, naming both versions.

Before TASK-0486 the base interfaces carried no version, so a plugin written against an older
``VectorDBAdapter`` (a renamed method, a changed signature) was loaded, instantiated and then
failed at its first call with an ``AttributeError`` or a ``TypeError`` that named neither the
plugin nor the reason. A plugin now declares the adapter API it was built for, and the factory
compares it with the one this installation provides before handing the class to anything.
"""

from __future__ import annotations

import re
from importlib.metadata import EntryPoint
from types import ModuleType

import pytest

from fasterrag.adapters.embeddings import base as embeddings_base
from fasterrag.adapters.embeddings import factory as embeddings_factory
from fasterrag.adapters.llm import base as llm_base
from fasterrag.adapters.llm import factory as llm_factory
from fasterrag.adapters.plugin_api import require_compatible_api
from fasterrag.adapters.vectordb import base as vectordb_base
from fasterrag.adapters.vectordb import factory as vectordb_factory
from fasterrag.errors import ConfigError, ErrorCode

KINDS = [
    pytest.param(vectordb_factory, vectordb_base, vectordb_base.VectorDBAdapter, id="vectordb"),
    pytest.param(
        embeddings_factory, embeddings_base, embeddings_base.EmbeddingAdapter, id="embeddings"
    ),
    pytest.param(llm_factory, llm_base, llm_base.LLMAdapter, id="llm"),
]


def register(monkeypatch: pytest.MonkeyPatch, factory: ModuleType, loaded: object) -> None:
    """Pretend an installed distribution registered ``acme`` under the factory's group."""
    entry = EntryPoint(name="acme", value="acme.adapter:Adapter", group="ignored")
    monkeypatch.setattr(entry.__class__, "load", lambda self: loaded, raising=False)
    monkeypatch.setattr(factory, "_plugin_entry_points", lambda: {"acme": entry})


def plugin(base: type, declared: str | None) -> type:
    """Return a plugin class declaring ``declared`` as its adapter API, or nothing."""
    namespace: dict[str, object] = {}
    if declared is not None:
        namespace["adapter_api_version"] = declared
    return type("AcmeAdapter", (base,), namespace)


def bumped(version: str, *, major: int = 0, minor: int = 0) -> str:
    """Return ``version`` with its major or minor component moved."""
    host_major, host_minor = (int(part) for part in version.split("."))
    return f"{host_major + major}.{host_minor + minor}"


@pytest.mark.parametrize(("factory", "base_module", "base"), KINDS)
def test_every_base_interface_states_its_adapter_api_version(
    factory: ModuleType, base_module: ModuleType, base: type
) -> None:
    """A plugin author cannot target a version the interface does not state."""
    version = base_module.ADAPTER_API_VERSION

    assert re.fullmatch(r"\d+\.\d+", version)
    assert "ADAPTER_API_VERSION" in base_module.__all__


@pytest.mark.parametrize(("factory", "base_module", "base"), KINDS)
def test_a_plugin_built_for_the_current_api_is_resolved(
    factory: ModuleType, base_module: ModuleType, base: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check refuses only what it has a reason to refuse."""
    accepted = plugin(base, base_module.ADAPTER_API_VERSION)
    register(monkeypatch, factory, accepted)

    assert factory.resolve_adapter_class("acme") is accepted


@pytest.mark.parametrize(("factory", "base_module", "base"), KINDS)
def test_a_plugin_built_for_another_major_version_is_refused_naming_both(
    factory: ModuleType, base_module: ModuleType, base: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: such a plugin loaded and failed at its first call, naming neither version."""
    host = base_module.ADAPTER_API_VERSION
    declared = bumped(host, major=1)
    register(monkeypatch, factory, plugin(base, declared))

    with pytest.raises(ConfigError) as caught:
        factory.resolve_adapter_class("acme")

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert declared in caught.value.detail
    assert host in caught.value.detail
    assert "acme" in caught.value.detail


@pytest.mark.parametrize(("factory", "base_module", "base"), KINDS)
def test_a_plugin_built_for_a_newer_minor_version_is_refused(
    factory: ModuleType, base_module: ModuleType, base: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A minor version adds members; a plugin written against them would call what is absent."""
    host = base_module.ADAPTER_API_VERSION
    declared = bumped(host, minor=1)
    register(monkeypatch, factory, plugin(base, declared))

    with pytest.raises(ConfigError, match=re.escape(declared)):
        factory.resolve_adapter_class("acme")


@pytest.mark.parametrize(("factory", "base_module", "base"), KINDS)
def test_a_plugin_that_declares_no_version_is_refused_naming_the_one_to_declare(
    factory: ModuleType, base_module: ModuleType, base: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a declaration there is nothing to compare, so the mistake is named instead."""
    register(monkeypatch, factory, plugin(base, None))

    with pytest.raises(ConfigError) as caught:
        factory.resolve_adapter_class("acme")

    assert "adapter_api_version" in caught.value.detail
    assert base_module.ADAPTER_API_VERSION in caught.value.detail


@pytest.mark.parametrize(("factory", "base_module", "base"), KINDS)
def test_a_built_in_adapter_is_not_asked_for_a_declaration(
    factory: ModuleType, base_module: ModuleType, base: type
) -> None:
    """Built-ins ship with the interface they implement, so there is nothing to mismatch."""
    for name in ("chroma", "openai", "ollama"):
        if name in factory._BUILTIN_ADAPTERS:
            assert issubclass(factory.resolve_adapter_class(name), base)
            return
    pytest.fail("no lightweight built-in adapter to resolve in this group")


@pytest.mark.parametrize("declared", ["one", "1", "1.0.0", "", "v1.0"])
def test_a_malformed_declaration_is_refused_as_such(declared: str) -> None:
    """A version that cannot be compared is not silently treated as compatible."""
    with pytest.raises(ConfigError, match=r"MAJOR\.MINOR"):
        require_compatible_api(
            type("Acme", (), {"adapter_api_version": declared}), host="1.0", source="acme"
        )


def test_a_declaration_that_is_not_a_string_is_refused() -> None:
    """``adapter_api_version = 1.0`` is a float, and a float cannot hold ``1.10``."""
    with pytest.raises(ConfigError, match="adapter_api_version"):
        require_compatible_api(
            type("Acme", (), {"adapter_api_version": 1.0}), host="1.0", source="acme"
        )
