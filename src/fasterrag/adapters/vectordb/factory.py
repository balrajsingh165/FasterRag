"""Selects the concrete vector database adapter from configuration.

``vector_db.provider`` is the entire backend-swap surface: changing that one line picks
a different adapter with no application-code change (``docs/adr/ADR-0002``). Third
parties register additional providers through the ``fasterrag.vectordb`` entry point
without forking (``docs/python-api.md`` §Extending).

Built-in names always win over entry points, so an installed package cannot silently
take over ``qdrant`` and intercept a deployment's vectors.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Final

from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError

__all__ = [
    "ENTRY_POINT_GROUP",
    "AdapterFactory",
    "available_providers",
    "create_vector_db_adapter",
]

ENTRY_POINT_GROUP: Final = "fasterrag.vectordb"

AdapterFactory = Callable[[Settings], VectorDBAdapter]

_BUILTIN_ADAPTERS: Final[dict[str, str]] = {
    "qdrant": "fasterrag.adapters.vectordb.qdrant:QdrantAdapter",
}

# TODO: TASK-0049 ships the milvus, weaviate, pinecone, pgvector, and chroma adapters.
_PLANNED_ADAPTERS: Final[frozenset[str]] = frozenset(
    {"milvus", "weaviate", "pinecone", "pgvector", "chroma"}
)


def _plugin_entry_points() -> dict[str, EntryPoint]:
    """Return third-party adapters registered under the entry-point group."""
    return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}


def available_providers() -> dict[str, str]:
    """Return every selectable provider name mapped to where it came from."""
    providers = dict.fromkeys(_BUILTIN_ADAPTERS, "built-in")
    for name, entry in _plugin_entry_points().items():
        if name not in providers:
            providers[name] = f"plugin ({entry.value})"
    return providers


def _load_builtin(target: str) -> type[VectorDBAdapter]:
    """Import a built-in adapter from its ``module:attribute`` reference."""
    module_name, _, attribute = target.partition(":")
    module = import_module(module_name)
    loaded: object = getattr(module, attribute)
    return _require_adapter_class(loaded, target)


def _require_adapter_class(loaded: object, source: str) -> type[VectorDBAdapter]:
    """Confirm a resolved object really implements the adapter contract."""
    if not (isinstance(loaded, type) and issubclass(loaded, VectorDBAdapter)):
        raise ConfigError(
            f"{source} is not a VectorDBAdapter subclass; a registered provider must "
            "implement the adapter contract and pass the shared contract test suite"
        )
    return loaded


def resolve_adapter_class(provider: str) -> type[VectorDBAdapter]:
    """Return the adapter class registered for ``provider``.

    Args:
        provider: The value of ``vector_db.provider``.

    Returns:
        The adapter class, built-in or plugin-registered.

    Raises:
        ConfigError: If the provider is unknown, is documented but not yet implemented,
            or resolves to something that is not a ``VectorDBAdapter``.
    """
    builtin = _BUILTIN_ADAPTERS.get(provider)
    if builtin is not None:
        return _load_builtin(builtin)

    plugin = _plugin_entry_points().get(provider)
    if plugin is not None:
        return _require_adapter_class(
            plugin.load(), f"{ENTRY_POINT_GROUP} entry point {provider!r}"
        )

    if provider in _PLANNED_ADAPTERS:
        raise ConfigError(
            f"vector_db.provider {provider!r} is specified but its adapter is not built "
            "yet in this version; use 'qdrant', or register your own implementation "
            f"through the {ENTRY_POINT_GROUP!r} entry point"
        )

    known = ", ".join(sorted(available_providers()))
    raise ConfigError(
        f"vector_db.provider {provider!r} is not registered; available providers: {known}"
    )


def create_vector_db_adapter(settings: Settings) -> VectorDBAdapter:
    """Build the adapter named by ``vector_db.provider``.

    Args:
        settings: Validated configuration. The whole object is passed so an adapter can
            read its connection settings and the shared reliability timeouts.

    Returns:
        A ready-to-use adapter. No connection is opened until it is first used.

    Raises:
        ConfigError: If the configured provider cannot be resolved.
    """
    adapter_class = resolve_adapter_class(settings.vector_db.provider)
    return adapter_class(settings)
