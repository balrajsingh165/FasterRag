"""Selects the concrete embedding adapter from configuration.

Mirrors the vector database factory: ``embeddings.provider`` is the whole selection
surface, third parties register providers through the ``fasterrag.embeddings`` entry
point, and built-in names always win so an installed package cannot take over
``huggingface`` and silently reroute a deployment's vectors.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Final

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError

__all__ = [
    "ENTRY_POINT_GROUP",
    "available_providers",
    "create_embedding_adapter",
    "resolve_adapter_class",
]

ENTRY_POINT_GROUP: Final = "fasterrag.embeddings"

_BUILTIN_ADAPTERS: Final[dict[str, str]] = {
    "huggingface": "fasterrag.adapters.embeddings.huggingface:HuggingFaceEmbedder",
    "openai": "fasterrag.adapters.embeddings.openai:OpenAIEmbedder",
    "cohere": "fasterrag.adapters.embeddings.cohere:CohereEmbedder",
    "ollama": "fasterrag.adapters.embeddings.ollama:OllamaEmbedder",
}


def _plugin_entry_points() -> dict[str, EntryPoint]:
    """Return third-party embedding providers registered under the entry-point group."""
    return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}


def available_providers() -> dict[str, str]:
    """Return every selectable provider name mapped to where it came from."""
    providers = dict.fromkeys(_BUILTIN_ADAPTERS, "built-in")
    for name, entry in _plugin_entry_points().items():
        if name not in providers:
            providers[name] = f"plugin ({entry.value})"
    return providers


def _require_adapter_class(loaded: object, source: str) -> type[EmbeddingAdapter]:
    """Confirm a resolved object really implements the embedding contract."""
    if not (isinstance(loaded, type) and issubclass(loaded, EmbeddingAdapter)):
        raise ConfigError(
            f"{source} is not an EmbeddingAdapter subclass; a registered provider must "
            "implement the adapter contract"
        )
    return loaded


def resolve_adapter_class(provider: str) -> type[EmbeddingAdapter]:
    """Return the adapter class registered for ``provider``.

    Raises:
        ConfigError: If the provider is unknown or resolves to something that is not an
            ``EmbeddingAdapter``.
    """
    builtin = _BUILTIN_ADAPTERS.get(provider)
    if builtin is not None:
        module_name, _, attribute = builtin.partition(":")
        return _require_adapter_class(getattr(import_module(module_name), attribute), builtin)

    plugin = _plugin_entry_points().get(provider)
    if plugin is not None:
        return _require_adapter_class(
            plugin.load(), f"{ENTRY_POINT_GROUP} entry point {provider!r}"
        )

    known = ", ".join(sorted(available_providers()))
    raise ConfigError(
        f"embeddings.provider {provider!r} is not registered; available providers: {known}"
    )


def create_embedding_adapter(settings: Settings) -> EmbeddingAdapter:
    """Build the adapter named by ``embeddings.provider``.

    Returns:
        A ready-to-use adapter. No model is loaded and no connection opened until it is
        first used.

    Raises:
        ConfigError: If the configured provider cannot be resolved.
    """
    return resolve_adapter_class(settings.embeddings.provider)(settings)
