"""Selects the concrete LLM adapter from configuration.

Mirrors the vector database and embedding factories: ``llm.provider`` is the whole selection
surface, third parties register providers through the ``fasterrag.llm`` entry point, and
built-in names always win so an installed package cannot silently take over a deployment's
generation.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Final

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError

__all__ = [
    "ENTRY_POINT_GROUP",
    "available_providers",
    "create_llm_adapter",
    "resolve_adapter_class",
]

ENTRY_POINT_GROUP: Final = "fasterrag.llm"

_BUILTIN_ADAPTERS: Final[dict[str, str]] = {
    "openai": "fasterrag.adapters.llm.openai:OpenAIGenerator",
    "openai_compatible": "fasterrag.adapters.llm.openai:OpenAICompatibleGenerator",
    "anthropic": "fasterrag.adapters.llm.anthropic:AnthropicGenerator",
    "cohere": "fasterrag.adapters.llm.cohere:CohereGenerator",
    "ollama": "fasterrag.adapters.llm.ollama:OllamaGenerator",
}


def _plugin_entry_points() -> dict[str, EntryPoint]:
    """Return third-party generation providers registered under the entry-point group."""
    return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}


def available_providers() -> dict[str, str]:
    """Return every selectable provider name mapped to where it came from."""
    providers = dict.fromkeys(_BUILTIN_ADAPTERS, "built-in")
    for name, entry in _plugin_entry_points().items():
        if name not in providers:
            providers[name] = f"plugin ({entry.value})"
    return providers


def _require_adapter_class(loaded: object, source: str) -> type[LLMAdapter]:
    """Confirm a resolved object really implements the generation contract."""
    if not (isinstance(loaded, type) and issubclass(loaded, LLMAdapter)):
        raise ConfigError(
            f"{source} is not an LLMAdapter subclass; a registered provider must implement "
            "the adapter contract"
        )
    return loaded


def resolve_adapter_class(provider: str) -> type[LLMAdapter]:
    """Return the adapter class registered for ``provider``.

    Raises:
        ConfigError: If the provider is unknown or resolves to something that is not an
            ``LLMAdapter``.
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
    raise ConfigError(f"llm.provider {provider!r} is not registered; available providers: {known}")


def create_llm_adapter(settings: Settings) -> LLMAdapter:
    """Build the adapter named by ``llm.provider``.

    Returns:
        A ready-to-use adapter. No connection is opened until it is first used.

    Raises:
        ConfigError: If the configured provider cannot be resolved.
    """
    return resolve_adapter_class(settings.llm.provider)(settings)
