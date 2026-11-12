"""Which provider names configuration may select, decided without loading a provider.

The adapter factories resolve a provider name to a class; configuration has to decide,
earlier and with less, whether a name is selectable at all. Asking a factory would make
``config/`` import ``adapters/``, and every adapter imports the schema, so the dependency
would run in both directions. This module answers the one question validation needs — is
this name built in or installed — from plain strings and distribution metadata alone.

**The built-in names are repeated here, not imported.** They are the keys of each factory's
``_BUILTIN_ADAPTERS``; ``tests/unit/config/test_provider_names.py`` fails when the two
disagree, which is the whole of what keeps a copy honest. Until TASK-0439 the schema held a
third copy as closed ``Literal`` types, so a plugin the factories could resolve was refused
at load before any factory ran.

**A built-in name is accepted before the environment is looked at**, for the reason the
factories give: an installed package must not be able to take over ``qdrant``. It also
means a configuration naming only built-in providers never scans anything.

**Nothing is imported and nothing is executed.** ``importlib.metadata`` reads each installed
distribution's ``entry_points.txt``; the object an entry point refers to is loaded only by
the factory that goes on to build it. A plugin whose module is broken therefore still
validates here and fails where it is constructed, with the factory's own message.

**Nothing is cached.** One scan took between 8.5 and 14.4 ms across 23 runs over 170
installed distributions (2026-11-12, this repository's development environment, loaded
machine), and validation reaches it only for a name that is not built in. A cache would
have to be invalidated whenever ``sys.path`` changes, which is exactly what installing a
plugin does.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Final

__all__ = [
    "BUILTIN_EMBEDDING_PROVIDERS",
    "BUILTIN_LLM_PROVIDERS",
    "BUILTIN_VECTOR_DB_PROVIDERS",
    "EMBEDDINGS_GROUP",
    "LLM_GROUP",
    "VECTOR_DB_GROUP",
    "registered_names",
    "require_selectable",
]

VECTOR_DB_GROUP: Final = "fasterrag.vectordb"
EMBEDDINGS_GROUP: Final = "fasterrag.embeddings"
LLM_GROUP: Final = "fasterrag.llm"

BUILTIN_VECTOR_DB_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"qdrant", "pgvector", "chroma", "milvus", "weaviate"}
)
BUILTIN_EMBEDDING_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"huggingface", "openai", "cohere", "ollama"}
)
BUILTIN_LLM_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"openai", "openai_compatible", "anthropic", "cohere", "ollama"}
)


def registered_names(group: str) -> frozenset[str]:
    """Return every name installed distributions register under ``group``."""
    return frozenset(entry.name for entry in entry_points(group=group))


def require_selectable(value: str, *, builtin: frozenset[str], group: str) -> str:
    """Return ``value`` when it names a built-in or an installed plugin, else refuse it."""
    if value in builtin:
        return value

    installed = registered_names(group)
    if value in installed:
        return value

    raise ValueError(
        f"{value!r} is not an installed provider; {_alternatives(builtin, installed, group)}. "
        f"Install the package that registers {value!r} under that group, or use one of "
        "the names listed"
    )


def _alternatives(builtin: frozenset[str], installed: frozenset[str], group: str) -> str:
    """Name every provider that would have been accepted, built-in and plugin apart."""
    plugins = ", ".join(sorted(installed - builtin)) or "none"
    return (
        f"built-in providers: {', '.join(sorted(builtin))}; plugins installed under the "
        f"{group!r} entry-point group: {plugins}"
    )
