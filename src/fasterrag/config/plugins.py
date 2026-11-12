"""Names third parties register under an entry-point group, read without loading them.

The adapter factories resolve a provider name to a class; configuration has to decide,
earlier and with less, whether a name is selectable at all. Asking a factory would make
``config/`` import ``adapters/``, and every adapter imports the schema, so the dependency
would run in both directions. This module answers the one question validation needs — is
this name installed — from distribution metadata alone.

**Nothing is imported and nothing is executed.** ``importlib.metadata`` reads each installed
distribution's ``entry_points.txt``; the object an entry point refers to is loaded only by
the factory that goes on to build it. A plugin whose module is broken therefore still
validates here and fails where it is constructed, with the factory's own message.

**Nothing is cached.** One scan took between 8.5 and 14.4 ms across 23 runs over 170
installed distributions (2026-11-12, this repository's development environment, loaded
machine), and validation reaches it only for a name that is not built in, so a
configuration naming built-in providers never pays it. A cache would have to be
invalidated whenever ``sys.path`` changes, which is exactly what installing a plugin does.
"""

from __future__ import annotations

from importlib.metadata import entry_points

__all__ = ["registered_names"]


def registered_names(group: str) -> frozenset[str]:
    """Return every name installed distributions register under ``group``."""
    return frozenset(entry.name for entry in entry_points(group=group))
