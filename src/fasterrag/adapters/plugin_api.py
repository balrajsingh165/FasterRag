"""The adapter API version check every factory applies to a plugin it loads (TASK-0486).

Each base interface states ``ADAPTER_API_VERSION`` as ``MAJOR.MINOR``, and a plugin class
declares the version it was written against in the class attribute ``adapter_api_version``.
The declaration has to be a literal: importing the constant would always read the installed
value and prove nothing. A plugin built for another major version, for a newer minor version
(it may call members this installation lacks), with no declaration or with a malformed one is
refused when its entry point is loaded, with a ``ConfigError`` naming both versions, rather
than instantiated and left to fail at its first call with an ``AttributeError`` that names
neither. Built-in adapters are not asked: they ship with the interface they implement.

The check itself lives in :mod:`fasterrag.core.plugin_api` since TASK-0513, because chunker
and parser plugins are loaded in ``core`` and ``core`` may not import ``adapters``; this
module re-exports it for the adapter factories. It is not in ``config/plugins.py`` because
that module must import nothing from fasterrag (validation reads entry-point names only).
"""

from __future__ import annotations

from fasterrag.core.plugin_api import API_VERSION_ATTRIBUTE, require_compatible_api

__all__ = ["API_VERSION_ATTRIBUTE", "require_compatible_api"]
