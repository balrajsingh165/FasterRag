"""The adapter API version check every factory applies to a plugin it loads (TASK-0486).

Each base interface states ``ADAPTER_API_VERSION`` as ``MAJOR.MINOR``, and a plugin class
declares the version it was written against in the class attribute ``adapter_api_version``.
The declaration has to be a literal: importing the constant would always read the installed
value and prove nothing. A plugin built for another major version, for a newer minor version
(it may call members this installation lacks), with no declaration or with a malformed one is
refused when its entry point is loaded, with a ``ConfigError`` naming both versions, rather
than instantiated and left to fail at its first call with an ``AttributeError`` that names
neither. Built-in adapters are not asked: they ship with the interface they implement.

This lives under ``adapters/`` rather than in ``config/plugins.py`` because that module must
import nothing from fasterrag (validation reads entry-point names only).
"""

from __future__ import annotations

from typing import Final

from fasterrag.errors import ConfigError

__all__ = ["API_VERSION_ATTRIBUTE", "require_compatible_api"]

API_VERSION_ATTRIBUTE: Final = "adapter_api_version"


def _parse_api_version(version: str, source: str) -> tuple[int, int]:
    """Split a ``MAJOR.MINOR`` adapter API version, refusing any other shape."""
    major, separator, minor = version.partition(".")
    if not (separator and major.isdigit() and minor.isdigit()):
        raise ConfigError(
            f"{source} declares adapter_api_version {version!r}, which is not of the form "
            "MAJOR.MINOR (for example '1.0')"
        )
    return int(major), int(minor)


def require_compatible_api[AdapterClass: type](
    adapter: AdapterClass, *, host: str, source: str
) -> AdapterClass:
    """Return ``adapter`` when the API it declares is one ``host`` provides, else refuse it.

    Same major, and a minor no newer than the host's. ``source`` names the plugin.
    """
    declared = getattr(adapter, API_VERSION_ATTRIBUTE, None)
    if not isinstance(declared, str):
        raise ConfigError(
            f"{source} declares no adapter_api_version string; set the class attribute "
            f"adapter_api_version = {host!r}, the adapter API this fasterrag provides, once "
            "the plugin passes the adapter contract suite against it"
        )

    plugin_major, plugin_minor = _parse_api_version(declared, source)
    host_major, host_minor = _parse_api_version(host, "this fasterrag")
    if plugin_major != host_major:
        raise ConfigError(
            f"{source} was built for adapter API {declared}, but this fasterrag provides "
            f"adapter API {host}; install a release of the plugin built for {host_major}.x, "
            f"or a fasterrag release providing adapter API {plugin_major}.x"
        )
    if plugin_minor > host_minor:
        raise ConfigError(
            f"{source} was built for adapter API {declared}, newer than the adapter API "
            f"{host} this fasterrag provides, and may call members it lacks; upgrade fasterrag"
        )
    return adapter
