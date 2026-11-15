"""Loading a plugin class from an entry point, and the API version check every one passes.

Each extension interface states its version as ``MAJOR.MINOR`` — ``ADAPTER_API_VERSION`` in
the three adapter base modules, ``CHUNKER_API_VERSION`` and ``PARSER_API_VERSION`` in
``core.chunking`` and ``core.parsing`` — and a plugin class declares the version it was
written against in the class attribute ``adapter_api_version``. One attribute name serves
every interface, so a plugin author learns one rule. The declaration has to be a literal:
importing the constant would always read the installed value and prove nothing. A plugin
built for another major version, for a newer minor version (it may call members this
installation lacks), with no declaration or with a malformed one is refused when its entry
point is loaded, with a ``ConfigError`` naming both versions, rather than instantiated and
left to fail at its first call with an ``AttributeError`` that names neither. Built-ins are
not asked: they ship with the interface they implement.

The check lives in ``core`` because chunkers and parsers are loaded here, and ``core`` may
not import ``adapters``; ``adapters.plugin_api`` re-exports it for the adapter factories.

**Loaded classes are cached per process** (:func:`load_plugin_class`). A chunker or parser
is resolved for every document a parse worker handles, and one entry-point scan costs
milliseconds over a large environment; a package installed while a process runs is seen
after a restart, which is when its code could be imported anyway.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Final

from fasterrag.errors import ConfigError

__all__ = [
    "API_VERSION_ATTRIBUTE",
    "load_plugin_class",
    "require_compatible_api",
    "reset_loaded_plugins",
]

API_VERSION_ATTRIBUTE: Final = "adapter_api_version"

_loaded: dict[tuple[str, str], type] = {}


def _parse_api_version(version: str, source: str) -> tuple[int, int]:
    """Split a ``MAJOR.MINOR`` API version, refusing any other shape."""
    major, separator, minor = version.partition(".")
    if not (separator and major.isdigit() and minor.isdigit()):
        raise ConfigError(
            f"{source} declares adapter_api_version {version!r}, which is not of the form "
            "MAJOR.MINOR (for example '1.0')"
        )
    return int(major), int(minor)


def require_compatible_api[PluginClass: type](
    plugin: PluginClass, *, host: str, source: str, interface: str = "adapter"
) -> PluginClass:
    """Return ``plugin`` when the API it declares is one ``host`` provides, else refuse it.

    Same major, and a minor no newer than the host's. ``source`` names the plugin and
    ``interface`` the API it implements (``adapter``, ``chunker``, ``parser``).
    """
    declared = getattr(plugin, API_VERSION_ATTRIBUTE, None)
    if not isinstance(declared, str):
        suite = ", once the plugin passes the adapter contract suite against it"
        raise ConfigError(
            f"{source} declares no adapter_api_version string; set the class attribute "
            f"adapter_api_version = {host!r}, the {interface} API this fasterrag provides"
            f"{suite if interface == 'adapter' else ''}"
        )

    plugin_major, plugin_minor = _parse_api_version(declared, source)
    host_major, host_minor = _parse_api_version(host, "this fasterrag")
    if plugin_major != host_major:
        raise ConfigError(
            f"{source} was built for {interface} API {declared}, but this fasterrag provides "
            f"{interface} API {host}; install a release of the plugin built for {host_major}.x, "
            f"or a fasterrag release providing {interface} API {plugin_major}.x"
        )
    if plugin_minor > host_minor:
        raise ConfigError(
            f"{source} was built for {interface} API {declared}, newer than the {interface} "
            f"API {host} this fasterrag provides, and may call members it lacks; upgrade "
            "fasterrag"
        )
    return plugin


def load_plugin_class(
    group: str, name: str, *, host: str, interface: str, members: tuple[str, ...]
) -> type:
    """Return the class ``name`` is registered as under ``group``, checked and cached.

    Refused with a ``ConfigError`` when nothing registers it, it fails to import, it is not
    a class defining every callable in ``members``, or its declared API is incompatible.
    """
    cached = _loaded.get((group, name))
    if cached is not None:
        return cached

    entry = next((found for found in entry_points(group=group) if found.name == name), None)
    if entry is None:
        raise ConfigError(
            f"no installed package registers {name!r} under the {group!r} entry-point group; "
            "install the package that provides it, or change the configuration naming it"
        )

    source = f"{group} entry point {name!r}"
    try:
        loaded: object = entry.load()
    except Exception as exc:
        raise ConfigError(
            f"{source} ({entry.value}) could not be imported: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(loaded, type) or not all(
        callable(getattr(loaded, member, None)) for member in members
    ):
        raise ConfigError(
            f"{source} ({entry.value}) is not a {interface} class; a {interface} plugin is a "
            f"class defining {', '.join(members)}"
        )

    checked = require_compatible_api(loaded, host=host, source=source, interface=interface)
    _loaded[(group, name)] = checked
    return checked


def reset_loaded_plugins() -> None:
    """Forget every plugin class this process loaded, so the next use scans again."""
    _loaded.clear()
