"""Fail-fast configuration loader.

Reads ``config.yaml`` through the pydantic-settings YAML source, validates the whole
schema, and confirms that every environment variable the configuration references is
actually present. Any violation raises :class:`~fasterrag.errors.ConfigError` naming the
offending key or variable — startup aborts rather than serving a misconfigured process.

Secret values are never read, echoed, or logged here: the loader only ever asserts that
a named variable exists and is non-blank.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_settings import YamlConfigSettingsSource
from yaml import YAMLError, safe_load

from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError
from fasterrag.observability.logging import get_logger

__all__ = ["DEFAULT_CONFIG_PATH", "DEFAULT_ENV_FILE", "apply_overrides", "load_settings"]

DEFAULT_CONFIG_PATH: Final = Path("config.yaml")
DEFAULT_ENV_FILE: Final = Path(".env")

_CHUNK_SIZE_WARN_ABOVE: Final = 1024

_logger = get_logger(__name__)


def load_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    require_env: bool = True,
    overrides: Sequence[str] | None = None,
) -> Settings:
    """Load and validate configuration, or fail with a ``ConfigError``.

    Args:
        path: Path to ``config.yaml``.
        env_file: Optional ``.env`` file loaded into the environment before the
            presence check. Existing environment variables always win; ``None`` skips
            the file and checks the process environment as-is.
        require_env: Whether a referenced-but-absent environment variable is an error.
        overrides: ``dotted.key=value`` strings applied over the file's contents before
            validation, so an override is held to exactly the same schema, range, and
            cross-field rules a file value is.

    Returns:
        The validated, immutable settings.

    Raises:
        ConfigError: If the file is missing or unreadable, the YAML is malformed, any
            key violates the schema or a cross-field rule, an override is malformed or
            names an unknown key, or a referenced environment variable is absent or blank.
    """
    config_path = Path(path)
    raw = _read_yaml(config_path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    settings = _validate(raw, config_path)
    _reject_unenforced_settings(settings)

    # CRITICAL: `require_env` defaults to True and every execution path must leave it
    # there. It exists for read-only inspection — `config show` is most useful on exactly
    # the broken installation whose missing variables would otherwise stop it from
    # printing anything. Passing False anywhere that goes on to serve, ingest, or query
    # would trade a startup failure naming the variable for a runtime one that does not.
    if require_env:
        _require_referenced_env_vars(settings, env_file)

    _warn_about_risky_settings(settings)
    return settings


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return ``raw`` with each ``dotted.key=value`` override merged in.

    Values are parsed as YAML scalars, so ``true``, ``512``, ``0.75``, and ``null`` arrive
    as the types the schema expects rather than as strings. A value that is genuinely a
    string still works, because YAML reads a bare word as one.

    Overrides are merged *before* validation rather than set on a built ``Settings``.
    Settings are immutable and cross-field rules run at construction, so patching an
    instance afterwards would either fail or skip the very rules that catch a bad
    combination — ``--set retrieval.top_k=200`` must still be checked against
    ``rerank_top_n``.

    Args:
        raw: The configuration mapping read from the file.
        overrides: ``dotted.key=value`` strings, applied in order.

    Returns:
        A new mapping with the overrides applied.

    Raises:
        ConfigError: If an override has no ``=``, has an empty key, or traverses through
            a key whose existing value is a scalar.
    """
    merged = deepcopy(raw)

    for override in overrides:
        key, separator, value = override.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ConfigError(
                f"--set expects dotted.key=value but got {override!r}; "
                "for example --set chunking.chunk_size=512"
            )

        try:
            parsed = safe_load(value)
        except YAMLError as exc:
            raise ConfigError(f"--set {key} has an unparseable value: {exc}") from exc

        target = merged
        parts = key.split(".")
        for part in parts[:-1]:
            existing = target.get(part)
            if existing is None:
                existing = {}
                target[part] = existing
            elif not isinstance(existing, dict):
                raise ConfigError(
                    f"--set {key} cannot descend into {part!r}, which is a value rather "
                    "than a section"
                )
            target = existing
        target[parts[-1]] = parsed

    return merged


def _read_yaml(config_path: Path) -> dict[str, Any]:
    """Parse the YAML document, mapping every read failure onto ``ConfigError``."""
    if not config_path.is_file():
        # CRITICAL: this fix string must name a command, not a repository. It is the first
        # error a `pip install` user ever sees, and telling them to copy a file from a
        # checkout they do not have leaves them with nowhere to go.
        raise ConfigError(
            f"configuration file not found: {config_path}; run 'fasterrag config init' to "
            "write the canonical one here, or pass --config to point at an existing file"
        )

    try:
        raw = YamlConfigSettingsSource(
            Settings,
            yaml_file=config_path,
            # CRITICAL: utf-8-sig, not utf-8. Windows editors and PowerShell write a UTF-8
            # BOM by default, and under plain utf-8 it becomes part of the first key — so a
            # file whose first line reads `app:` is rejected for an unknown key named
            # `﻿app`, which is invisible in every editor that wrote it.
            yaml_file_encoding="utf-8-sig",
        )()
    except YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(
            f"{config_path} must contain a mapping of configuration sections"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"{config_path} could not be read: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping of configuration sections")
    return raw


def _validate(raw: dict[str, Any], config_path: Path) -> Settings:
    """Validate the parsed document, reporting every offending key at once."""
    try:
        return Settings(**raw)
    except ValidationError as exc:
        violations = _format_violations(exc)

    # CRITICAL: raised outside the except block so the ValidationError is neither chained
    # nor kept as context. Its string form echoes offending input values, which would leak
    # a credential mistakenly pasted into config.yaml into logs and tracebacks.
    raise ConfigError(f"{config_path} is invalid:\n{violations}")


def _format_violations(exc: ValidationError) -> str:
    """Render validation failures as ``key: reason`` lines.

    Only the location and the reason are included. Offending input values are
    deliberately omitted so that a credential mistakenly pasted into ``config.yaml``
    cannot reach the logs.
    """
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def _reject_unenforced_settings(settings: Settings) -> None:
    """Refuse to start under a security or budget setting that nothing enforces.

    These keys validate, and the schema keeps them because ``docs/config-reference.md``
    specifies them — but no code path consumes any of them yet. Accepting them silently is
    the worst of the three options: an operator who sets ``security.auth: true`` reads that
    as authentication being on, and gets an open API that reports itself configured. A
    startup failure naming the missing slice is the only outcome that cannot be mistaken
    for protection.

    Mirrors the ``cache.backend: redis`` rejection in ``core/cache``, which fails the same
    way for the same reason.

    Raises:
        ConfigError: If any accepted-but-unenforced setting is enabled.
    """
    enabled: list[str] = []

    # TODO: each of these is removed from this list by the slice named beside it.
    # security.auth and security.multi_tenancy both left this list — auth with TASK-0046 and
    # tenancy with TASK-0179. Both are enforced by AuthMiddleware, which refuses to start
    # without a usable key and refuses any request whose tenant does not match its key.
    if settings.cost.per_query_token_budget:
        enabled.append("  - cost.per_query_token_budget (the cost governor is not built)")
    if settings.cost.per_tenant_token_budget:
        enabled.append("  - cost.per_tenant_token_budget (the cost governor is not built)")

    if enabled:
        listed = "\n".join(enabled)
        raise ConfigError(
            "these settings are accepted by the schema but enforced by nothing yet:\n"
            f"{listed}\n"
            "leave them at their defaults until the slice that implements them ships — "
            "starting with one enabled would report a protection the system does not have"
        )


def _require_referenced_env_vars(settings: Settings, env_file: str | Path | None) -> None:
    """Enforce cross-field rule 9: every referenced environment variable must be set."""
    if env_file is not None:
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(env_path, override=False)

    missing: list[str] = []
    for name, config_key in sorted(settings.referenced_env_vars().items()):
        value = os.environ.get(name)
        if value is None or not value.strip():
            missing.append(f"  - {name} (referenced by {config_key})")

    if missing:
        listed = "\n".join(missing)
        raise ConfigError(
            "required environment variables are missing or blank:\n"
            f"{listed}\n"
            "set them in .env (never in config.yaml) or unset the config keys that "
            "reference them"
        )


def _warn_about_risky_settings(settings: Settings) -> None:
    """Log the documented warnings for settings that are valid but risky."""
    if settings.chunking.chunk_size > _CHUNK_SIZE_WARN_ABOVE:
        _logger.warning(
            "chunking.chunk_size is above the practical working range; retrieval quality "
            "degrades toward the documented context cliff near 2500 tokens",
            extra={"chunk_size": settings.chunking.chunk_size},
        )

    if settings.index.reindex.strategy == "in_place":
        _logger.warning(
            "index.reindex.strategy is 'in_place': this is a development-only path with "
            "no zero-downtime guarantee; use 'blue_green' in production",
        )
