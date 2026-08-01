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
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_settings import YamlConfigSettingsSource
from yaml import YAMLError

from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError
from fasterrag.observability.logging import get_logger

__all__ = ["DEFAULT_CONFIG_PATH", "DEFAULT_ENV_FILE", "load_settings"]

DEFAULT_CONFIG_PATH: Final = Path("config.yaml")
DEFAULT_ENV_FILE: Final = Path(".env")

_CHUNK_SIZE_WARN_ABOVE: Final = 1024

_logger = get_logger(__name__)


def load_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
) -> Settings:
    """Load and validate configuration, or fail with a ``ConfigError``.

    Args:
        path: Path to ``config.yaml``.
        env_file: Optional ``.env`` file loaded into the environment before the
            presence check. Existing environment variables always win; ``None`` skips
            the file and checks the process environment as-is.

    Returns:
        The validated, immutable settings.

    Raises:
        ConfigError: If the file is missing or unreadable, the YAML is malformed, any
            key violates the schema or a cross-field rule, or a referenced environment
            variable is absent or blank.
    """
    config_path = Path(path)
    raw = _read_yaml(config_path)
    settings = _validate(raw, config_path)
    _require_referenced_env_vars(settings, env_file)
    _warn_about_risky_settings(settings)
    return settings


def _read_yaml(config_path: Path) -> dict[str, Any]:
    """Parse the YAML document, mapping every read failure onto ``ConfigError``."""
    if not config_path.is_file():
        raise ConfigError(
            f"configuration file not found: {config_path}; copy the canonical config.yaml "
            "from the repository root or pass --config"
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
