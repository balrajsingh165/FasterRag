"""The single way a CLI command turns parsed arguments into validated settings.

Every command loads configuration the same way, which is what makes ``--set`` work
everywhere rather than only on the commands that remembered to thread it through. A command
calling ``load_settings(args.config)`` directly would silently ignore the flag, and an
override that is accepted on the command line and then does nothing is worse than one that
is rejected.
"""

from __future__ import annotations

import argparse

from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings

__all__ = ["overrides_from", "settings_from"]


def overrides_from(args: argparse.Namespace) -> list[str]:
    """Return the ``--set`` overrides, empty when the flag was never given."""
    overrides = getattr(args, "overrides", None)
    return list(overrides) if overrides else []


def settings_from(args: argparse.Namespace, *, require_env: bool = True) -> Settings:
    """Load configuration named by ``--config``, with any ``--set`` overrides applied.

    Args:
        args: Parsed arguments carrying ``config`` and optionally ``overrides``.
        require_env: Whether a referenced-but-absent environment variable is an error.
            Left true by everything except read-only inspection.

    Returns:
        The validated settings.

    Raises:
        ConfigError: If the file or any override is invalid.
    """
    return load_settings(
        args.config,
        require_env=require_env,
        overrides=overrides_from(args),
    )
