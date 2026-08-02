"""Access to the canonical ``config.yaml`` that ships inside the package.

A repository checkout has ``config.yaml`` at its root; a ``pip install`` has no repository.
Both need the same starting file, and they need it to be the *same* file — a template that
drifts from the documented canonical config is worse than no template, because it teaches
keys that the reference no longer describes.

So the packaged copy is not a copy. ``pyproject.toml`` force-includes the repository's
``config.yaml`` into the wheel at ``fasterrag/data/config.yaml``; this module reads whichever
of the two is present, preferring the packaged one so an installed package never depends on
the current working directory.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Final

from fasterrag.errors import ConfigError

__all__ = [
    "CONFIG_TEMPLATE_RESOURCE",
    "ENV_TEMPLATE_RESOURCE",
    "canonical_config_text",
    "env_template_text",
]

CONFIG_TEMPLATE_RESOURCE: Final = "data/config.yaml"
ENV_TEMPLATE_RESOURCE: Final = "data/.env.example"

# CRITICAL: relative to this file, `../../..` is the repository root. Used only when the
# packaged resource is absent, which is the editable-install and running-from-a-checkout
# case; a built wheel always has the resource and never reaches this.
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]


def _template_text(resource: str, checkout_name: str) -> str:
    """Return a packaged template, falling back to the repository copy.

    Raises:
        ConfigError: If neither can be read, which means the distribution was built without
            its data files. Reported as a packaging fault rather than a user error, because
            it is one.
    """
    package = resources.files("fasterrag").joinpath(resource)
    if package.is_file():
        return package.read_text(encoding="utf-8")

    checkout = _REPOSITORY_ROOT / checkout_name
    if checkout.is_file():
        return checkout.read_text(encoding="utf-8")

    raise ConfigError(
        f"this installation is missing its packaged {checkout_name} template; the "
        f"distribution was built without fasterrag/{resource}, which is a packaging fault — "
        "reinstall from a complete wheel or run from a repository checkout"
    )


def canonical_config_text() -> str:
    """Return the canonical ``config.yaml`` as text.

    Returns:
        The documented default configuration, byte-identical to the repository's
        ``config.yaml``.
    """
    return _template_text(CONFIG_TEMPLATE_RESOURCE, "config.yaml")


def env_template_text() -> str:
    """Return the ``.env.example`` secrets template as text.

    Returns:
        The documented variable inventory with placeholder values. Every line names a
        variable; none carries a real secret, which is what makes it safe to ship.
    """
    return _template_text(ENV_TEMPLATE_RESOURCE, ".env.example")
