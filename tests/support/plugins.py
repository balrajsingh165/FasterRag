"""Install a fake third-party distribution, so entry points are discovered for real.

A plugin test that replaces a factory's lookup function proves the factory calls its lookup
function. It cannot prove that a package a user installs is found, and it cannot reach a
second process at all: a pool worker started with ``spawn`` re-imports every module and
never sees a patch applied in the parent.

So nothing inside fasterRag is patched here. The helper writes the two metadata files a
real installation leaves behind — ``METADATA`` and ``entry_points.txt`` inside a
``.dist-info`` directory — and puts their parent on ``sys.path``. ``importlib.metadata``
then finds the distribution exactly as it finds one that ``pip`` installed, in this process
and in any child that inherits the path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

__all__ = ["install_distribution"]


def install_distribution(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_points: Mapping[str, Mapping[str, str]],
    *,
    name: str = "acme-fasterrag",
) -> Path:
    """Register ``group -> {entry name -> module:attribute}`` as an installed distribution."""
    site = root / f"site-{name}"
    metadata = site / f"{name.replace('-', '_')}-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )

    lines: list[str] = []
    for group, entries in entry_points.items():
        lines.append(f"[{group}]")
        lines.extend(f"{entry} = {target}" for entry, target in entries.items())
        lines.append("")
    (metadata / "entry_points.txt").write_text("\n".join(lines), encoding="utf-8")

    monkeypatch.syspath_prepend(str(site))
    return site
