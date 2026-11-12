"""What every example shares: where the sample documents are and which config to load."""

from __future__ import annotations

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def config_path() -> Path:
    """Return the config named on the command line, or ``config.yaml`` here; stop if absent."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Create one first: fasterrag config init --profile local"
        )
    return path


def require(condition: object, message: str) -> None:
    """Exit non-zero with ``message`` unless ``condition`` holds, so CI notices a regression."""
    if not condition:
        raise SystemExit(f"example failed: {message}")
