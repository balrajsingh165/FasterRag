"""Put this checkout's ``src`` first on ``sys.path``, so a script judges the tree it lives in.

A script run as ``python scripts/check_openapi_drift.py`` gets ``scripts/`` as ``sys.path[0]``
and finds ``fasterrag`` wherever the interpreter's editable install points. In a linked
worktree sharing the main checkout's virtualenv that is the main checkout, so the gate read
one tree's code against another tree's documents and reported green on a worktree whose new
route was undocumented (TASK-0522). Importing this module puts the ``src`` beside the script
ahead of every other entry. It must be imported before any ``fasterrag`` import; a script that
imports ``fasterrag`` lazily may call :func:`put_this_checkout_first` instead.

``scripts/check_package.py`` deliberately does not use it: that gate exists to import the
built wheel in a clean virtualenv, never a checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = REPOSITORY_ROOT / "src"


def put_this_checkout_first() -> Path:
    """Move this checkout's ``src`` to the front of ``sys.path`` once, and return it."""
    source = str(SOURCE_ROOT)
    while source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    return SOURCE_ROOT


put_this_checkout_first()
