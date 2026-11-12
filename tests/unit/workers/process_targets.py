"""Parse targets a spawned worker process can import.

A process pool pickles its task by qualified name, and on Windows and macOS the child is a
fresh interpreter that re-imports the module holding it. A closure, a lambda or a
``monkeypatch`` applied in the parent therefore does not exist in the child, so every
stand-in for ``parse_and_chunk`` that has to run in a real worker lives here, at module
level. The module imports nothing a child would not load anyway, which keeps worker start-up
at the cost of ``fasterrag`` itself.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.workers.cpu_pool import parse_and_chunk
from fasterrag.workers.queues import DocumentTask, ParseOutcome

HANG: Final = "hang"
"""File stem of a document whose parse never returns within any test's patience."""

PAUSE: Final = "pause"
"""File stem of a document that parses after :data:`PAUSE_SECONDS`."""

SLOW: Final = "slow"
"""File stem of a document that parses after :data:`SLOW_SECONDS`."""

PAUSE_SECONDS: Final = 2.0
SLOW_SECONDS: Final = 3.0

_DELAYS: Final[dict[str, float]] = {HANG: 3600.0, PAUSE: PAUSE_SECONDS, SLOW: SLOW_SECONDS}


def die(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Kill the worker process outright, the way the kernel's OOM killer would."""
    os._exit(1)


def parse_after_a_delay(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Parse normally, after sleeping for as long as the source's file stem asks."""
    time.sleep(_DELAYS.get(Path(task.source).stem, 0.0))
    return parse_and_chunk(task, settings)
