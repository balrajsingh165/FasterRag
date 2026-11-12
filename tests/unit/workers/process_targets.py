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

from fasterrag.config.schema import Settings
from fasterrag.workers.cpu_pool import parse_and_chunk
from fasterrag.workers.queues import DocumentTask, ParseOutcome

HANG_MARKER = "hang"
"""A source whose file name contains this never returns from :func:`hang_on_marker`."""

_FOREVER_SECONDS = 3600.0


def die(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Kill the worker process outright, the way the kernel's OOM killer would."""
    os._exit(1)


def hang_on_marker(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Parse normally, except for a marked source, which blocks for an hour."""
    if HANG_MARKER in os.path.basename(task.source):
        time.sleep(_FOREVER_SECONDS)
    return parse_and_chunk(task, settings)
