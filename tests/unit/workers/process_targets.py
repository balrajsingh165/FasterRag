"""Parse targets a spawned worker process can import.

A process pool pickles its task by qualified name, and on Windows and macOS the child is a
fresh interpreter that re-imports the module holding it. A closure, a lambda or a
``monkeypatch`` applied in the parent therefore does not exist in the child, so every
stand-in for ``parse_and_chunk`` that has to run in a real worker lives here, at module
level. The module imports nothing a child would not load anyway, which keeps worker start-up
at the cost of ``fasterrag`` itself.

The stand-in for the semantic strategy's sentence embedder is installed by a worker
*initializer* for the same reason: the parent's factory is not the child's.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.core.chunking import SentenceEmbedder
from fasterrag.workers.cpu_pool import parse_and_chunk, use_sentence_embedder_factory
from fasterrag.workers.queues import DocumentTask, ParseOutcome

HANG: Final = "hang"
"""File stem of a document whose parse blocks until the test releases it, or for an hour."""

PAUSE: Final = "pause"
"""File stem of a document that parses after :data:`PAUSE_SECONDS`."""

STALL_ONCE: Final = "stall_once"
"""File stem of a document whose first parse blocks like :data:`HANG` and later ones do not."""

PAUSE_SECONDS: Final = 2.0
HANG_SECONDS: Final = 3600.0

_DELAYS: Final[dict[str, float]] = {PAUSE: PAUSE_SECONDS}


def marker(source: Path, kind: str) -> Path:
    """Return the file beside ``source`` that says ``kind`` happened to its parse."""
    return source.with_name(f"{source.name}.{kind}")


def release(source: Path) -> None:
    """Let a blocked parse of ``source`` return, from the test or from a watchdog thread."""
    marker(source, "release").touch()


def _block(source: Path) -> None:
    """Announce that this parse has started, then block until released or for an hour."""
    marker(source, "started").touch()
    deadline = time.monotonic() + HANG_SECONDS
    while not marker(source, "release").exists() and time.monotonic() < deadline:
        time.sleep(0.05)


def die(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Kill the worker process outright, the way the kernel's OOM killer would."""
    os._exit(1)


def parse_after_a_delay(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Parse normally, after blocking or sleeping for as long as the source's file stem asks."""
    source = Path(task.source)
    if source.stem == HANG or (
        source.stem == STALL_ONCE and not marker(source, "started").exists()
    ):
        _block(source)
    time.sleep(_DELAYS.get(source.stem, 0.0))
    return parse_and_chunk(task, settings)


class HashingEmbedder:
    """A sentence embedder that loads nothing: each vector is read off the sentence's hash."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        digests = [hashlib.sha256(text.encode()).digest() for text in texts]
        return [[float(byte) for byte in digest[:8]] for digest in digests]


def record_sentence_embedders(log_dir: str, load_seconds: float) -> None:
    """Worker initializer: build a :class:`HashingEmbedder` instead of loading a model.

    Each build sleeps ``load_seconds``, standing in for a model load, and appends the
    configured model name to a file named for the building process's pid.
    """

    def build(settings: Settings) -> SentenceEmbedder:
        time.sleep(load_seconds)
        with (Path(log_dir) / f"{os.getpid()}.log").open("a", encoding="utf-8") as log:
            log.write(settings.embeddings.model + "\n")
        return HashingEmbedder()

    use_sentence_embedder_factory(build)
