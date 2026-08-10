"""Journal writes against a genuinely full filesystem, executed inside a container.

This module is not a test. It is the payload ``tests/chaos/test_real_faults.py`` copies into
a Linux container whose journal directory is a tmpfs mounted with a hard size limit. It
drives the real :class:`~fasterrag.services.journal.Journal` write path against a filesystem
that really returns ``ENOSPC`` and writes one JSON object to stdout describing what the
operating system reported and what state survived it.

Two writes are attempted once the filesystem is full, because they are not the same question:
starting a *new* job has to allocate, while checkpointing a *running* one rewrites a record
that already exists. The report keeps them apart so the test module can say which failed.

Nothing is asserted here. The probe observes; the test module judges. Keeping the two apart
is what makes the recorded behavior a measurement rather than a restatement of what the
author expected to happen.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Final

from fasterrag.errors import FasterRagError
from fasterrag.services.journal import Journal

MOUNT: Final = Path("/journal")
COLLECTION: Final = "docs"
SOURCES: Final = [{"type": "path", "value": "corpus/"}]
BASELINE_INDEX: Final = 41
LATER_INDEX: Final = 99
BLOCK_BYTES: Final = 4096
DETAIL_LIMIT: Final = 200


def describe(exc: BaseException) -> dict[str, Any]:
    """Return the JSON-safe identity of an exception the probe caught."""
    code = getattr(exc, "code", None)
    return {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "errno": getattr(exc, "errno", None),
        "typed": isinstance(exc, FasterRagError),
        "code": getattr(code, "value", None),
        "detail": str(exc)[:DETAIL_LIMIT],
    }


def fill(path: Path) -> int:
    """Consume every free byte of the filesystem holding ``path`` and return the count.

    Unbuffered ``os.write`` rather than a file object: a buffered write surfaces the failure
    when its buffer is flushed, which for the final partial block is at close time, and an
    ``ENOSPC`` raised by ``close`` is far harder to attribute to the write that caused it.
    """
    written = 0
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        for block in (b"\0" * BLOCK_BYTES, b"\0"):
            while True:
                try:
                    written += os.write(descriptor, block)
                except OSError:
                    break
    finally:
        os.close(descriptor)
    return written


def observe(journal: Journal, job: str) -> dict[str, Any]:
    """Return whether a job record is still readable, and the checkpoint it carries."""
    try:
        record = journal.load_job(job)
    except Exception as exc:
        return {"loaded": False, **describe(exc)}
    return {
        "loaded": True,
        "checkpoint": record.checkpoint.last_document_index if record.checkpoint else None,
    }


def main() -> None:
    """Run the probe and write its single JSON report to stdout."""
    report: dict[str, Any] = {}
    journal = Journal(MOUNT / "journal", checkpoint_every=1)

    job = journal.create_job(COLLECTION, SOURCES)
    job = journal.checkpoint(job, BASELINE_INDEX)
    report["baseline"] = observe(journal, job.job_id)

    filler = MOUNT / "filler.bin"
    report["filled_bytes"] = fill(filler)
    report["free_bytes"] = shutil.disk_usage(MOUNT).free

    try:
        journal.create_job(COLLECTION, SOURCES)
    except Exception as exc:
        report["create_while_full"] = describe(exc)
    else:
        report["create_while_full"] = None

    try:
        journal.checkpoint(job, LATER_INDEX)
    except Exception as exc:
        report["checkpoint_while_full"] = describe(exc)
    else:
        report["checkpoint_while_full"] = None

    report["reload_while_full"] = observe(journal, job.job_id)

    filler.unlink()

    try:
        recovered = journal.create_job(COLLECTION, SOURCES)
    except Exception as exc:
        report["create_after_free"] = {"ok": False, **describe(exc)}
    else:
        report["create_after_free"] = {"ok": True, "job_id": recovered.job_id}

    report["reload_after_free"] = observe(journal, job.job_id)

    sys.stdout.write(f"{json.dumps(report)}\n")


if __name__ == "__main__":
    main()
