"""Jobs a dead process left ``running`` are settled when the next one starts (TASK-0436).

A process killed outright settles nothing, so its jobs stayed ``running`` for good: nothing
looked for them at startup. These tests prepare a journal directory the way such an exit
leaves it and run the pass a starting process runs.
"""

import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.jobs import (
    INTERRUPTED,
    JobRegistry,
    reap_at_startup,
    reap_interrupted,
)
from fasterrag.services.journal import Checkpoint, DocumentRecord, JobRecord, Journal

SOURCES = [{"type": "path", "value": "corpus/"}]
STALE_SECONDS = 120
LONG_AGO = 3600.0


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal")


def job_file(journal: Journal, job_id: str) -> Path:
    return journal.root / "jobs" / f"{job_id}.json"


def age(path: Path, seconds: float) -> None:
    """Make a file look as if it was last written ``seconds`` ago."""
    written = time.time() - seconds
    os.utime(path, (written, written))


def left_behind(
    journal: Journal, status: str = "running", *, quiet_for: float = LONG_AGO, checkpoint: int = -1
) -> JobRecord:
    """Leave a job record on disk as a process that exited without settling it would."""
    record = journal.create_job_blocking("default", SOURCES)
    record = replace(
        record,
        status=status,
        checkpoint=Checkpoint(checkpoint, "2026-11-12T00:00:00+00:00") if checkpoint >= 0 else None,
    )
    journal.save_job_blocking(record)
    age(job_file(journal, record.job_id), quiet_for)
    return record


async def test_a_stale_running_job_is_settled_and_a_fresh_one_is_left_alone(
    journal: Journal,
) -> None:
    """Nothing looked for a job a dead process left ``running``, so it stayed that way."""
    stale = left_behind(journal, checkpoint=41)
    fresh = left_behind(journal, quiet_for=1)

    report = await reap_interrupted(journal, stale_seconds=STALE_SECONDS)

    settled = await journal.load_job(stale.job_id)
    assert (settled.status, settled.reason_code) == ("failed", INTERRUPTED)
    assert settled.finished_at is not None
    assert settled.checkpoint == stale.checkpoint
    assert (await journal.load_job(fresh.job_id)).status == "running"
    assert report.settled == (stale.job_id,)
    assert report.recent == 1


async def test_a_job_still_recording_documents_is_not_stale(journal: Journal) -> None:
    """A job between two checkpoints writes its document log, not its record."""
    alive = left_behind(journal)
    await journal.record_document(alive.job_id, DocumentRecord("doc_1", "corpus/a.md", "indexed"))

    report = await reap_interrupted(journal, stale_seconds=STALE_SECONDS)

    assert (await journal.load_job(alive.job_id)).status == "running"
    assert report.settled == ()


@pytest.mark.parametrize("status", ["queued", "completed", "partial", "failed"])
async def test_a_job_that_is_not_running_is_never_touched(journal: Journal, status: str) -> None:
    record = left_behind(journal, status)

    report = await reap_interrupted(journal, stale_seconds=STALE_SECONDS)

    untouched = await journal.load_job(record.job_id)
    assert (untouched.status, untouched.reason_code) == (status, None)
    assert report.settled == ()


async def test_a_job_this_process_is_running_is_skipped(journal: Journal) -> None:
    """A slow document must not get a live job settled by its own process."""
    mine = left_behind(journal)

    report = await reap_interrupted(journal, stale_seconds=STALE_SECONDS, skip={mine.job_id})

    assert (await journal.load_job(mine.job_id)).status == "running"
    assert report.settled == ()


async def test_a_journal_that_does_not_exist_yet_reaps_nothing_and_creates_nothing(
    tmp_path: Path,
) -> None:
    journal = Journal(tmp_path / "never-written")

    report = await reap_interrupted(journal, stale_seconds=STALE_SECONDS)

    assert (report.settled, report.recent) == ((), 0)
    assert not journal.root.exists()


async def test_an_unreadable_record_does_not_stop_the_pass(journal: Journal) -> None:
    """One torn file must not leave every other interrupted job ``running``."""
    torn = left_behind(journal)
    job_file(journal, torn.job_id).write_text("{not an envelope", encoding="utf-8")
    age(job_file(journal, torn.job_id), LONG_AGO)
    stale = left_behind(journal)

    report = await reap_interrupted(journal, stale_seconds=STALE_SECONDS)

    assert report.settled == (stale.job_id,)
    assert report.unreadable == 1


async def test_a_job_too_recent_at_startup_is_settled_on_the_second_look(journal: Journal) -> None:
    """A restart seconds after a crash finds the orphan still fresh; one look would miss it."""
    orphan = left_behind(journal, quiet_for=0)

    await asyncio.wait_for(reap_at_startup(journal, JobRegistry(), stale_seconds=1), timeout=10)

    settled = await journal.load_job(orphan.job_id)
    assert (settled.status, settled.reason_code) == ("failed", INTERRUPTED)


async def test_startup_does_not_wait_when_every_record_is_old(journal: Journal) -> None:
    """The second look exists for fresh records; without one it would only delay shutdown."""
    left_behind(journal, "completed")
    started = time.monotonic()

    await reap_at_startup(journal, JobRegistry(), stale_seconds=STALE_SECONDS)

    assert time.monotonic() - started < 5


def test_the_staleness_window_defaults_to_two_minutes() -> None:
    assert Settings().reliability.stale_job_seconds == 120
