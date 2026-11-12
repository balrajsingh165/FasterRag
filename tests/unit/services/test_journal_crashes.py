"""A job record survives a crash between any two file operations (TASK-0437).

``_write_atomically`` used to rename the live record to ``.prev`` and only then write the new
one, so between those two steps there was no live record: a reader reported ``NOT_FOUND`` for
a job that existed, and a crash in that window left the job findable only by its fallback.
These tests drive the shipped writer through every file operation it makes, crash it or read
the journal at each one, and assert a reader only ever sees a whole record — the one before
the write or the one it was writing.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services import journal as journal_module
from fasterrag.services.journal import Checkpoint, JobRecord, Journal

SOURCES = [{"type": "path", "value": "/data/docs"}]
CRASH = "crash"
TORN = "torn"
FULL = "full"
MOST_OPERATIONS_IN_ONE_WRITE = 12


class SimulatedCrash(BaseException):
    """The process stopped here: no handler runs, as none would after a real kill."""


class FileOperations:
    """Counts the journal's file operations, and fails or looks at the journal at one of them."""

    def __init__(
        self,
        *,
        fail_at: int | None = None,
        kind: str = CRASH,
        look_at: frozenset[int] = frozenset(),
        look: Callable[[], None] | None = None,
    ) -> None:
        """Arm a failure of ``kind`` at operation ``fail_at`` and a look at each of ``look_at``."""
        self.fail_at = fail_at
        self.kind = kind
        self.look_at = look_at
        self.look = look
        self.performed: list[str] = []

    def before(self, operation: str) -> bool:
        """Run what is armed for the next operation; return whether to tear it."""
        index = len(self.performed)
        self.performed.append(operation)
        if self.look is not None and index in self.look_at:
            self.look()
        if index != self.fail_at:
            return False
        if self.kind == FULL:
            raise OSError(errno.ENOSPC, "No space left on device")
        if self.kind == TORN and operation == "write":
            return True
        raise SimulatedCrash(f"{operation} #{index}")


class _OsSpy:
    """The ``os`` module as the journal sees it, with every mutating call counted first."""

    def __init__(self, operations: FileOperations) -> None:
        """Wrap the real module."""
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        """Pass everything not intercepted straight through."""
        return getattr(os, name)

    def replace(self, source: Any, target: Any) -> None:
        """Count a rename, then make it."""
        self._operations.before("replace")
        os.replace(source, target)

    def fsync(self, descriptor: int) -> None:
        """Count a flush to disk, then make it."""
        self._operations.before("fsync")
        os.fsync(descriptor)


@contextmanager
def instrumented(operations: FileOperations) -> Iterator[None]:
    """Route the journal's file operations through ``operations`` for the duration."""
    real_write = Path.write_text

    def write_text(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if operations.before("write"):
            real_write(path, data[: len(data) // 2], *args, **kwargs)
            raise SimulatedCrash("torn write")
        return real_write(path, data, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(journal_module, "os", _OsSpy(operations))
        patch.setattr(Path, "write_text", write_text)
        yield


def at(record: JobRecord, index: int) -> JobRecord:
    """Return ``record`` checkpointed at ``index``."""
    return replace(record, checkpoint=Checkpoint(index, f"2026-11-12T00:00:{index % 60:02d}+00:00"))


def seen_index(journal: Journal, job: str) -> int | None:
    """Return the checkpoint index a reader sees for ``job`` right now."""
    record = journal.load_job_blocking(job)
    return record.checkpoint.last_document_index if record.checkpoint else None


def operations_in_one_rewrite(tmp_path: Path) -> int:
    """Count the file operations one rewrite of an existing record makes."""
    journal = Journal(tmp_path / "count")
    record = journal.create_job_blocking("default", SOURCES)
    journal.save_job_blocking(at(record, 1))
    counter = FileOperations()
    with instrumented(counter):
        journal.save_job_blocking(at(record, 2))
    return len(counter.performed)


def test_a_rewrite_is_made_of_the_operations_this_file_injects_failures_into(
    tmp_path: Path,
) -> None:
    """If the writer stopped using these calls, the tests below would inject nothing."""
    count = operations_in_one_rewrite(tmp_path)

    assert 3 <= count <= MOST_OPERATIONS_IN_ONE_WRITE


@pytest.mark.parametrize("kind", [CRASH, TORN])
def test_a_crash_at_any_operation_of_a_rewrite_leaves_the_old_or_the_new_record(
    tmp_path: Path, kind: str
) -> None:
    """A crash between the rename-away and the write left no live record at all."""
    for fail_at in range(operations_in_one_rewrite(tmp_path)):
        journal = Journal(tmp_path / f"{kind}-{fail_at}")
        record = journal.create_job_blocking("default", SOURCES)
        journal.save_job_blocking(at(record, 1))

        with (
            instrumented(FileOperations(fail_at=fail_at, kind=kind)),
            pytest.raises(SimulatedCrash),
        ):
            journal.save_job_blocking(at(record, 2))

        assert seen_index(journal, record.job_id) in {1, 2}, f"crashed at operation {fail_at}"


def test_a_crash_at_any_operation_of_the_first_write_leaves_no_job_or_a_whole_one(
    tmp_path: Path,
) -> None:
    probe = Journal(tmp_path / "probe")
    counter = FileOperations()
    with instrumented(counter):
        probe.create_job_blocking("default", SOURCES)

    for fail_at in range(len(counter.performed)):
        journal = Journal(tmp_path / f"first-{fail_at}")
        record = JobRecord(job_id=f"job_first_{fail_at}", collection="default")
        with instrumented(FileOperations(fail_at=fail_at)), pytest.raises(SimulatedCrash):
            journal.save_job_blocking(record)

        if journal._job_path(record.job_id).exists():
            assert journal.load_job_blocking(record.job_id) == record
        else:
            with pytest.raises(IngestionError) as caught:
                journal.load_job_blocking(record.job_id)
            assert caught.value.code is ErrorCode.NOT_FOUND, f"crashed at operation {fail_at}"


def test_a_reader_at_every_operation_of_a_rewrite_sees_the_old_or_the_new_record(
    tmp_path: Path,
) -> None:
    """The window a concurrent reader fell into: between two renames there was no record."""
    journal = Journal(tmp_path / "journal")
    record = journal.create_job_blocking("default", SOURCES)
    journal.save_job_blocking(at(record, 1))
    live = journal._job_path(record.job_id)
    seen: list[int | None] = []
    present: list[bool] = []

    def look() -> None:
        seen.append(seen_index(journal, record.job_id))
        present.append(live.exists())

    count = operations_in_one_rewrite(tmp_path)
    with instrumented(FileOperations(look_at=frozenset(range(count)), look=look)):
        journal.save_job_blocking(at(record, 2))

    assert len(seen) == count
    assert set(seen) <= {1, 2}
    assert all(present), "the live record was moved away before its replacement existed"
    assert seen_index(journal, record.job_id) == 2


def test_a_full_disk_mid_rewrite_keeps_the_record_readable(tmp_path: Path) -> None:
    for fail_at in range(operations_in_one_rewrite(tmp_path)):
        journal = Journal(tmp_path / f"full-{fail_at}")
        record = journal.create_job_blocking("default", SOURCES)
        journal.save_job_blocking(at(record, 1))

        with instrumented(FileOperations(fail_at=fail_at, kind=FULL)):
            try:
                journal.save_job_blocking(at(record, 2))
            except IngestionError:
                expected = {1, 2}
            else:
                expected = {2}

        assert seen_index(journal, record.job_id) in expected, f"failed at operation {fail_at}"
        assert list(journal.root.rglob("*.tmp")) == [], f"failed at operation {fail_at}"


@settings(max_examples=60, stateful_step_count=12, deadline=None)
class JournalUnderCrashes(RuleBasedStateMachine):
    """Interleaves rewrites, crashes and full disks at any operation, and reads in between."""

    def __init__(self) -> None:
        """Start from one job with no checkpoint."""
        super().__init__()
        self.directory = Path(tempfile.mkdtemp(prefix="journal-crashes-"))
        self.journal = Journal(self.directory / "journal")
        self.job = self.journal.create_job_blocking("default", SOURCES)
        self.version = 0
        self.acceptable: set[int | None] = {None}

    @rule(
        fail_at=st.none() | st.integers(0, MOST_OPERATIONS_IN_ONE_WRITE),
        kind=st.sampled_from([CRASH, TORN, FULL]),
        look_at=st.frozensets(st.integers(0, MOST_OPERATIONS_IN_ONE_WRITE), max_size=4),
    )
    def rewrite(self, fail_at: int | None, kind: str, look_at: frozenset[int]) -> None:
        """Rewrite the record, failing at ``fail_at`` and reading at each of ``look_at``."""
        self.version += 1
        during = self.acceptable | {self.version}

        def look() -> None:
            assert seen_index(self.journal, self.job.job_id) in during

        operations = FileOperations(fail_at=fail_at, kind=kind, look_at=look_at, look=look)
        try:
            with instrumented(operations):
                self.journal.save_job_blocking(at(self.job, self.version))
        except (SimulatedCrash, IngestionError):
            self.acceptable = during
            return
        self.acceptable = {self.version}

    @rule()
    def restart(self) -> None:
        """A new process opens the same directory, debris from earlier crashes and all."""
        self.journal = Journal(self.journal.root)

    @invariant()
    def a_reader_sees_a_whole_record_it_was_promised(self) -> None:
        """Never missing, never torn: the last write that finished or one that crashed."""
        assert seen_index(self.journal, self.job.job_id) in self.acceptable

    def teardown(self) -> None:
        """Remove the journal this run wrote."""
        shutil.rmtree(self.directory, ignore_errors=True)


TestJournalUnderCrashes = JournalUnderCrashes.TestCase


def test_two_writers_and_a_reader_never_see_a_missing_record(tmp_path: Path) -> None:
    """Two writers shared one fixed temporary name, and a reader could fall between renames."""
    journal = Journal(tmp_path / "journal")
    record = journal.create_job_blocking("default", SOURCES)
    journal.save_job_blocking(at(record, 0))
    failures: list[BaseException] = []
    stop = threading.Event()

    def write(offset: int) -> None:
        try:
            for index in range(offset, offset + 40):
                journal.save_job_blocking(at(record, index))
        except BaseException as exc:
            failures.append(exc)
            raise

    def read() -> None:
        try:
            while not stop.is_set():
                assert seen_index(journal, record.job_id) is not None
        except BaseException as exc:
            failures.append(exc)
            raise

    reader = threading.Thread(target=read)
    writers = [threading.Thread(target=write, args=(offset,)) for offset in (100, 200)]
    reader.start()
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
    stop.set()
    reader.join()

    assert failures == []
    assert list(journal.root.rglob("*.tmp")) == []
