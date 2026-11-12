import asyncio
import errno
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services import journal as journal_module
from fasterrag.services.journal import DocumentRecord, JobRecord, Journal

SOURCES = [{"type": "path", "value": "/data/docs"}]


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal", checkpoint_every=3)


def test_a_created_job_is_persisted_and_reloadable(journal: Journal) -> None:
    created = journal.create_job_blocking("default", SOURCES)

    reloaded = journal.load_job_blocking(created.job_id)

    assert reloaded.job_id == created.job_id
    assert reloaded.collection == "default"
    assert reloaded.sources == SOURCES
    assert reloaded.status == "queued"


def test_an_unknown_job_is_reported_as_not_found(journal: Journal) -> None:
    with pytest.raises(IngestionError, match="unknown ingest job") as caught:
        journal.load_job_blocking("job_missing")

    assert caught.value.code is ErrorCode.NOT_FOUND


def test_a_replayed_idempotency_key_returns_the_original_job(journal: Journal) -> None:
    first = journal.create_job_blocking("default", SOURCES, idempotency_key="key-1")
    second = journal.create_job_blocking("default", SOURCES, idempotency_key="key-1")

    assert second.job_id == first.job_id


def test_different_keys_create_different_jobs(journal: Journal) -> None:
    first = journal.create_job_blocking("default", SOURCES, idempotency_key="key-1")
    second = journal.create_job_blocking("default", SOURCES, idempotency_key="key-2")

    assert first.job_id != second.job_id


def test_checkpoints_are_written_only_at_the_configured_interval(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)

    for index in range(5):
        record = journal.checkpoint_blocking(record, index)

    assert record.checkpoint is not None
    assert record.checkpoint.last_document_index == 2


def test_a_crash_resumes_from_the_document_after_the_checkpoint(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    for index in range(6):
        record = journal.checkpoint_blocking(record, index)

    resumed = journal.load_job_blocking(record.job_id)

    assert journal.resume_index(resumed) == 6


def test_a_job_with_no_checkpoint_resumes_from_the_beginning(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)

    assert journal.resume_index(record) == 0


def test_checkpointing_is_skipped_when_the_journal_is_disabled(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal", checkpoint_every=1, enabled=False)
    record = journal.create_job_blocking("default", SOURCES)

    updated = journal.checkpoint_blocking(record, 0)

    assert updated.checkpoint is None


def test_a_torn_write_falls_back_to_the_previous_good_record(
    journal: Journal, caplog: pytest.LogCaptureFixture
) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    for index in range(6):
        record = journal.checkpoint_blocking(record, index)

    path = journal._job_path(record.job_id)
    path.write_text('{"checksum": "bogus", "body": "{}"}', encoding="utf-8")

    recovered = journal.load_job_blocking(record.job_id)

    assert recovered.checkpoint is not None
    assert recovered.checkpoint.last_document_index == 2


def test_a_truncated_record_falls_back_too(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    record = journal.checkpoint_blocking(record, 2)
    journal.save_job_blocking(record)

    journal._job_path(record.job_id).write_text('{"checksum": "abc", "bo', encoding="utf-8")

    assert journal.load_job_blocking(record.job_id).job_id == record.job_id


def test_a_missing_live_record_falls_back_to_the_previous_good_record(journal: Journal) -> None:
    """A reader between a rewrite's two renames got NOT_FOUND for a job that existed."""
    record = journal.create_job_blocking("default", SOURCES)
    record = journal.checkpoint_blocking(record, 2)
    path = journal._job_path(record.job_id)
    path.replace(path.with_suffix(path.suffix + ".prev"))

    recovered = journal.load_job_blocking(record.job_id)

    assert recovered.job_id == record.job_id
    assert recovered.checkpoint is not None
    assert recovered.checkpoint.last_document_index == 2


def test_a_missing_live_record_with_a_torn_previous_copy_is_corrupt_not_unknown(
    journal: Journal,
) -> None:
    """A job that left a trace on disk is damaged, which is not the same as never created."""
    record = journal.create_job_blocking("default", SOURCES)
    path = journal._job_path(record.job_id)
    path.unlink()
    path.with_suffix(path.suffix + ".prev").write_text("garbage", encoding="utf-8")

    with pytest.raises(IngestionError, match="corrupt") as caught:
        journal.load_job_blocking(record.job_id)

    assert caught.value.code is ErrorCode.CHUNK_FAILED


def test_a_record_that_is_not_text_falls_back_like_any_other_torn_write(journal: Journal) -> None:
    """Bytes that are not UTF-8 raised a decode error instead of reaching the checksum."""
    record = journal.create_job_blocking("default", SOURCES)
    record = journal.checkpoint_blocking(record, 2)
    journal.save_job_blocking(record)
    journal._job_path(record.job_id).write_bytes(b"\xff\xfe\x00torn")

    assert journal.load_job_blocking(record.job_id).job_id == record.job_id


def busy_for(monkeypatch: pytest.MonkeyPatch, path: Path, reads: float) -> list[Path]:
    """Make the first ``reads`` reads of ``path`` fail as Windows does mid-``os.replace``."""
    real = Path.read_text
    seen: list[Path] = []

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == path:
            seen.append(self)
            if len(seen) <= reads:
                raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return seen


def test_a_record_busy_for_the_first_reads_is_read_once_it_settles(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read racing a rewrite on Windows raised a raw PermissionError, which became a 500."""
    record = journal.create_job_blocking("default", SOURCES)
    record = journal.checkpoint_blocking(record, 2)
    seen = busy_for(monkeypatch, journal._job_path(record.job_id), reads=3)

    loaded = journal.load_job_blocking(record.job_id)

    assert loaded.checkpoint == record.checkpoint
    assert len(seen) == 4


def test_a_record_that_stays_busy_is_read_from_its_previous_copy(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    record = journal.checkpoint_blocking(record, 2)
    busy_for(monkeypatch, journal._job_path(record.job_id), reads=float("inf"))

    loaded = journal.load_job_blocking(record.job_id)

    assert loaded.job_id == record.job_id
    assert loaded.checkpoint is None


def test_a_record_that_stays_busy_with_no_previous_copy_is_a_retryable_typed_error(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never a raw OSError: the caller gets a typed error that says to try again."""
    record = journal.create_job_blocking("default", SOURCES)
    busy_for(monkeypatch, journal._job_path(record.job_id), reads=float("inf"))

    with pytest.raises(IngestionError, match="being rewritten") as caught:
        journal.load_job_blocking(record.job_id)

    assert caught.value.code is ErrorCode.NOT_READY
    assert caught.value.retryable is True
    assert isinstance(caught.value.__cause__, PermissionError)


def test_a_job_looked_up_under_a_root_that_is_a_file_is_unknown(tmp_path: Path) -> None:
    blocker = tmp_path / "journal"
    blocker.write_text("a file where the journal directory should be", encoding="utf-8")

    with pytest.raises(IngestionError) as caught:
        Journal(blocker).load_job_blocking("job_missing")

    assert caught.value.code is ErrorCode.NOT_FOUND


def test_an_unrecoverable_record_is_a_typed_error(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    path = journal._job_path(record.job_id)
    path.write_text("garbage", encoding="utf-8")
    path.with_suffix(path.suffix + ".prev").write_text("also garbage", encoding="utf-8")

    with pytest.raises(IngestionError, match="corrupt"):
        journal.load_job_blocking(record.job_id)


def test_writes_never_leave_a_temporary_file_behind(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    journal.save_job_blocking(record)

    assert list((journal.root / "jobs").glob("*.tmp")) == []


def test_document_outcomes_are_recorded_and_filterable(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    journal.record_document_blocking(
        record.job_id, DocumentRecord(document_id="d_1", source="a.pdf", status="indexed")
    )
    journal.record_document_blocking(
        record.job_id, DocumentRecord(document_id="d_2", source="b.pdf", status="deduplicated")
    )

    assert len(list(journal.documents_blocking(record.job_id))) == 2
    indexed = list(journal.documents_blocking(record.job_id, status="indexed"))
    assert [entry.document_id for entry in indexed] == ["d_1"]


def test_documents_of_an_unstarted_job_are_empty(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)

    assert list(journal.documents_blocking(record.job_id)) == []


def test_a_dead_letter_entry_carries_a_machine_readable_reason(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)

    entry = journal.dead_letter_blocking(
        record.job_id,
        document="d_1",
        source="broken.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="the PDF could not be read",
        attempts=3,
        trace_id="a" * 32,
    )

    assert entry.dead_lettered is True
    assert entry.reason_code == ErrorCode.PARSE_FAILED.value
    assert entry.attempts == 3
    assert entry.first_failed_at is not None
    assert journal.dead_lettered_blocking(record.job_id) == [entry]


def test_dead_lettering_does_not_stop_other_documents(journal: Journal) -> None:
    record = journal.create_job_blocking("default", SOURCES)
    journal.dead_letter_blocking(
        record.job_id,
        document="d_1",
        source="broken.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )
    journal.record_document_blocking(
        record.job_id, DocumentRecord(document_id="d_2", source="fine.pdf", status="indexed")
    )

    assert journal.counts_blocking(record.job_id) == {"dead_lettered": 1, "indexed": 1, "total": 2}


def test_content_hashes_make_a_re_ingest_a_no_op(journal: Journal) -> None:
    assert journal.is_duplicate_blocking("default", "hash-1") is False

    journal.remember_content_blocking("default", "hash-1", "d_1")

    assert journal.is_duplicate_blocking("default", "hash-1") is True
    assert journal.known_content_blocking("default") == {"hash-1": "d_1"}


def test_dedup_is_scoped_per_collection(journal: Journal) -> None:
    journal.remember_content_blocking("alpha", "hash-1", "d_1")

    assert journal.is_duplicate_blocking("beta", "hash-1") is False


def test_an_unknown_collection_has_no_hashes(journal: Journal) -> None:
    assert journal.known_content_blocking("never-used") == {}


def test_a_forgotten_document_is_no_longer_a_duplicate(journal: Journal) -> None:
    """The dedup file was append-only, so content removed from a corpus could never return."""
    journal.remember_content_blocking("default", "hash-1", "d_1")

    assert journal.forget_document_blocking("default", "d_1") is True

    assert journal.is_duplicate_blocking("default", "hash-1") is False
    assert journal.known_content_blocking("default") == {}


def test_a_forgotten_document_leaves_the_lockfile_inputs(journal: Journal) -> None:
    """The lockfile is built from ``document_hashes``, so it listed documents long deleted."""
    journal.remember_content_blocking("default", "hash-1", "d_1")
    journal.remember_content_blocking("default", "hash-2", "d_2")

    journal.forget_document_blocking("default", "d_1")

    assert journal.document_hashes_blocking("default") == {"d_2": "hash-2"}


def test_forgetting_takes_every_hash_the_document_was_indexed_under(journal: Journal) -> None:
    journal.remember_content_blocking("default", "hash-old", "d_1")
    journal.remember_content_blocking("default", "hash-new", "d_1")

    journal.forget_document_blocking("default", "d_1")

    assert journal.known_content_blocking("default") == {}


def test_a_document_remembered_under_a_new_hash_drops_its_old_one(journal: Journal) -> None:
    """A superseded hash stayed a duplicate, so restoring an edited document was skipped."""
    journal.remember_content_blocking("default", "hash-old", "d_1")
    journal.remember_content_blocking("default", "hash-new", "d_1")

    assert journal.known_content_blocking("default") == {"hash-new": "d_1"}
    assert journal.document_hashes_blocking("default") == {"d_1": "hash-new"}


def test_a_hash_another_document_took_over_is_not_dropped_with_the_first(
    journal: Journal,
) -> None:
    journal.remember_content_blocking("default", "hash-1", "d_1")
    journal.remember_content_blocking("default", "hash-1", "d_2")

    journal.forget_document_blocking("default", "d_1")

    assert journal.known_content_blocking("default") == {"hash-1": "d_2"}


def test_forgetting_one_document_keeps_another_holding_other_content(journal: Journal) -> None:
    journal.remember_content_blocking("default", "hash-1", "d_1")
    journal.remember_content_blocking("default", "hash-2", "d_2")

    journal.forget_document_blocking("default", "d_1")

    assert journal.known_content_blocking("default") == {"hash-2": "d_2"}


def test_a_tombstone_is_one_appended_record(journal: Journal, tmp_path: Path) -> None:
    """The format other tools read: the remembered hash stays, a tombstone follows it."""
    journal.remember_content_blocking("default", "hash-1", "d_1")

    journal.forget_document_blocking("default", "d_1")

    path = tmp_path / "journal" / "collections" / "default" / "content-hashes.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"content_hash": "hash-1", "document_id": "d_1"}
    assert set(lines[1]) == {"deleted_at", "document_id", "tombstone"}
    assert lines[1]["document_id"] == "d_1"
    assert lines[1]["tombstone"] is True


def test_a_document_ingested_again_after_its_tombstone_is_known_again(journal: Journal) -> None:
    """A tombstone ends one life of a document, not every later one."""
    journal.remember_content_blocking("default", "hash-1", "d_1")
    journal.forget_document_blocking("default", "d_1")

    journal.remember_content_blocking("default", "hash-1", "d_1")

    assert journal.known_content_blocking("default") == {"hash-1": "d_1"}


def test_forgetting_an_unknown_document_writes_nothing(journal: Journal, tmp_path: Path) -> None:
    """A delete that is retried must not grow the file once per retry."""
    journal.remember_content_blocking("default", "hash-1", "d_1")
    journal.forget_document_blocking("default", "d_1")
    path = tmp_path / "journal" / "collections" / "default" / "content-hashes.jsonl"
    before = path.read_text(encoding="utf-8")

    assert journal.forget_document_blocking("default", "d_1") is False
    assert journal.forget_document_blocking("default", "never-ingested") is False
    assert journal.forget_document_blocking("never-used", "d_1") is False

    assert path.read_text(encoding="utf-8") == before


def test_a_tombstone_is_scoped_to_its_collection(journal: Journal) -> None:
    journal.remember_content_blocking("alpha", "hash-1", "d_1")
    journal.remember_content_blocking("beta", "hash-1", "d_1")

    journal.forget_document_blocking("alpha", "d_1")

    assert journal.known_content_blocking("beta") == {"hash-1": "d_1"}


def test_a_job_record_round_trips_through_its_serialized_form() -> None:
    record = JobRecord(job_id="job_1", collection="default", counts={"indexed": 2})

    restored = JobRecord.from_dict(json.loads(json.dumps(record.as_dict())))

    assert restored == record


def test_a_full_filesystem_becomes_a_typed_error_not_a_raw_oserror(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """failure-modes.md row 33 promises the OS write error arrives typed, and it did not."""
    full = OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(full))

    with pytest.raises(IngestionError) as caught:
        journal.create_job_blocking("default", SOURCES)

    assert caught.value.code is ErrorCode.INTERNAL
    assert "No space left on device" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_a_failed_publish_leaves_no_temporary_file_behind(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray ``.tmp`` would be mistaken for a torn write by whoever looked next."""
    created = journal.create_job_blocking("default", SOURCES)
    real_replace = os.replace

    def fail_on_publish(src: object, dst: object, **kwargs: object) -> None:
        if str(src).endswith(".tmp"):
            raise OSError(errno.ENOSPC, "No space left on device")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", fail_on_publish)

    with pytest.raises(IngestionError):
        journal.save_job_blocking(created)

    monkeypatch.undo()
    assert list(journal.root.rglob("*.tmp")) == []


class TestTheAsyncFaceKeepsFileIoOffTheEventLoop:
    """TASK-0399: every journal operation a coroutine reaches runs in a worker thread.

    The cases above drive the ``*_blocking`` twins, because that is where the behaviour
    lives. These prove the other half: that the async method does the same thing, and does
    it somewhere other than the thread running the loop.
    """

    async def test_a_created_job_is_persisted_and_reloadable(self, journal: Journal) -> None:
        created = await journal.create_job("default", SOURCES)

        reloaded = await journal.load_job(created.job_id)

        assert reloaded.job_id == created.job_id
        assert reloaded.sources == SOURCES

    async def test_a_replayed_idempotency_key_returns_the_original_job(
        self, journal: Journal
    ) -> None:
        first = await journal.create_job("default", SOURCES, idempotency_key="key-1")
        second = await journal.create_job("default", SOURCES, idempotency_key="key-1")

        assert second.job_id == first.job_id

    async def test_every_write_happens_off_the_thread_running_the_loop(
        self, journal: Journal
    ) -> None:
        """The whole point: the loop's thread must not be the one touching the disk."""
        loop_thread = threading.get_ident()
        seen: list[int] = []
        real = Journal.create_job_blocking

        def record(self: Journal, *args: Any, **kwargs: Any) -> JobRecord:
            seen.append(threading.get_ident())
            return real(self, *args, **kwargs)

        with mock.patch.object(Journal, "create_job_blocking", record):
            await journal.create_job("default", SOURCES)

        assert seen == [seen[0]]
        assert seen[0] != loop_thread

    async def test_a_scan_does_not_stall_a_concurrent_coroutine(self, journal: Journal) -> None:
        """A slow journal must cost its own caller and nobody else (BENCH-0006)."""
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0)
                ticks += 1

        real = Journal.create_job_blocking

        def slowly(self: Journal, *args: Any, **kwargs: Any) -> JobRecord:
            time.sleep(0.05)
            return real(self, *args, **kwargs)

        beat = asyncio.create_task(heartbeat())
        try:
            with mock.patch.object(Journal, "create_job_blocking", slowly):
                await journal.create_job("default", SOURCES)
        finally:
            beat.cancel()

        assert ticks > 0

    async def test_a_checkpoint_that_is_not_due_pays_no_thread_hop(self, journal: Journal) -> None:
        """Most documents are not checkpoints, so most must not reach a worker thread."""
        record = await journal.create_job("default", SOURCES)

        with mock.patch.object(Journal, "checkpoint_blocking") as written:
            unchanged = await journal.checkpoint(record, 0)

        written.assert_not_called()
        assert unchanged is record

    async def test_a_checkpoint_that_is_due_is_written(self, journal: Journal) -> None:
        record = await journal.create_job("default", SOURCES)

        for index in range(3):
            record = await journal.checkpoint(record, index)

        assert record.checkpoint is not None
        assert record.checkpoint.last_document_index == 2

    async def test_the_per_document_pair_is_one_hop(self, journal: Journal) -> None:
        """Two appends per document would be two handoffs; it has to be one."""
        created = await journal.create_job("default", SOURCES)
        entry = DocumentRecord(
            document_id="doc-1", source="a.md", status="indexed", content_hash="hash-1"
        )

        with mock.patch("asyncio.to_thread", wraps=asyncio.to_thread) as offloaded:
            await journal.record_indexed(created.job_id, "default", entry, dedup=True)

        assert offloaded.call_count == 1
        assert [record.document_id for record in await journal.documents(created.job_id)] == [
            "doc-1"
        ]
        assert await journal.known_content("default") == {"hash-1": "doc-1"}

    async def test_the_per_document_pair_honours_the_dedup_toggle(self, journal: Journal) -> None:
        created = await journal.create_job("default", SOURCES)
        entry = DocumentRecord(
            document_id="doc-1", source="a.md", status="indexed", content_hash="hash-1"
        )

        await journal.record_indexed(created.job_id, "default", entry, dedup=False)

        assert await journal.known_content("default") == {}
        assert len(await journal.documents(created.job_id)) == 1


def refuse_appends(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    """Make every append to a journal log fail with ``error``; reads still work."""
    real_open = Path.open

    def open_log(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self.suffix == ".jsonl" and "a" in mode:
            raise error
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_log)


APPENDS: dict[str, Callable[[Journal], object]] = {
    "record_document": lambda journal: journal.record_document_blocking(
        "job_1", DocumentRecord(document_id="d_1", source="a.md", status="indexed")
    ),
    "remember_content": lambda journal: journal.remember_content_blocking(
        "default", "hash-1", "d_1"
    ),
    "tombstone": lambda journal: journal.forget_document_blocking("default", "d_1"),
}


@pytest.mark.parametrize("append", sorted(APPENDS))
def test_a_full_disk_on_an_append_is_a_typed_error_not_a_raw_oserror(
    journal: Journal, monkeypatch: pytest.MonkeyPatch, append: str
) -> None:
    """TASK-0475: the appends had no handler, so a full disk escaped as a raw OSError."""
    journal.remember_content_blocking("default", "hash-1", "d_1")
    refuse_appends(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))

    with pytest.raises(IngestionError) as caught:
        APPENDS[append](journal)

    assert caught.value.code is ErrorCode.INTERNAL
    assert "No space left on device" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_an_append_under_a_root_that_is_a_file_is_a_typed_error(tmp_path: Path) -> None:
    blocker = tmp_path / "journal"
    blocker.write_text("a file where the journal directory should be", encoding="utf-8")

    with pytest.raises(IngestionError):
        APPENDS["record_document"](Journal(blocker))


def test_a_line_a_crash_left_unfinished_is_skipped_and_the_next_append_survives(
    journal: Journal,
) -> None:
    """A torn last line made every later read of the log raise a JSON error."""
    journal.remember_content_blocking("default", "hash-1", "d_1")
    path = journal.root / "collections" / "default" / "content-hashes.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"content_hash": "hash-2", "docu')

    journal.remember_content_blocking("default", "hash-3", "d_3")

    assert journal.known_content_blocking("default") == {"hash-1": "d_1", "hash-3": "d_3"}


def test_a_torn_document_line_is_skipped_too(journal: Journal) -> None:
    APPENDS["record_document"](journal)
    path = journal._documents_path("job_1")
    with path.open("ab") as handle:
        handle.write(b'{"document_id": "d_2", "sou')

    APPENDS["record_document"](journal)

    assert [entry.document_id for entry in journal.documents_blocking("job_1")] == ["d_1", "d_1"]


def test_a_checkpoint_flushes_the_logs_it_covers_before_it_is_written(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appends are not flushed one by one, so the checkpoint must not outrun them."""
    record = journal.create_job_blocking("default", SOURCES)
    journal.record_indexed_blocking(
        record.job_id,
        "default",
        DocumentRecord(document_id="d_1", source="a.md", status="indexed", content_hash="h"),
        dedup=True,
    )
    flushed: list[str] = []
    real_sync = journal_module._sync_file

    def sync(path: Path) -> None:
        flushed.append(path.name)
        real_sync(path)

    monkeypatch.setattr(journal_module, "_sync_file", sync)

    journal.checkpoint_blocking(record, 2)

    assert flushed[:2] == ["documents.jsonl", "content-hashes.jsonl"]
    assert any(name.endswith(".tmp") for name in flushed[2:])


def test_an_append_waits_for_the_lock_and_gives_up_typed(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = journal._documents_path("job_1")
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(journal_module, "_LOCK_TIMEOUT_SECONDS", 0.05)

    with journal_module._exclusive(path), pytest.raises(IngestionError) as caught:
        APPENDS["record_document"](journal)

    assert caught.value.code is ErrorCode.NOT_READY
    assert journal.documents_blocking("job_1") == []


def job_reads(monkeypatch: pytest.MonkeyPatch, journal: Journal) -> list[Path]:
    """Record every job record the journal opens from now on."""
    real = Path.read_text
    opened: list[Path] = []

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.parent == journal.root / "jobs":
            opened.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return opened


def test_a_keyed_lookup_opens_one_record_however_many_jobs_exist(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lookup globbed and read every job file: 633 ms at 10,000 jobs (BENCH-0004)."""
    for index in range(30):
        journal.create_job_blocking("default", SOURCES, idempotency_key=f"key-{index}")
    opened = job_reads(monkeypatch, journal)

    found = journal.find_by_idempotency_key_blocking("key-17")
    missing = journal.find_by_idempotency_key_blocking("never-sent")

    assert found is not None
    assert found.idempotency_key == "key-17"
    assert missing is None
    assert len(opened) == 1


def test_a_journal_written_before_the_index_is_indexed_on_first_lookup(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operators have journals on disk with no index; their keys must still replay."""
    first = journal.create_job_blocking("default", SOURCES, idempotency_key="key-1")
    journal.create_job_blocking("default", SOURCES, idempotency_key="key-2", tenant="acme")
    shutil.rmtree(journal.root / "job-keys")

    assert journal.find_by_idempotency_key_blocking("key-1") == first
    assert journal.find_by_idempotency_key_blocking("key-2") is None
    opened = job_reads(monkeypatch, journal)
    assert journal.create_job_blocking("default", SOURCES, idempotency_key="key-1") == first
    assert len(opened) == 1


def test_an_entry_naming_a_job_that_is_gone_is_rebuilt_not_trusted(journal: Journal) -> None:
    """An interruption between the entry and the record leaves an entry naming nothing."""
    first = journal.create_job_blocking("default", SOURCES, idempotency_key="key-1")
    journal._job_path(first.job_id).unlink()

    second = journal.create_job_blocking("default", SOURCES, idempotency_key="key-1")

    assert second.job_id != first.job_id
    assert journal.find_by_idempotency_key_blocking("key-1") == second


def test_an_entry_naming_another_tenants_job_is_never_returned(journal: Journal) -> None:
    """TASK-0318's scoping must survive the index: a key is per tenant."""
    theirs = journal.create_job_blocking("default", SOURCES, idempotency_key="k", tenant="a")
    ours = journal.create_job_blocking("default", SOURCES, idempotency_key="k", tenant="b")
    index = journal.root / "job-keys"
    swapped = index / journal_module._key_entry_name("k", "b")
    swapped.write_bytes((index / journal_module._key_entry_name("k", "a")).read_bytes())

    assert journal.find_by_idempotency_key_blocking("k", tenant="a") == theirs
    assert journal.find_by_idempotency_key_blocking("k", tenant="b") == ours


def test_the_lookup_on_an_empty_journal_creates_nothing(journal: Journal) -> None:
    assert journal.find_by_idempotency_key_blocking("key-1") is None
    assert not journal.root.exists()
