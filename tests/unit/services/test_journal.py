import asyncio
import errno
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fasterrag.errors import ErrorCode, IngestionError
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
