import json
from pathlib import Path

import pytest

from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.journal import DocumentRecord, JobRecord, Journal

SOURCES = [{"type": "path", "value": "/data/docs"}]


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal", checkpoint_every=3)


def test_a_created_job_is_persisted_and_reloadable(journal: Journal) -> None:
    created = journal.create_job("default", SOURCES)

    reloaded = journal.load_job(created.job_id)

    assert reloaded.job_id == created.job_id
    assert reloaded.collection == "default"
    assert reloaded.sources == SOURCES
    assert reloaded.status == "queued"


def test_an_unknown_job_is_reported_as_not_found(journal: Journal) -> None:
    with pytest.raises(IngestionError, match="unknown ingest job") as caught:
        journal.load_job("job_missing")

    assert caught.value.code is ErrorCode.NOT_FOUND


def test_a_replayed_idempotency_key_returns_the_original_job(journal: Journal) -> None:
    first = journal.create_job("default", SOURCES, idempotency_key="key-1")
    second = journal.create_job("default", SOURCES, idempotency_key="key-1")

    assert second.job_id == first.job_id


def test_different_keys_create_different_jobs(journal: Journal) -> None:
    first = journal.create_job("default", SOURCES, idempotency_key="key-1")
    second = journal.create_job("default", SOURCES, idempotency_key="key-2")

    assert first.job_id != second.job_id


def test_checkpoints_are_written_only_at_the_configured_interval(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)

    for index in range(5):
        record = journal.checkpoint(record, index)

    assert record.checkpoint is not None
    assert record.checkpoint.last_document_index == 2


def test_a_crash_resumes_from_the_document_after_the_checkpoint(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)
    for index in range(6):
        record = journal.checkpoint(record, index)

    resumed = journal.load_job(record.job_id)

    assert journal.resume_index(resumed) == 6


def test_a_job_with_no_checkpoint_resumes_from_the_beginning(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)

    assert journal.resume_index(record) == 0


def test_checkpointing_is_skipped_when_the_journal_is_disabled(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal", checkpoint_every=1, enabled=False)
    record = journal.create_job("default", SOURCES)

    updated = journal.checkpoint(record, 0)

    assert updated.checkpoint is None


def test_a_torn_write_falls_back_to_the_previous_good_record(
    journal: Journal, caplog: pytest.LogCaptureFixture
) -> None:
    record = journal.create_job("default", SOURCES)
    for index in range(6):
        record = journal.checkpoint(record, index)

    path = journal._job_path(record.job_id)
    path.write_text('{"checksum": "bogus", "body": "{}"}', encoding="utf-8")

    recovered = journal.load_job(record.job_id)

    assert recovered.checkpoint is not None
    assert recovered.checkpoint.last_document_index == 2


def test_a_truncated_record_falls_back_too(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)
    record = journal.checkpoint(record, 2)
    journal.save_job(record)

    journal._job_path(record.job_id).write_text('{"checksum": "abc", "bo', encoding="utf-8")

    assert journal.load_job(record.job_id).job_id == record.job_id


def test_an_unrecoverable_record_is_a_typed_error(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)
    path = journal._job_path(record.job_id)
    path.write_text("garbage", encoding="utf-8")
    path.with_suffix(path.suffix + ".prev").write_text("also garbage", encoding="utf-8")

    with pytest.raises(IngestionError, match="corrupt"):
        journal.load_job(record.job_id)


def test_writes_never_leave_a_temporary_file_behind(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)
    journal.save_job(record)

    assert list((journal.root / "jobs").glob("*.tmp")) == []


def test_document_outcomes_are_recorded_and_filterable(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)
    journal.record_document(
        record.job_id, DocumentRecord(document_id="d_1", source="a.pdf", status="indexed")
    )
    journal.record_document(
        record.job_id, DocumentRecord(document_id="d_2", source="b.pdf", status="deduplicated")
    )

    assert len(list(journal.documents(record.job_id))) == 2
    indexed = list(journal.documents(record.job_id, status="indexed"))
    assert [entry.document_id for entry in indexed] == ["d_1"]


def test_documents_of_an_unstarted_job_are_empty(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)

    assert list(journal.documents(record.job_id)) == []


def test_a_dead_letter_entry_carries_a_machine_readable_reason(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)

    entry = journal.dead_letter(
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
    assert journal.dead_lettered(record.job_id) == [entry]


def test_dead_lettering_does_not_stop_other_documents(journal: Journal) -> None:
    record = journal.create_job("default", SOURCES)
    journal.dead_letter(
        record.job_id,
        document="d_1",
        source="broken.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )
    journal.record_document(
        record.job_id, DocumentRecord(document_id="d_2", source="fine.pdf", status="indexed")
    )

    assert journal.counts(record.job_id) == {"dead_lettered": 1, "indexed": 1, "total": 2}


def test_content_hashes_make_a_re_ingest_a_no_op(journal: Journal) -> None:
    assert journal.is_duplicate("default", "hash-1") is False

    journal.remember_content("default", "hash-1", "d_1")

    assert journal.is_duplicate("default", "hash-1") is True
    assert journal.known_content("default") == {"hash-1": "d_1"}


def test_dedup_is_scoped_per_collection(journal: Journal) -> None:
    journal.remember_content("alpha", "hash-1", "d_1")

    assert journal.is_duplicate("beta", "hash-1") is False


def test_an_unknown_collection_has_no_hashes(journal: Journal) -> None:
    assert journal.known_content("never-used") == {}


def test_a_job_record_round_trips_through_its_serialized_form() -> None:
    record = JobRecord(job_id="job_1", collection="default", counts={"indexed": 2})

    restored = JobRecord.from_dict(json.loads(json.dumps(record.as_dict())))

    assert restored == record
