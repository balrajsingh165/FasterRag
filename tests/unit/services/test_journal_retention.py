"""``ingestion.journal.retention_days`` bounds the journal (TASK-0474).

Every job left a record, a ``.prev`` copy, a per-document log and an idempotency index entry
behind for good, and a collection's dedup file kept every hash it was ever told, superseded
or tombstoned, so the journal grew with every ingest and never shrank.
"""

import os
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.jobs import INTERRUPTED, JobRegistry, reap_at_startup
from fasterrag.services.journal import DocumentRecord, JobRecord, Journal, create_journal

SOURCES = [{"type": "path", "value": "corpus/"}]
RETENTION_DAYS = 7
LONG_AGO = timedelta(days=30)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal", retention_days=RETENTION_DAYS)


def finished(
    journal: Journal,
    status: str = "completed",
    *,
    ago: timedelta = LONG_AGO,
    key: str | None = None,
    reason: str | None = None,
) -> JobRecord:
    """Leave a job on disk that settled ``ago``, with one logged document."""
    record = journal.create_job_blocking("default", SOURCES, idempotency_key=key)
    when = datetime.now(tz=UTC) - ago
    record = replace(record, status=status, reason_code=reason, finished_at=when.isoformat())
    journal.save_job_blocking(record)
    journal.save_job_blocking(record)
    journal.record_document_blocking(record.job_id, DocumentRecord("d_1", "a.md", "indexed"))
    stamp = when.timestamp()
    for path in journal.root.joinpath("jobs").rglob(f"{record.job_id}*"):
        if path.is_file():
            os.utime(path, (stamp, stamp))
    os.utime(journal.root / "jobs" / record.job_id / "documents.jsonl", (stamp, stamp))
    return record


def job_files(journal: Journal, job_id: str) -> list[Path]:
    """Return every file the journal holds for one job."""
    return [path for path in journal.root.joinpath("jobs").rglob("*") if job_id in str(path)]


async def test_a_job_finished_before_the_window_is_pruned_with_everything_it_left(
    journal: Journal,
) -> None:
    """The record, its previous copy, its document log and its key entry were kept for good."""
    old = finished(journal, key="nightly-1")
    assert job_files(journal, old.job_id)

    report = await journal.prune()

    assert report.jobs == (old.job_id,)
    assert job_files(journal, old.job_id) == []
    with pytest.raises(IngestionError) as gone:
        await journal.load_job(old.job_id)
    assert gone.value.code is ErrorCode.NOT_FOUND
    assert await journal.find_by_idempotency_key("nightly-1") is None


@pytest.mark.parametrize("status", ["completed", "partial", "failed"])
async def test_a_job_finished_inside_the_window_is_kept(journal: Journal, status: str) -> None:
    recent = finished(journal, status, ago=timedelta(days=RETENTION_DAYS - 1))

    report = await journal.prune()

    assert report.jobs == ()
    assert (await journal.load_job(recent.job_id)).status == status


@pytest.mark.parametrize(
    ("status", "reason"), [("queued", None), ("running", None), ("failed", INTERRUPTED)]
)
async def test_a_job_that_can_still_run_is_never_pruned(
    journal: Journal, status: str, reason: str | None
) -> None:
    """An unsettled or interrupted job holds a checkpoint a resume still needs."""
    kept = finished(journal, status, reason=reason)

    report = await journal.prune()

    assert report.jobs == ()
    assert (await journal.load_job(kept.job_id)).status == status
    assert len(await journal.documents(kept.job_id)) == 1


async def test_the_key_of_a_pruned_job_leaves_no_index_entry_behind(journal: Journal) -> None:
    """An entry naming a pruned job would send the next lookup into a full rebuild scan."""
    old = finished(journal, key="nightly-1")
    kept = finished(journal, key="nightly-2", ago=timedelta(days=1))

    await journal.prune()

    entries = [
        path for path in (journal.root / "job-keys").glob("*.json") if path.name != "built.json"
    ]
    assert len(entries) == 1
    assert kept.job_id in entries[0].read_text(encoding="utf-8")
    again = await journal.create_job("default", SOURCES, idempotency_key="nightly-1")
    assert again.job_id != old.job_id


async def test_retention_off_prunes_nothing(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal")
    old = finished(journal)

    report = await journal.prune()

    assert report.jobs == ()
    assert journal.prune_due() is False
    assert (await journal.load_job(old.job_id)).status == "completed"


def test_retention_is_off_by_default_and_read_from_configuration(tmp_path: Path) -> None:
    assert Settings().ingestion.journal.retention_days == 0
    configured = Settings.model_validate({"ingestion": {"journal": {"retention_days": 14}}})
    assert create_journal(configured, tmp_path).retention_days == 14


async def test_a_pass_is_due_once_and_not_again_within_the_hour(journal: Journal) -> None:
    assert journal.prune_due() is True

    await journal.prune()

    assert journal.prune_due() is False


async def test_a_starting_server_prunes_what_is_past_the_window(journal: Journal) -> None:
    old = finished(journal)

    await reap_at_startup(journal, JobRegistry(), stale_seconds=120)

    assert job_files(journal, old.job_id) == []


def remember(journal: Journal, document: str, digest: str) -> None:
    journal.remember_content_blocking("default", digest, document)


def dedup_lines(journal: Journal) -> int:
    path = journal.root / "collections" / "default" / "content-hashes.jsonl"
    return len(path.read_text(encoding="utf-8").splitlines())


async def test_compacting_the_dedup_file_keeps_every_live_document(journal: Journal) -> None:
    """Superseded and tombstoned hashes were kept for good; a live one must never be dropped."""
    remember(journal, "d_1", "h1")
    remember(journal, "d_1", "h1-edited")
    remember(journal, "d_2", "h2")
    journal.forget_document_blocking("default", "d_2")
    remember(journal, "d_3", "h3")
    before = journal.known_content_blocking("default")

    report = await journal.prune()

    assert journal.known_content_blocking("default") == before == {"h1-edited": "d_1", "h3": "d_3"}
    assert report.dedup_lines == 3
    assert dedup_lines(journal) == 2


async def test_a_compacted_dedup_file_still_takes_tombstones_and_new_hashes(
    journal: Journal,
) -> None:
    remember(journal, "d_1", "h1")
    remember(journal, "d_1", "h1")
    await journal.prune()

    assert journal.forget_document_blocking("default", "d_1") is True
    remember(journal, "d_2", "h1")

    assert journal.known_content_blocking("default") == {"h1": "d_2"}


async def test_a_dedup_file_with_nothing_to_drop_is_not_rewritten(journal: Journal) -> None:
    remember(journal, "d_1", "h1")
    path = journal.root / "collections" / "default" / "content-hashes.jsonl"
    written = path.stat().st_mtime_ns
    time.sleep(0.01)

    report = await journal.prune()

    assert report.dedup_lines == 0
    assert path.stat().st_mtime_ns == written


OPERATIONS = st.lists(
    st.tuples(
        st.sampled_from(["remember", "forget"]),
        st.sampled_from(["d_1", "d_2", "d_3"]),
        st.sampled_from(["h1", "h2", "h3"]),
    ),
    max_size=20,
)


@settings(max_examples=60, deadline=None)
@given(before=OPERATIONS, after=OPERATIONS)
def test_compaction_never_changes_what_the_dedup_file_means(
    tmp_path_factory: pytest.TempPathFactory,
    before: list[tuple[str, str, str]],
    after: list[tuple[str, str, str]],
) -> None:
    """Compacting at any point gives the same answers, then and after any later appends."""
    compacted = Journal(tmp_path_factory.mktemp("compacted"), retention_days=1)
    untouched = Journal(tmp_path_factory.mktemp("untouched"))

    def apply(operations: list[tuple[str, str, str]]) -> None:
        for operation, document, digest in operations:
            for journal in (compacted, untouched):
                if operation == "remember":
                    remember(journal, document, digest)
                else:
                    journal.forget_document_blocking("default", document)

    apply(before)
    compacted.prune_blocking()
    assert compacted.known_content_blocking("default") == untouched.known_content_blocking(
        "default"
    )
    apply(after)
    assert compacted.known_content_blocking("default") == untouched.known_content_blocking(
        "default"
    )
