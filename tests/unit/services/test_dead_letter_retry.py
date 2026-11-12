"""A dead-letter retry is idempotent and bounded (TASK-0474).

``POST /v1/ingest/{job_id}/retry-dlq`` created its job with no idempotency key, so every
repeated call started another ingest of the same failed documents, and
``ingestion.dlq.max_retries`` was written onto each dead-letter entry and enforced nowhere.
"""

from pathlib import Path

import pytest

from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.ingestion import plan_dead_letter_retry, retry_generation
from fasterrag.services.journal import JobRecord, Journal

SOURCES = [{"type": "path", "value": "corpus/"}]


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal")


def fail(journal: Journal, job: JobRecord, *sources: str) -> None:
    """Dead-letter one document per source under ``job``."""
    for index, source in enumerate(sources):
        journal.dead_letter_blocking(
            job.job_id,
            document=f"d_{index}",
            source=source,
            reason_code=ErrorCode.PARSE_FAILED,
            detail="unreadable",
            attempts=3,
        )


async def retried(journal: Journal, job: JobRecord, *, max_retries: int = 3) -> JobRecord:
    """Plan a retry of ``job``'s dead letters and create it the way the route does."""
    plan = await plan_dead_letter_retry(journal, job.job_id, tenant=None, max_retries=max_retries)
    assert plan.idempotency_key is not None
    return await journal.create_job(
        job.collection,
        [{"type": "path", "value": source} for source in plan.sources],
        idempotency_key=plan.idempotency_key,
        tenant=plan.original.tenant,
    )


async def test_planning_the_same_retry_twice_derives_the_same_key(journal: Journal) -> None:
    """With no key there was nothing for a repeat to be recognised by."""
    job = journal.create_job_blocking("default", SOURCES)
    fail(journal, job, "corpus/a.pdf", "corpus/b.pdf")

    first = await retried(journal, job)
    second = await retried(journal, job)

    assert second.job_id == first.job_id
    assert retry_generation(first) == 1


async def test_a_job_with_no_dead_letters_plans_nothing(journal: Journal) -> None:
    job = journal.create_job_blocking("default", SOURCES)

    plan = await plan_dead_letter_retry(journal, job.job_id, tenant=None, max_retries=3)

    assert (plan.sources, plan.idempotency_key) == ((), None)


async def test_a_dead_letter_added_after_a_retry_is_retried_by_a_new_job(journal: Journal) -> None:
    """A job resumed after its first retry can dead-letter more; those must not be lost."""
    job = journal.create_job_blocking("default", SOURCES)
    fail(journal, job, "corpus/a.pdf")
    first = await retried(journal, job)
    fail(journal, job, "corpus/a.pdf", "corpus/b.pdf")

    second = await retried(journal, job)

    assert second.job_id != first.job_id
    assert retry_generation(second) == 1


async def test_each_retry_of_a_retry_is_one_generation_further(journal: Journal) -> None:
    job = journal.create_job_blocking("default", SOURCES)
    fail(journal, job, "corpus/a.pdf")
    first = await retried(journal, job)
    fail(journal, first, "corpus/a.pdf")

    second = await retried(journal, first)

    assert [retry_generation(record) for record in (job, first, second)] == [0, 1, 2]


async def test_a_retry_past_max_retries_is_refused_with_a_typed_error(journal: Journal) -> None:
    """``ingestion.dlq.max_retries`` was recorded and never enforced."""
    job = journal.create_job_blocking("default", SOURCES)
    fail(journal, job, "corpus/a.pdf")
    first = await retried(journal, job, max_retries=1)
    fail(journal, first, "corpus/a.pdf")

    with pytest.raises(IngestionError) as refused:
        await plan_dead_letter_retry(journal, first.job_id, tenant=None, max_retries=1)

    assert refused.value.code is ErrorCode.DLQ_RETRIES_EXHAUSTED
    assert refused.value.retryable is False
    assert "ingestion.dlq.max_retries" in refused.value.detail
    assert len(list((journal.root / "jobs").glob("*.json"))) == 2


async def test_max_retries_of_zero_refuses_every_retry(journal: Journal) -> None:
    job = journal.create_job_blocking("default", SOURCES)
    fail(journal, job, "corpus/a.pdf")

    with pytest.raises(IngestionError) as refused:
        await plan_dead_letter_retry(journal, job.job_id, tenant=None, max_retries=0)

    assert refused.value.code is ErrorCode.DLQ_RETRIES_EXHAUSTED


async def test_a_retry_made_before_the_limit_was_lowered_is_still_returned(
    journal: Journal,
) -> None:
    """Repeating a retry that already exists starts nothing, so the limit has nothing to stop."""
    job = journal.create_job_blocking("default", SOURCES)
    fail(journal, job, "corpus/a.pdf")
    first = await retried(journal, job, max_retries=3)

    plan = await plan_dead_letter_retry(journal, job.job_id, tenant=None, max_retries=0)

    assert plan.idempotency_key == first.idempotency_key


def test_a_caller_supplied_key_is_generation_zero() -> None:
    assert retry_generation(JobRecord("job_1", "default", idempotency_key="nightly-1")) == 0
    assert retry_generation(JobRecord("job_1", "default", idempotency_key="retry-dlq:x")) == 0
