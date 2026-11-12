"""``fasterrag ingest --resume JOB_ID`` (TASK-0436).

The journal recorded how far an interrupted job had got and no command could use it: the
only way to continue was to submit the whole corpus again and let deduplication discard
what was already indexed.
"""

import argparse
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import pipeline
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.parser import build_parser
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.journal import Checkpoint, JobRecord, Journal

JOB_ID = "job_interrupted"


class ResumingIngest:
    """An ingestion service that hands back one interrupted job and settles it."""

    def __init__(self) -> None:
        self.resumed: list[str] = []
        self.ran: list[str] = []
        self.metadata: dict[str, Any] | None = None
        self.ingested = False
        self.refusal: IngestionError | None = None
        self.closed = False

    async def accept_resume(self, job_id: str) -> JobRecord:
        if self.refusal is not None:
            raise self.refusal
        self.resumed.append(job_id)
        return JobRecord(
            job_id=job_id,
            collection="docs",
            status="queued",
            checkpoint=Checkpoint(last_document_index=41, written_at="2026-11-12T00:00:00+00:00"),
        )

    async def run(self, record: JobRecord, *, metadata: dict[str, Any] | None = None) -> JobRecord:
        self.ran.append(record.job_id)
        self.metadata = metadata
        return JobRecord(
            job_id=record.job_id,
            collection=record.collection,
            status="completed",
            counts={"indexed": 7, "dead_lettered": 0},
        )

    async def ingest(self, sources: Any, **kwargs: Any) -> JobRecord:
        self.ingested = True
        return JobRecord(job_id="job_new", collection="docs", status="completed")

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResumingIngest:
    """Install the service and journal an ingest command builds."""
    built = ResumingIngest()
    journal = Journal(tmp_path / "journal")
    monkeypatch.setattr(pipeline, "IngestionService", lambda *args, **kwargs: built)
    monkeypatch.setattr(pipeline, "create_journal", lambda settings: journal)
    monkeypatch.setattr(pipeline, "create_lock_store", lambda settings: object())
    return built


def ingest_args(config: str, *flags: str) -> argparse.Namespace:
    """Build an ingest namespace through the real parser."""
    return build_parser().parse_args(["ingest", "--config", config, *flags])


def test_resume_is_accepted_without_any_source() -> None:
    """The parser demanded a source, which a resumed job already has."""
    args = build_parser().parse_args(["ingest", "--resume", JOB_ID])

    assert (args.resume, args.sources) == (JOB_ID, [])


async def test_resume_runs_the_interrupted_job_and_reports_where_it_continued(
    config: str, service: ResumingIngest, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing could continue an interrupted job; the checkpoint was never read."""
    code = await pipeline.run_ingest(ingest_args(config, "--resume", JOB_ID), Console())
    printed = capsys.readouterr().out

    assert code == ExitCode.SUCCESS
    assert service.resumed == [JOB_ID]
    assert service.ran == [JOB_ID]
    assert service.ingested is False
    assert service.closed is True
    assert f"job             {JOB_ID}" in printed
    assert "resuming        from document 42" in printed
    assert "status          completed" in printed
    assert "indexed         7 chunks" in printed
    assert printed.count(f"job             {JOB_ID}") == 1


async def test_resume_carries_metadata_given_again(config: str, service: ResumingIngest) -> None:
    """Chunk metadata is not in the job record, so a resumed run only has what it is given."""
    await pipeline.run_ingest(
        ingest_args(config, "--resume", JOB_ID, "--metadata", "team=legal"), Console()
    )

    assert service.metadata == {"team": "legal"}


async def test_a_job_that_cannot_be_resumed_is_reported_and_fails(
    config: str, service: ResumingIngest, capsys: pytest.CaptureFixture[str]
) -> None:
    service.refusal = IngestionError(
        f"job {JOB_ID} is completed, not interrupted", code=ErrorCode.CONFLICT
    )

    code = await pipeline.run_ingest(ingest_args(config, "--resume", JOB_ID), Console())
    captured = capsys.readouterr()

    assert code == ExitCode.FAILURE
    assert service.ran == []
    assert service.closed is True
    assert ErrorCode.CONFLICT.value in captured.err
    assert "not interrupted" in captured.err


@pytest.mark.parametrize("extra", [["corpus/"], ["--dry-run"]])
async def test_resume_refuses_new_sources_and_a_dry_run(
    config: str, service: ResumingIngest, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Quietly ignoring either would run something other than what was typed."""
    code = await pipeline.run_ingest(ingest_args(config, "--resume", JOB_ID, *extra), Console())

    assert code == ExitCode.USAGE
    assert service.resumed == []
    assert "--resume" in capsys.readouterr().err


async def test_ingest_with_neither_a_source_nor_a_job_is_a_usage_error(
    config: str, service: ResumingIngest, capsys: pytest.CaptureFixture[str]
) -> None:
    """The parser used to enforce this; it cannot now that ``--resume`` takes no source."""
    code = await pipeline.run_ingest(ingest_args(config), Console())

    assert code == ExitCode.USAGE
    assert service.ingested is False
    assert "--resume JOB_ID" in capsys.readouterr().err
