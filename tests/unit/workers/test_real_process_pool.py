"""The ingestion pipeline through the production ``ProcessPoolExecutor``.

Every other pool and pipeline test injects a ``ThreadPoolExecutor``, so for fourteen slices
nothing pickled a ``Settings`` object, a ``DocumentTask`` or a ``ParseOutcome`` across a
process boundary, started a real worker, or saw a real ``BrokenProcessPool`` (TASK-0455).
These tests pass no ``executor_factory`` at all: what runs is what a deployment runs.

They are marked ``must_run`` because a test of the real pool that is skipped in every lane
proves nothing. Chunking is ``fixed`` with the estimating token counter so a worker loads no
model, and the stand-in parse targets live in ``process_targets.py`` because a spawned child
has to be able to import them.
"""

from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from fasterrag.workers import cpu_pool
from tests.unit.services.test_ingestion import KeywordLegAdapter, StubEmbedder
from tests.unit.workers import process_targets

pytestmark = pytest.mark.must_run

NAMES = ("expenses.md", "leave.md", "travel.md")


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "embeddings": {"batch_size": 8},
        "workers": {"cpu_pool_size": 2, "embedding_pool_size": 1, "queue_depth": 200},
        "chunking": {
            "strategy": "fixed",
            "token_counter": "estimate",
            "chunk_size": 64,
            "overlap": 8,
        },
        "reliability": {"retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False}},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for name in NAMES:
        (root / name).write_text(
            f"# {name}\n\n" + f"The policy in {name} applies to every employee. " * 20,
            encoding="utf-8",
        )
    return root


def build(tmp_path: Path, configured: Settings) -> tuple[IngestionService, KeywordLegAdapter]:
    """Return a service that builds its own executor, and the adapter it writes through."""
    adapter = KeywordLegAdapter(configured)
    service = IngestionService(
        configured,
        journal=Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=adapter,
        router=TieringRouter(StubEmbedder(configured)),
    )
    return service, adapter


async def test_three_documents_ingest_through_the_default_executor(
    tmp_path: Path, corpus: Path
) -> None:
    """No test ran the real process pool, so nothing had ever crossed its pickle boundary."""
    service, adapter = build(tmp_path, settings())

    record = await service.accept([str(corpus / name) for name in NAMES])
    settled = await service.run(record)

    assert settled.status == "completed"
    assert settled.counts["parsed"] == 3
    assert settled.counts["dead_lettered"] == 0
    assert settled.counts["indexed"] == len(adapter.points) > 0
    indexed = service.journal.documents_blocking(record.job_id, status="indexed")
    assert sorted(Path(entry.source).name for entry in indexed) == sorted(NAMES)


async def test_a_failed_document_keeps_its_reason_code_across_the_process_boundary(
    tmp_path: Path, corpus: Path
) -> None:
    """A typed error raised in a worker is pickled back; its code must survive the trip."""
    service, _ = build(tmp_path, settings())

    record = await service.accept([str(corpus / NAMES[0]), str(corpus / "absent.md")])
    settled = await service.run(record)

    dead = service.journal.dead_lettered_blocking(record.job_id)
    assert settled.status == "partial"
    assert [entry.reason_code for entry in dead] == ["PARSE_FAILED"]
    assert "absent.md" in (dead[0].detail or "")


async def test_a_worker_failure_is_filed_under_the_trace_id_of_its_job(
    tmp_path: Path, corpus: Path
) -> None:
    """An error raised in a worker carried the worker's own trace id, never the job's."""
    service, _ = build(tmp_path, settings())

    record = await service.accept([str(corpus / NAMES[0]), str(corpus / "absent.md")])
    await service.run(record)

    indexed = service.journal.documents_blocking(record.job_id, status="indexed")
    dead = service.journal.dead_lettered_blocking(record.job_id)
    assert indexed[0].trace_id is not None
    assert dead[0].trace_id == indexed[0].trace_id


async def test_a_killed_worker_fails_the_job_with_a_typed_error_and_settles_it(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real BrokenProcessPool had never been raised in a test, only its message built."""
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.die)
    service, adapter = build(tmp_path, settings())

    record = await service.accept([str(corpus / name) for name in NAMES])
    with pytest.raises(IngestionError) as raised:
        await service.run(record)

    assert raised.value.code is ErrorCode.CHUNK_FAILED
    assert not raised.value.retryable
    assert service.journal.load_job_blocking(record.job_id).status == "failed"
    assert service.journal.load_job_blocking(record.job_id).finished_at is not None
    assert adapter.points == []
