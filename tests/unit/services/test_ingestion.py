"""The ingestion service's ``run`` orchestration, without a vector database.

``IngestionService.run`` is the primary write path — source resolution, both worker pools,
journal checkpointing, dead-lettering, the lockfile, cache invalidation, and the ingestion
metrics. It was covered only by a Docker-gated integration test, so on any machine without a
daemon the whole path went unexecuted and CI's unit leg never touched it.

The vector database is faked here rather than run. What is being tested is the orchestration
between the pieces, and a real backend adds a daemon dependency without exercising any of it.
"""

from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError
from fasterrag.observability import metrics
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from tests.unit.workers.test_indexer import RecordingAdapter

DIMENSIONS = 8


def threads(workers: int) -> Executor:
    """Run the CPU pool in threads, so a test spawns no processes."""
    return ThreadPoolExecutor(max_workers=workers)


class StubEmbedder(EmbeddingAdapter):
    """Returns a fixed-width vector per text, and can be made to fail."""

    provider = "stub"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.error: EmbedError | None = None
        self.embedded: list[str] = []

    @property
    def model(self) -> str:
        return "stub-model"

    @property
    def model_version(self) -> str:
        return "stub-model-v1"

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if self.error is not None:
            raise self.error
        self.embedded.extend(texts)
        return EmbeddingResult(
            vectors=[[1.0] + [0.0] * (DIMENSIONS - 1) for _ in texts],
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text])).vectors[0]

    async def health(self) -> Any:
        from fasterrag.adapters.vectordb.base import HealthStatus

        return HealthStatus(healthy=True, detail="stub")

    async def close(self) -> None:
        return None


class RecordingCache:
    """Records invalidation, which is what a corpus change is supposed to trigger."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    async def invalidate(self, reason: str) -> None:
        self.reasons.append(reason)

    async def close(self) -> None:
        return None


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "embeddings": {"batch_size": 8},
        "workers": {"embedding_pool_size": 1, "queue_depth": 200},
        "chunking": {"chunk_size": 128, "overlap": 16},
        "reliability": {"retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False}},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "expenses.md").write_text(
        "# Expenses\n\nThe meal allowance is 41 pounds per day for UK travel.\n", encoding="utf-8"
    )
    (root / "leave.md").write_text(
        "# Leave\n\nParental leave is 26 weeks at full pay.\n", encoding="utf-8"
    )
    return root


def build(
    tmp_path: Path,
    configured: Settings | None = None,
    *,
    embedder: StubEmbedder | None = None,
    **kwargs: Any,
) -> tuple[IngestionService, RecordingAdapter]:
    """Return a service and the adapter it writes through."""
    resolved = configured or settings()
    adapter = RecordingAdapter(resolved)
    service = IngestionService(
        resolved,
        journal=Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=adapter,
        router=TieringRouter(embedder or StubEmbedder(resolved)),
        executor_factory=threads,
        **kwargs,
    )
    return service, adapter


def sources(corpus: Path) -> list[str]:
    return [str(corpus / "expenses.md"), str(corpus / "leave.md")]


async def test_a_job_runs_to_completion(tmp_path: Path, corpus: Path) -> None:
    service, _ = build(tmp_path)

    record = await service.accept(sources(corpus))
    settled = await service.run(record)

    assert settled.status == "completed"
    assert settled.counts["parsed"] == 2


async def test_every_chunk_reaches_the_vector_database(tmp_path: Path, corpus: Path) -> None:
    service, adapter = build(tmp_path)

    settled = await service.run(await service.accept(sources(corpus)))

    assert settled.counts["indexed"] == len(adapter.points)
    assert adapter.points


async def test_the_collection_is_created(tmp_path: Path, corpus: Path) -> None:
    service, adapter = build(tmp_path)

    await service.run(await service.accept(sources(corpus)))

    assert adapter.specs


async def test_the_job_is_settled_in_the_journal(tmp_path: Path, corpus: Path) -> None:
    """A job left 'running' after the process exits is indistinguishable from a hung one."""
    service, _ = build(tmp_path)

    record = await service.accept(sources(corpus))
    settled = await service.run(record)

    assert service.journal.load_job(record.job_id).status == settled.status


async def test_counts_add_up(tmp_path: Path, corpus: Path) -> None:
    service, _ = build(tmp_path)

    counts = (await service.run(await service.accept(sources(corpus)))).counts

    assert counts["total"] == 2
    assert counts["parsed"] + counts["deduplicated"] + counts["dead_lettered"] == counts["total"]


async def test_reingesting_the_same_corpus_deduplicates(tmp_path: Path, corpus: Path) -> None:
    """D3: replaying a job must be a no-op, not a second copy of every chunk."""
    service, adapter = build(tmp_path)

    await service.run(await service.accept(sources(corpus)))
    written = len(adapter.points)
    second = await service.run(await service.accept(sources(corpus)))

    assert second.counts["deduplicated"] == 2
    assert len(adapter.points) == written


async def test_an_unreadable_source_is_dead_lettered(tmp_path: Path, corpus: Path) -> None:
    """A corrupt file becomes an entry with a reason code; the job keeps going."""
    service, _ = build(tmp_path)

    record = await service.accept([*sources(corpus), str(corpus / "absent.md")])
    settled = await service.run(record)

    assert settled.counts["dead_lettered"] == 1
    assert settled.counts["parsed"] == 2
    assert service.journal.dead_lettered(record.job_id)


async def test_a_failed_embedding_dead_letters_rather_than_losing_the_job(
    tmp_path: Path, corpus: Path
) -> None:
    """The provider failing must not leave documents silently unindexed and unreported."""
    configured = settings()
    embedder = StubEmbedder(configured)
    embedder.error = EmbedError("provider down", retryable=True)
    service, _ = build(tmp_path, configured, embedder=embedder)

    record = await service.accept(sources(corpus))
    settled = await service.run(record)

    assert settled.counts["indexed"] == 0
    assert settled.counts["dead_lettered"] > 0
    assert settled.status != "completed"


async def test_the_cache_is_invalidated_when_the_corpus_changes(
    tmp_path: Path, corpus: Path
) -> None:
    """A cached answer describes the corpus as it was; indexing is what makes it wrong."""
    cache = RecordingCache()
    service, _ = build(tmp_path, cache=cache)

    await service.run(await service.accept(sources(corpus)))

    assert cache.reasons


async def test_the_cache_is_left_alone_when_nothing_was_indexed(
    tmp_path: Path, corpus: Path
) -> None:
    """Throwing away a warm cache for a job that changed nothing is pure cost."""
    cache = RecordingCache()
    service, _ = build(tmp_path, cache=cache)
    await service.run(await service.accept(sources(corpus)))
    cache.reasons.clear()

    await service.run(await service.accept(sources(corpus)))

    assert cache.reasons == []


async def test_resuming_skips_what_a_checkpoint_already_covered(
    tmp_path: Path, corpus: Path
) -> None:
    """Re-running after a crash continues rather than restarting (D3)."""
    service, _ = build(tmp_path)
    record = await service.accept(sources(corpus))
    await service.run(record)

    resumed = await service.run(service.journal.load_job(record.job_id))

    assert resumed.counts["parsed"] == 0


async def test_ingestion_metrics_are_published(tmp_path: Path, corpus: Path) -> None:
    """A counter written nowhere reads identically to nothing having been ingested."""
    before = metrics.INGEST_DOCUMENTS.value(status="succeeded")
    service, _ = build(tmp_path)

    await service.run(await service.accept(sources(corpus)))

    assert metrics.INGEST_DOCUMENTS.value(status="succeeded") > before


async def test_metadata_reaches_the_indexed_points(tmp_path: Path, corpus: Path) -> None:
    service, adapter = build(tmp_path)

    await service.run(await service.accept(sources(corpus)), metadata={"team": "finance"})

    assert any("finance" in str(point.payload) for point in adapter.points)


async def test_a_tenant_is_carried_onto_the_job(tmp_path: Path, corpus: Path) -> None:
    service, _ = build(tmp_path)

    record = await service.accept(sources(corpus), tenant="acme")

    assert record.tenant == "acme"


async def test_replaying_an_idempotency_key_returns_the_first_job(
    tmp_path: Path, corpus: Path
) -> None:
    """Otherwise a retried request ingests the corpus twice."""
    service, _ = build(tmp_path)

    first = await service.accept(sources(corpus), idempotency_key="abc")
    second = await service.accept(sources(corpus), idempotency_key="abc")

    assert first.job_id == second.job_id
