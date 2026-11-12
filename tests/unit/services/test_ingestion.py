"""The ingestion service's ``run`` orchestration, without a vector database.

``IngestionService.run`` is the primary write path — source resolution, both worker pools,
journal checkpointing, dead-lettering, the lockfile, cache invalidation, and the ingestion
metrics. It was covered only by a Docker-gated integration test, so on any machine without a
daemon the whole path went unexecuted and CI's unit leg never touched it.

The vector database is faked here rather than run. What is being tested is the orchestration
between the pieces, and a real backend adds a daemon dependency without exercising any of it.

The fakes report each collection's keyword leg from the spec it was created with, because a
fake that always answered ``sparse=False`` would make the acceptance preflight (TASK-0320)
fire on every re-ingest, and one that always answered ``True`` could not show it firing at
all. ``tests/unit/services/test_keyword_leg_write_paths.py`` holds the same cases against a
live backend, skipped wherever Docker is not running.
"""

from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import CollectionInfo, CollectionSpec, Point, UpsertResult
from fasterrag.config.schema import Settings
from fasterrag.core import chunking
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError
from fasterrag.observability import metrics
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from fasterrag.workers import cpu_pool
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


class KeywordLegAdapter(RecordingAdapter):
    """Reports each collection's keyword leg from the spec it was created with."""

    async def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(name=spec.name, vectors=len(self.points), sparse=spec.sparse)
            for spec in self.specs
        ]


class DenseOnlyAdapter(KeywordLegAdapter):
    """A backend already holding a collection that was indexed without a keyword leg.

    Creating over it leaves the layout alone and ``upsert`` raises what the Qdrant adapter
    raises for a sparse point on a dense-only collection — the failure acceptance pre-empts.
    """

    def __init__(self, settings: Settings, *, collection: str = "default") -> None:
        super().__init__(settings)
        self.specs.append(CollectionSpec(name=collection, dimensions=DIMENSIONS, sparse=False))

    async def create_collection(self, spec: CollectionSpec) -> None:
        if any(existing.name == spec.name for existing in self.specs):
            return
        await super().create_collection(spec)

    async def upsert(self, points: list[Point]) -> UpsertResult:
        for point in points:
            if point.sparse is not None:
                raise EmbedError(
                    f"point {point.point_id!r} carries a sparse vector but collection "
                    f"{point.collection!r} has no sparse index; recreate it with sparse "
                    "enabled",
                    retryable=False,
                )
        return await super().upsert(points)


class CountingAdapter(KeywordLegAdapter):
    """Counts the metadata round trips the preflight costs."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.listings = 0

    async def list_collections(self) -> list[CollectionInfo]:
        self.listings += 1
        return await super().list_collections()


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
    adapter: RecordingAdapter | None = None,
    **kwargs: Any,
) -> tuple[IngestionService, RecordingAdapter]:
    """Return a service and the adapter it writes through."""
    resolved = configured or settings()
    backend = adapter or KeywordLegAdapter(resolved)
    service = IngestionService(
        resolved,
        journal=Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=backend,
        router=TieringRouter(embedder or StubEmbedder(resolved)),
        executor_factory=threads,
        **kwargs,
    )
    return service, backend


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

    assert service.journal.load_job_blocking(record.job_id).status == settled.status


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
    assert service.journal.dead_lettered_blocking(record.job_id)


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

    resumed = await service.run(service.journal.load_job_blocking(record.job_id))

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


async def test_a_dense_only_collection_is_refused_at_acceptance(
    tmp_path: Path, corpus: Path
) -> None:
    """TASK-0320: the CLI and library paths were accepted and then died at first upsert."""
    service, _ = build(tmp_path, adapter=DenseOnlyAdapter(settings()))

    with pytest.raises(FasterRagError) as refused:
        await service.accept(sources(corpus))

    assert refused.value.code is ErrorCode.CONFLICT
    assert "'default'" in refused.value.detail
    assert "fasterrag index reembed" in refused.value.detail
    assert "retrieval.hybrid" in refused.value.detail


async def test_nothing_is_recorded_for_a_refused_job(tmp_path: Path, corpus: Path) -> None:
    """A job record for a run that cannot start is a queued job nobody will ever settle."""
    service, _ = build(tmp_path, adapter=DenseOnlyAdapter(settings()))

    with pytest.raises(FasterRagError):
        await service.accept(sources(corpus))

    assert not list(service.journal.root.rglob("*.json"))


async def test_the_refusal_replaces_the_embed_provider_error_the_operator_used_to_get(
    tmp_path: Path, corpus: Path
) -> None:
    """The old failure names the embedding provider and a point id for an index mistake.

    A job that bypasses ``accept`` no longer raises out of ``run``: a refused write is a
    per-document failure now, so every document is dead-lettered and the job settles failed.
    """
    service, _ = build(tmp_path, adapter=DenseOnlyAdapter(settings()))
    bypassed = service.journal.create_job_blocking(
        "default", [{"type": "path", "value": source} for source in sources(corpus)]
    )

    settled = await service.run(bypassed)
    dead = await service.journal.dead_lettered(bypassed.job_id)

    assert settled.status == "failed"
    assert {entry.reason_code for entry in dead} == {ErrorCode.EMBED_PROVIDER_ERROR.value}
    assert all("fasterrag index reembed" not in (entry.detail or "") for entry in dead)

    with pytest.raises(FasterRagError) as new:
        await service.accept(sources(corpus))

    assert new.value.code is ErrorCode.CONFLICT


async def test_a_library_ingest_is_refused_before_a_document_is_read(
    tmp_path: Path, corpus: Path
) -> None:
    """``ingest`` is what the facade and ``fasterrag ingest`` call, and it goes through accept."""
    service, adapter = build(tmp_path, adapter=DenseOnlyAdapter(settings()))

    with pytest.raises(FasterRagError) as refused:
        await service.ingest(sources(corpus))

    assert refused.value.code is ErrorCode.CONFLICT
    assert adapter.points == []


async def test_a_collection_nothing_has_indexed_yet_is_accepted(
    tmp_path: Path, corpus: Path
) -> None:
    """The first ingest creates the collection with its keyword leg; it cannot mismatch."""
    service, _ = build(tmp_path)

    assert (await service.accept(sources(corpus))).status == "queued"


async def test_a_dense_only_collection_is_accepted_when_hybrid_is_off(
    tmp_path: Path, corpus: Path
) -> None:
    """Dense-only is a valid configuration, not a fault; only the mismatch is refused."""
    configured = settings(retrieval={"hybrid": False})
    service, _ = build(tmp_path, configured, adapter=DenseOnlyAdapter(configured))

    settled = await service.run(await service.accept(sources(corpus)))

    assert settled.status == "completed"


async def test_the_preflight_costs_one_round_trip_per_job_not_per_document(
    tmp_path: Path, corpus: Path
) -> None:
    """Per job it is noise beside a corpus; per document it would be a tax on every one."""
    adapter = CountingAdapter(settings())
    service, _ = build(tmp_path, adapter=adapter)

    await service.accept(sources(corpus))
    accepted = adapter.listings
    await service.run(await service.accept(sources(corpus)))

    assert accepted == 1
    assert adapter.listings == 2


async def test_a_document_that_failed_to_embed_is_not_remembered_as_indexed(
    tmp_path: Path, corpus: Path
) -> None:
    """The retry-is-a-no-op defect: a failed document used to be recorded as indexed.

    Its content hash went into the dedup file when its chunks were enqueued, so the retry
    found it there, reported it deduplicated, and the document was never indexed at all.
    """
    embedder = StubEmbedder(settings())
    embedder.error = EmbedError("the provider is down", retryable=False)
    service, adapter = build(tmp_path, embedder=embedder)

    first = await service.run(await service.accept(sources(corpus)))
    embedder.error = None
    second = await service.run(await service.accept(sources(corpus)))

    assert first.status == "failed"
    assert first.counts["dead_lettered"] == 2
    assert second.status == "completed"
    assert second.counts["deduplicated"] == 0
    assert adapter.points


async def test_a_failed_document_is_dead_lettered_once_not_once_per_chunk(
    tmp_path: Path, corpus: Path
) -> None:
    embedder = StubEmbedder(settings())
    embedder.error = EmbedError("the provider is down", retryable=False)
    service, _ = build(tmp_path, embedder=embedder)

    record = await service.accept(sources(corpus))
    settled = await service.run(record)
    dead = await service.journal.dead_lettered(record.job_id)

    assert settled.counts["chunked"] >= 2
    assert sorted(entry.source for entry in dead) == sorted(sources(corpus))


class DenseTokenCounter:
    """Every non-space character is a token, as CJK very nearly is."""

    def count(self, text: str) -> int:
        return len(text.strip())

    @property
    def chars_per_token(self) -> int:
        return 4


class ShiftingSentenceEmbedder:
    """Moves to a new topic every fifteenth sentence, so similarity alone makes long groups."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if index // 15 % 2 else [0.0, 1.0] for index in range(len(texts))]


async def test_a_semantic_job_with_a_local_model_completes_within_the_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic chunking was refused at acceptance: the parse workers had no embedding model.

    The counter makes every character a token, so a group of sentences merged up to the
    character window is four times the budget unless the budget pass re-splits it.
    """
    counter = DenseTokenCounter()
    monkeypatch.setattr(chunking, "create_token_counter", lambda _settings: counter)
    cpu_pool.use_sentence_embedder_factory(lambda _settings: ShiftingSentenceEmbedder())
    sentences = ["汉字" * (5 + index % 9) for index in range(40)]
    document = tmp_path / "dense.txt"
    document.write_text(". ".join(sentences) + ".", encoding="utf-8")
    configured = settings(chunking={"strategy": "semantic", "chunk_size": 64, "overlap": 8})
    service, adapter = build(tmp_path, configured)

    settled = await service.run(await service.accept([str(document)]))

    texts = [str(point.payload["text"]) for point in adapter.points]
    assert settled.status == "completed"
    assert settled.counts["indexed"] == len(texts) > 1
    assert all(counter.count(text) <= 64 + 8 for text in texts)


async def test_semantic_chunking_with_a_hosted_provider_is_refused_before_a_job_is_recorded(
    tmp_path: Path, corpus: Path
) -> None:
    """A hosted provider's client cannot run in a parse worker; every document would fail."""
    configured = settings(
        chunking={"strategy": "semantic", "chunk_size": 128, "overlap": 16},
        embeddings={"provider": "openai", "model": "text-embedding-3-small", "api_key_env": "K"},
    )
    service, _ = build(tmp_path, configured)

    with pytest.raises(FasterRagError) as refused:
        await service.accept(sources(corpus))

    assert refused.value.code is ErrorCode.CONFIG_INVALID
    assert "needs a local embedding model" in refused.value.detail
    assert "Set embeddings.provider to 'huggingface'" in refused.value.detail
    assert not list(service.journal.root.rglob("*.json"))


async def test_an_unexpected_failure_settles_the_job_as_failed(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job left 'running' by an untyped exception is indistinguishable from a slow one."""
    service, adapter = build(tmp_path)

    async def explode(points: list[Point]) -> UpsertResult:
        raise RuntimeError("a bug in the adapter")

    monkeypatch.setattr(adapter, "upsert", explode)
    record = await service.accept(sources(corpus))

    with pytest.raises(FasterRagError) as stopped:
        await service.run(record)

    assert stopped.value.code is ErrorCode.INTERNAL
    assert service.journal.load_job_blocking(record.job_id).status == "failed"


async def test_a_source_that_cannot_be_resolved_settles_the_job_as_failed(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    record = service.journal.create_job_blocking(
        "default", [{"type": "url", "value": "ftp://example.invalid/file.md"}]
    )

    with pytest.raises(FasterRagError):
        await service.run(record)

    assert service.journal.load_job_blocking(record.job_id).status == "failed"


@pytest.fixture
def empties(corpus: Path) -> list[str]:
    """Return a zero-byte Markdown file and a whitespace-only text file."""
    (corpus / "blank.md").write_bytes(b"")
    (corpus / "spaces.txt").write_text("  \n\t\n   \n", encoding="utf-8")
    return [str(corpus / "blank.md"), str(corpus / "spaces.txt")]


async def test_an_empty_document_is_skipped_rather_than_recorded_as_indexed(
    tmp_path: Path, corpus: Path, empties: list[str]
) -> None:
    """A document with no chunks was journalled 'indexed' though the index never held it."""
    service, adapter = build(tmp_path)

    record = await service.accept([*sources(corpus), *empties])
    settled = await service.run(record)

    indexed = service.journal.documents_blocking(record.job_id, status="indexed")
    skipped = service.journal.documents_blocking(record.job_id, status="skipped")
    assert sorted(entry.source for entry in indexed) == sorted(sources(corpus))
    assert sorted(entry.source for entry in skipped) == sorted(empties)
    assert {entry.reason_code for entry in skipped} == {"EMPTY_DOCUMENT"}
    assert settled.status == "completed"
    assert settled.counts["skipped"] == 2
    assert settled.counts["parsed"] == 2
    assert settled.counts["dead_lettered"] == 0
    assert {point.payload["source_uri"] for point in adapter.points} == set(sources(corpus))


async def test_an_empty_document_never_enters_the_dedup_file(
    tmp_path: Path, corpus: Path, empties: list[str]
) -> None:
    """The hash of empty content was remembered, so every later empty file 'deduplicated'."""
    service, _ = build(tmp_path)

    first = await service.run(await service.accept(empties[:1]))
    known = service.journal.known_content_blocking("default")
    second_record = await service.accept(empties[:1])
    second = await service.run(second_record)

    assert known == {}
    assert service.journal.known_content_blocking("default") == {}
    assert first.counts["skipped"] == 1
    assert second.counts["skipped"] == 1
    assert second.counts["deduplicated"] == 0
    skipped = service.journal.documents_blocking(second_record.job_id, status="skipped")
    assert [entry.source for entry in skipped] == empties[:1]


async def test_a_job_of_nothing_but_empty_documents_completes(
    tmp_path: Path, empties: list[str]
) -> None:
    """A skipped document is not a failure: nothing was lost, there was nothing to index."""
    service, adapter = build(tmp_path)

    record = await service.accept(empties)
    settled = await service.run(record)

    assert settled.status == "completed"
    assert settled.counts["total"] == settled.counts["skipped"] == 2
    assert settled.checkpoint is not None
    assert settled.checkpoint.last_document_index == 1
    assert adapter.points == []
    assert service.journal.dead_lettered_blocking(record.job_id) == []


async def test_skipped_documents_are_counted_apart_from_indexed_ones(
    tmp_path: Path, corpus: Path, empties: list[str]
) -> None:
    """A skipped document was published as a succeeded one in the ingestion metrics."""
    service, _ = build(tmp_path)
    succeeded = metrics.INGEST_DOCUMENTS.value(status="succeeded")
    skipped = metrics.INGEST_DOCUMENTS.value(status="skipped")

    await service.run(await service.accept([*sources(corpus), *empties]))

    assert metrics.INGEST_DOCUMENTS.value(status="succeeded") - succeeded == 2
    assert metrics.INGEST_DOCUMENTS.value(status="skipped") - skipped == 2
