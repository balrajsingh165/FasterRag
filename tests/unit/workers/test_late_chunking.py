"""Wiring for late chunking: carrying the document, routing to pooling, and ordering."""

from collections.abc import Sequence
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingResult
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import Segment, TextChunk
from fasterrag.workers.embed_pool import EmbeddingWorkerPool, PoolingAdapter
from fasterrag.workers.queues import ChunkPayload, EmbeddedBatch


def settings(strategy: str = "late") -> Settings:
    return Settings.model_validate({"chunking": {"strategy": strategy}})


def payload(document: str, index: int, start: int, end: int, text: str = "body") -> ChunkPayload:
    return ChunkPayload(
        chunk_id=f"{document}-{index}",
        document_id=document,
        source=f"{document}.md",
        content_hash="hash",
        chunk=TextChunk(
            text=text, start=start, end=end, chunk_index=index, token_count=1, strategy="late"
        ),
        document_text="the whole document",
    )


class Pooling:
    """Records what it was asked to pool, returning a vector naming each span."""

    provider = "fake"

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.spans: list[Sequence[Segment]] = []

    async def embed_pooled(self, document: str, spans: Sequence[Segment]) -> EmbeddingResult:
        self.documents.append(document)
        self.spans.append(list(spans))
        return EmbeddingResult(
            vectors=[[float(start)] for start, _ in spans], model="fake", model_version="1"
        )

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=[[-1.0] for _ in texts], model="fake", model_version="1")


class Plain:
    """A hosted-style adapter with no token-level output."""

    provider = "fake"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        self.texts.extend(texts)
        return EmbeddingResult(vectors=[[-1.0] for _ in texts], model="fake", model_version="1")


class Router:
    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def select(self, metadata: Any) -> Any:
        return self.adapter


class Sink:
    def __init__(self) -> None:
        self.batches: list[EmbeddedBatch] = []

    async def write(self, batch: EmbeddedBatch) -> None:
        self.batches.append(batch)


async def flush(adapter: Any, group: list[ChunkPayload], strategy: str = "late") -> Sink:
    sink = Sink()
    pool = EmbeddingWorkerPool(settings(strategy), Router(adapter), sink)  # type: ignore[arg-type]
    await pool._flush(group, 0)
    return sink


def test_a_pooling_adapter_satisfies_the_protocol() -> None:
    assert isinstance(Pooling(), PoolingAdapter)


def test_a_plain_adapter_does_not() -> None:
    assert not isinstance(Plain(), PoolingAdapter)


async def test_late_chunks_go_through_one_pass_per_document() -> None:
    """One pass per document is the whole feature; one per chunk would be ordinary."""
    adapter = Pooling()

    await flush(adapter, [payload("doc", index, index * 10, index * 10 + 10) for index in range(4)])

    assert len(adapter.documents) == 1
    assert adapter.spans[0] == [(0, 10), (10, 20), (20, 30), (30, 40)]


async def test_two_documents_get_a_pass_each() -> None:
    adapter = Pooling()

    await flush(adapter, [payload("a", 0, 0, 10), payload("b", 0, 0, 10), payload("a", 1, 10, 20)])

    assert len(adapter.documents) == 2
    assert [len(spans) for spans in adapter.spans] == [2, 1]


async def test_every_chunk_keeps_its_own_vector() -> None:
    """Interleaved documents are regrouped, so the write order must follow the vectors."""
    adapter = Pooling()

    sink = await flush(
        adapter, [payload("a", 0, 0, 10), payload("b", 0, 50, 60), payload("a", 1, 10, 20)]
    )

    batch = sink.batches[0]
    for chunk, vector in zip(batch.chunks, batch.vectors, strict=True):
        assert vector == [float(chunk.chunk.start)]


async def test_a_hosted_adapter_falls_back_to_ordinary_embedding() -> None:
    """`strategy: late` on a hosted model must be no better, not broken."""
    adapter = Plain()

    sink = await flush(adapter, [payload("doc", 0, 0, 10, text="chunk body")])

    assert adapter.texts == ["chunk body"]
    assert len(sink.batches[0].vectors) == 1


async def test_another_strategy_does_not_pool() -> None:
    adapter = Pooling()

    await flush(adapter, [payload("doc", 0, 0, 10)], strategy="recursive")

    assert adapter.documents == []


async def test_chunks_without_their_document_fall_back() -> None:
    """A chunk that reached the pool without its text must not pool against an empty one."""
    adapter = Pooling()
    orphan = payload("doc", 0, 0, 10)
    stripped = ChunkPayload(
        chunk_id=orphan.chunk_id,
        document_id=orphan.document_id,
        source=orphan.source,
        content_hash=orphan.content_hash,
        chunk=orphan.chunk,
    )

    await flush(adapter, [stripped])

    assert adapter.documents == []


async def test_the_pooled_batch_reports_the_model() -> None:
    sink = await flush(Pooling(), [payload("doc", 0, 0, 10)])

    assert sink.batches[0].model == "fake"
    assert sink.batches[0].model_version == "1"


@pytest.mark.parametrize("strategy", ["late", "recursive"])
def test_the_document_is_carried_only_for_late(strategy: str) -> None:
    """Carrying it always would double what every parsed document sends through IPC."""
    from fasterrag.workers.cpu_pool import parse_and_chunk

    configured = Settings.model_validate(
        {"chunking": {"strategy": strategy, "chunk_size": 96, "overlap": 8}}
    )
    from fasterrag.workers.cpu_pool import CpuWorkerPool

    task = CpuWorkerPool.tasks_for(["tests/eval/datasets/policies/corpus/expenses-uk-2026.md"])[0]

    outcome = parse_and_chunk(task, configured)

    assert bool(outcome.document_text) is (strategy == "late")
