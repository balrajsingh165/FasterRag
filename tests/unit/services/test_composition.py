"""The composition root: one embedding router for every entry point, cached when configured.

``CachingEmbeddingAdapter`` existed, was tested, and was constructed by nothing (TASK-0442),
so ``embeddings.cache.enabled`` validated, defaulted to ``true`` and changed no behaviour.
These tests go through the builder the API, the CLI, the facade and the ingestion service
all use, with only the provider replaced, so what they prove is the wiring and not the
wrapper.
"""

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings import tiering
from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core import cache as cache_package
from fasterrag.core.cache import (
    DEFAULT_CACHE_ROOT,
    EMBEDDING_CACHE_ROOT,
    CachingEmbeddingAdapter,
    MemoryStore,
    create_semantic_store,
)
from fasterrag.core.cache.store import CacheStore
from fasterrag.services import composition
from fasterrag.services.composition import create_embedding_router
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.workers.embed_pool import EmbeddingWorkerPool
from tests.unit.services.test_ingestion import KeywordLegAdapter, threads
from tests.unit.workers.test_late_chunking import Sink, payload

DIMENSIONS = 8
SRC = Path(__file__).resolve().parents[3] / "src" / "fasterrag"
SHIPPED_CACHE_ROOT = EMBEDDING_CACHE_ROOT


def test_the_shipped_embedding_cache_root_is_a_directory_inside_the_cache_root() -> None:
    assert SHIPPED_CACHE_ROOT == DEFAULT_CACHE_ROOT / "embedding"
    assert SHIPPED_CACHE_ROOT.parts == (".fasterrag", "cache", "embedding")


class Calls:
    """Every text a provider was asked to embed, across however many adapters were built."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.closed: list[str] = []


class CountingProvider(EmbeddingAdapter):
    """A provider that records what reaches it and embeds each text deterministically."""

    provider = "counting"

    def __init__(self, settings: Settings, calls: Calls) -> None:
        super().__init__(settings)
        self.calls = calls

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def model_version(self) -> str:
        return f"{self.config.model}-v1"

    @property
    def dimensions(self) -> int | None:
        return DIMENSIONS

    def vector(self, text: str) -> list[float]:
        return [float((sum(map(ord, text)) + offset) % 17) for offset in range(DIMENSIONS)]

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.documents.extend(texts)
        return EmbeddingResult(
            vectors=[self.vector(text) for text in texts],
            model=self.model,
            model_version=self.model_version,
            total_tokens=len(texts),
        )

    async def embed_query(self, text: str) -> list[float]:
        self.calls.queries.append(text)
        return self.vector(text)

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        self.calls.closed.append(self.model)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Calls:
    """Replace the provider, and run in a directory of the test's own.

    The default cache backend is ``disk`` under the working directory, exactly as a user
    gets it, so each test needs a working directory nothing else has written to — and the
    shipped location put back, since ``tests/conftest.py`` moves it for every other test.
    """
    recorded = Calls()
    monkeypatch.setattr(
        tiering, "create_embedding_adapter", lambda settings: CountingProvider(settings, recorded)
    )
    monkeypatch.setattr(cache_package, "EMBEDDING_CACHE_ROOT", SHIPPED_CACHE_ROOT)
    monkeypatch.chdir(tmp_path)
    return recorded


@pytest.fixture
def corpus(tmp_path: Path) -> list[str]:
    root = tmp_path / "corpus"
    root.mkdir()
    documents = {
        "expenses.md": "# Expenses\n\nThe meal allowance is 41 pounds per day for UK travel.\n",
        "leave.md": "# Leave\n\nParental leave is 26 weeks at full pay.\n",
        "travel.md": "# Travel\n\nBook rail before air for any journey under four hours.\n",
    }
    for name, text in documents.items():
        (root / name).write_text(text, encoding="utf-8")
    return [str(root / name) for name in documents]


def settings(**embeddings: Any) -> Settings:
    return Settings.model_validate(
        {
            "embeddings": {"batch_size": 8, **embeddings},
            "ingestion": {"dedup": False},
            "workers": {"embedding_pool_size": 1, "queue_depth": 200},
            "chunking": {"chunk_size": 128, "overlap": 16},
            "reliability": {"retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False}},
        }
    )


async def ingest(
    configured: Settings, sources: list[str], journal: Path
) -> tuple[JobRecord, KeywordLegAdapter]:
    """Run one ingest the way a fresh process would: a new service, building its own router."""
    backend = KeywordLegAdapter(configured)
    service = IngestionService(
        configured,
        journal=Journal(journal, checkpoint_every=1),
        adapter=backend,
        executor_factory=threads,
    )
    try:
        settled = await service.run(await service.accept(sources))
    finally:
        await service.router.close()
    return settled, backend


async def test_reingesting_an_unchanged_corpus_with_dedup_off_makes_no_provider_calls(
    tmp_path: Path, corpus: list[str], calls: Calls
) -> None:
    """Nothing built the embedding cache, so a re-ingest paid to embed every chunk again."""
    configured = settings()

    first, written = await ingest(configured, corpus, tmp_path / "journal")
    embedded = list(calls.documents)
    calls.documents.clear()

    second, rewritten = await ingest(configured, corpus, tmp_path / "journal")

    assert first.status == second.status == "completed"
    assert len(embedded) == first.counts["indexed"] > 0
    assert second.counts["indexed"] == first.counts["indexed"]
    assert calls.documents == []
    assert {point.point_id: point.vector for point in rewritten.points} == {
        point.point_id: pytest.approx(point.vector) for point in written.points
    }


async def test_the_cache_is_not_used_when_it_is_disabled(
    tmp_path: Path, corpus: list[str], calls: Calls
) -> None:
    """The switch must be read: off means every chunk reaches the provider every time."""
    configured = settings(cache={"enabled": False})

    first, _ = await ingest(configured, corpus, tmp_path / "journal")
    embedded = list(calls.documents)
    calls.documents.clear()
    await ingest(configured, corpus, tmp_path / "journal")

    assert len(embedded) == first.counts["indexed"] > 0
    assert sorted(calls.documents) == sorted(embedded)
    assert not (tmp_path / ".fasterrag" / "cache").exists()


def test_a_disabled_cache_leaves_the_adapters_unwrapped(calls: Calls) -> None:
    router = create_embedding_router(settings(cache={"enabled": False}))

    assert isinstance(router.default, CountingProvider)


def test_an_enabled_cache_wraps_the_default_adapter(calls: Calls) -> None:
    router = create_embedding_router(settings())

    assert isinstance(router.default, CachingEmbeddingAdapter)
    assert isinstance(router.default.inner, CountingProvider)


async def test_a_repeated_question_reaches_the_provider_once_across_two_processes(
    calls: Calls,
) -> None:
    configured = settings()

    first = create_embedding_router(configured)
    asked = await first.default.embed_query("what is the meal allowance?")
    await first.close()
    second = create_embedding_router(configured)
    again = await second.default.embed_query("what is the meal allowance?")
    await second.close()

    assert calls.queries == ["what is the meal allowance?"]
    assert again == pytest.approx(asked)


async def test_the_default_cache_lives_in_its_own_directory(
    tmp_path: Path, corpus: list[str], calls: Calls
) -> None:
    """Two disk caches in one directory would evict and clear each other's entries."""
    await ingest(settings(), corpus, tmp_path / "journal")

    root = tmp_path / ".fasterrag" / "cache"
    assert len(list((root / "embedding").glob("*.bin"))) == len(calls.documents)
    assert list(root.glob("*.bin")) == []


async def test_invalidating_a_disk_semantic_cache_leaves_the_embedding_cache_alone(
    tmp_path: Path, corpus: list[str], calls: Calls
) -> None:
    """Every ingest clears the semantic cache; sharing a directory would clear the vectors too."""
    configured = Settings.model_validate(
        {**settings().model_dump(mode="json"), "cache": {"backend": "disk"}}
    )
    semantic = create_semantic_store(configured)
    await semantic.set("sem:-:question", b"answer")

    await ingest(configured, corpus, tmp_path / "journal")
    assert [key for key, _ in await semantic.items()] != []
    await semantic.clear()
    calls.documents.clear()
    await ingest(configured, corpus, tmp_path / "journal")

    assert await semantic.items() == []
    assert calls.documents == []


def tiered(**cache: Any) -> Settings:
    return settings(
        model="default-model",
        cache=cache,
        tiering={
            "enabled": True,
            "rules": [
                {"match": {"priority_class": "archive"}, "provider": "ollama", "model": "cheap"},
                {"match": {"priority_class": "legal"}, "provider": "ollama", "model": "precise"},
            ],
        },
    )


async def test_every_tier_is_cached_and_no_tier_is_served_another_tier_s_vector(
    calls: Calls,
) -> None:
    router = create_embedding_router(tiered(backend="memory"))
    cheap = router.select({"priority_class": "archive"})
    precise = router.select({"priority_class": "legal"})

    for adapter in (router.default, cheap, precise):
        await adapter.embed_documents(["the same text"])
        await adapter.embed_documents(["the same text"])

    assert [type(adapter) for adapter in router.adapters()] == [CachingEmbeddingAdapter] * 3
    assert [adapter.model for adapter in router.adapters()] == [
        "default-model",
        "cheap",
        "precise",
    ]
    assert calls.documents == ["the same text"] * 3


class ClosingStore(MemoryStore):
    """Counts how many times it was released."""

    def __init__(self) -> None:
        super().__init__()
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1


async def test_closing_the_router_releases_every_provider_and_the_shared_store_once(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One store backs every tier; each wrapper closing it would release a pool three times."""
    store = ClosingStore()
    built: list[Settings] = []

    def one_store(configured: Settings) -> CacheStore:
        built.append(configured)
        return store

    monkeypatch.setattr(composition, "create_embedding_store", one_store)
    router = create_embedding_router(tiered())

    await router.close()

    assert len(built) == 1
    assert sorted(calls.closed) == ["cheap", "default-model", "precise"]
    assert store.closes == 1


class PoolingProvider(CountingProvider):
    """A provider that can late-chunk a document, as the local model adapter can."""

    def __init__(self, settings: Settings, calls: Calls) -> None:
        super().__init__(settings, calls)
        self.pooled: list[str] = []

    async def embed_pooled(
        self, document: str, spans: Sequence[tuple[int, int]]
    ) -> EmbeddingResult:
        self.pooled.append(document)
        return EmbeddingResult(
            vectors=[[float(start)] * DIMENSIONS for start, _ in spans],
            model=self.model,
            model_version=self.model_version,
        )


async def test_late_chunking_survives_the_cache_and_is_never_served_from_it(
    tmp_path: Path, calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behind a plain wrapper the pool saw no pooling adapter and embedded chunk by chunk."""
    built: list[PoolingProvider] = []

    def provider(configured: Settings) -> PoolingProvider:
        built.append(PoolingProvider(configured, calls))
        return built[-1]

    monkeypatch.setattr(tiering, "create_embedding_adapter", provider)
    configured = Settings.model_validate({"chunking": {"strategy": "late"}})
    group = [payload("doc", index, index * 10, index * 10 + 10) for index in range(3)]

    for _ in range(2):
        sink = Sink()
        router = create_embedding_router(configured)
        await EmbeddingWorkerPool(configured, router, sink)._flush(group, 0)
        await router.close()

    assert [inner.pooled for inner in built] == [["the whole document"]] * 2
    assert calls.documents == []
    assert sink.batches[0].vectors == [[0.0] * DIMENSIONS, [10.0] * DIMENSIONS, [20.0] * DIMENSIONS]
    assert not (tmp_path / ".fasterrag" / "cache").exists()


def _router_imports(tree: ast.AST) -> list[str]:
    """Return the modules a file imports ``create_embedding_router`` from."""
    return [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "create_embedding_router" for alias in node.names)
    ]


def test_nothing_outside_the_adapters_builds_a_router_that_bypasses_the_cache() -> None:
    """An entry point importing the bare builder gets a router the cache switch cannot reach."""
    allowed = {SRC / "services" / "composition.py"}
    offenders: Mapping[str, list[str]] = {
        str(path.relative_to(SRC)): modules
        for path in sorted(SRC.rglob("*.py"))
        if path not in allowed and (SRC / "adapters") not in path.parents
        if (modules := _router_imports(ast.parse(path.read_text(encoding="utf-8"))))
        and any(module.startswith("fasterrag.adapters") for module in modules)
    }

    assert offenders == {}


def test_the_guard_above_finds_the_entry_points_it_is_guarding() -> None:
    """A scan that matched no file at all would pass whatever the entry points imported."""
    importers = [
        path
        for path in SRC.rglob("*.py")
        if "fasterrag.services.composition"
        in _router_imports(ast.parse(path.read_text(encoding="utf-8")))
    ]

    assert len(importers) >= 8
