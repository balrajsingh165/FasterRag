"""A query against an index another embedding model built (TASK-0477).

``detect_drift`` ran only for ``index list`` and ``index lock verify``; the query path never
read the lock. Change ``embeddings.model`` to another model of the same dimension and every
query compared vectors from two embedding spaces and returned confident, wrong results.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, RetrievalError
from fasterrag.services import lockfile
from fasterrag.services.lockfile import LockStore, build_lock, verify_query_lock
from fasterrag.services.querying import RetrievalService
from tests.unit.services.test_querying import ScriptedAdapter, StubEmbedder, hit

COLLECTION = "handbook"


class ConfiguredEmbedder(StubEmbedder):
    """Reports whatever model ``embeddings.model`` names, as the built-in adapters do."""

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def model_version(self) -> str:
        return self.config.model

    async def embed_documents(self, texts: Any) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[0.1, 0.2, 0.3] for _ in texts], model=self.model, model_version=self.model
        )


def configured(model: str, mode: str = "refuse", **overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "embeddings": {"provider": "huggingface", "model": model},
        "retrieval": {"rerank": False, "hybrid": False, "verify_lock": mode},
        **overrides,
    }
    return Settings.model_validate(payload)


def index_with(model: str) -> None:
    """Write the lock an ingest under ``model`` leaves behind, in the working directory."""
    built = configured(model)
    LockStore().write(
        build_lock(
            COLLECTION, built, embedding_model=model, embedding_model_version=model, dimensions=3
        )
    )


def service(settings: Settings) -> RetrievalService:
    adapter = ScriptedAdapter(settings)
    adapter.dense = [hit("p1", 0.9)]
    return RetrievalService(settings, adapter, TieringRouter(ConfiguredEmbedder(settings)))


@pytest.fixture(autouse=True)
def _in_a_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


async def test_a_query_under_a_changed_embedding_model_is_refused() -> None:
    """The index was built by model-a; the configuration now says model-b, same dimension."""
    index_with("org/model-a")

    with pytest.raises(RetrievalError) as refused:
        await service(configured("org/model-b")).search("meal allowance", collection=COLLECTION)

    assert refused.value.code is ErrorCode.INDEX_DRIFT
    assert refused.value.retryable is False
    assert "fasterrag index reembed" in refused.value.detail
    assert "org/model-a" in refused.value.detail
    assert "org/model-b" in refused.value.detail


async def test_the_refusal_is_a_409_problem() -> None:
    index_with("org/model-a")

    with pytest.raises(RetrievalError) as refused:
        await service(configured("org/model-b")).search("meal allowance", collection=COLLECTION)

    assert refused.value.status == 409


async def test_the_model_the_index_was_built_with_is_served() -> None:
    index_with("org/model-a")

    result = await service(configured("org/model-a")).search("q", collection=COLLECTION)

    assert result.chunks


async def test_the_default_collection_is_checked_too() -> None:
    settings = configured("org/model-b", vector_db={"collection": {"default_name": COLLECTION}})
    index_with("org/model-a")

    with pytest.raises(RetrievalError):
        await service(settings).search("meal allowance")


async def test_a_collection_without_a_lock_is_served() -> None:
    """Nothing to compare against: an index from before lockfiles, or one never written."""
    result = await service(configured("org/model-b")).search("q", collection=COLLECTION)

    assert result.chunks


async def test_warn_mode_serves_the_query_and_says_why_it_should_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    index_with("org/model-a")

    with caplog.at_level(logging.WARNING):
        result = await service(configured("org/model-b", "warn")).search("q", collection=COLLECTION)

    assert result.chunks
    assert any("fasterrag index reembed" in record.getMessage() for record in caplog.records)


async def test_off_does_not_read_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    index_with("org/model-a")
    reads: list[str] = []
    original = LockStore.read

    def counting(store: LockStore, collection: str) -> Any:
        reads.append(collection)
        return original(store, collection)

    monkeypatch.setattr(LockStore, "read", counting)

    result = await service(configured("org/model-b", "off")).search("q", collection=COLLECTION)

    assert result.chunks
    assert reads == []


async def test_a_disabled_lockfile_is_not_consulted() -> None:
    """With ``index.lockfile`` off nothing keeps the lock current, so a stale one proves nothing."""
    index_with("org/model-a")
    settings = configured("org/model-b", index={"lockfile": False})

    result = await service(settings).search("q", collection=COLLECTION)

    assert result.chunks


async def test_a_verified_collection_is_not_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service is built per request; the lock is read once per process and model."""
    index_with("org/model-a")
    reads: list[str] = []
    original = LockStore.read

    def counting(store: LockStore, collection: str) -> Any:
        reads.append(collection)
        return original(store, collection)

    monkeypatch.setattr(LockStore, "read", counting)
    settings = configured("org/model-a")

    for _ in range(3):
        await service(settings).search("q", collection=COLLECTION)

    assert reads == [COLLECTION]


async def test_a_refusal_clears_once_the_index_is_rebuilt() -> None:
    """A refusal is not remembered: after a re-embed writes a matching lock, queries pass."""
    index_with("org/model-a")
    settings = configured("org/model-b")
    with pytest.raises(RetrievalError):
        await service(settings).search("q", collection=COLLECTION)

    index_with("org/model-b")
    result = await service(settings).search("q", collection=COLLECTION)

    assert result.chunks


async def test_the_check_is_callable_on_its_own() -> None:
    index_with("org/model-a")
    settings = configured("org/model-b")

    with pytest.raises(RetrievalError):
        await verify_query_lock(settings, TieringRouter(ConfiguredEmbedder(settings)), COLLECTION)

    lockfile.reset_verified_locks()
