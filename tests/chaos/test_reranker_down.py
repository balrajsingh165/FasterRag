"""The reranker-down chaos scenario FMEA row 18 names (TASK-0289).

Row 18 cited "Chaos: reranker-down rung" and no such scenario existed. The rung it names is
real and the unit suite covers it — ``tests/unit/services/test_querying.py`` drives a
``ScriptedReranker`` that raises on demand — but that double raises *the exception the
service's author expected*, at the service's own Protocol boundary. The whole translation
layer between a model that will not load and that exception is below the double, and it is
the part that actually fails in production: a cross-encoder load traverses
``huggingface_hub``, ``safetensors``, ``transformers`` and ``torch``, each raising its own
types, and anything leaking untyped out of
:meth:`fasterrag.core.rerank.CrossEncoderReranker._loaded` walks straight past
:meth:`RetrievalService._rerank`, whose ``except`` clause only catches ``FasterRagError``.
An untyped leak there does not degrade the query — it fails it, which is the exact outcome
D4's ladder exists to prevent.

So the fault here is injected below the double, in two ways. The first case drives the
**real** ``CrossEncoderReranker`` against the **real** ``sentence_transformers`` with a model
name the hub library rejects, so the exception is one the library genuinely raises rather
than one a test author chose. That name is malformed as a repository id, which the library
refuses locally before any request is made: the case needs no network, no cached weights and
no GPU, and it is skipped when the optional ``rerank`` extra is absent. The remaining cases
patch ``load_cross_encoder`` — the one seam between this package and the model stack — with
the *shape* a real failure arrives in (an out-of-memory ``RuntimeError``, a model whose
``predict`` dies mid-query), because exhausting real VRAM in a test is not reproducible and
simulating the symptom is legitimate where simulating the handling would not be.

What is asserted is the built behaviour. Two consequences of it are pinned here as limits
rather than as promises, so that changing either is a visible diff:

* a model is loaded lazily on first use, never at startup, so the failure always surfaces on
  a query and never during boot;
* ``ConfigError`` — the error raised when the ``rerank`` extra is not installed at all — is a
  ``FasterRagError``, so the ladder absorbs it and every query degrades silently instead of
  the process refusing to serve a misconfiguration. ``CrossEncoderReranker`` re-raises that
  one deliberately to keep it unwrapped; the service catches it anyway.

No timing figure appears below and nothing here is a measurement.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Final

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.rerank import CrossEncoderReranker
from fasterrag.errors import ConfigError, ErrorCode, FasterRagError, RetrievalError
from fasterrag.services.querying import FULL_MODE, HYBRID_ONLY_MODE, RetrievalService

pytestmark = pytest.mark.chaos

COLLECTION: Final = "docs"
VECTOR: Final[list[float]] = [0.1, 0.2, 0.3]
LOADER: Final = "fasterrag.core.rerank.cross_encoder.load_cross_encoder"

UNLOADABLE_MODEL: Final = "a model id with spaces"
"""A name ``huggingface_hub`` refuses as a repository id without making a request.

The refusal is local — the library validates the id before it resolves anything — so the
real-library case stays hermetic, needs no cached weights, and fails the same way on a
machine with no network at all.
"""

OOM: Final = "CUDA out of memory. Tried to allocate 2.20 GiB"
"""The message torch raises when a cross-encoder will not fit; the type is ``RuntimeError``."""


class StubRouter:
    """Supplies a fixed query vector so no embedding model is loaded."""

    def __init__(self) -> None:
        """Expose itself as its own default adapter, as the real router does."""
        self.default = self

    async def embed_query(self, text: str) -> Sequence[float]:
        """Return the fixed vector."""
        return VECTOR


class ScriptedAdapter(VectorDBAdapter):
    """Returns a fixed dense shortlist so the only failure domain is the reranker."""

    def __init__(self, settings: Settings) -> None:
        """Start with the shortlist every case in this module retrieves."""
        super().__init__(settings)
        self.dense: list[ScoredPoint] = [
            point("c_a", 0.90),
            point("c_b", 0.80),
            point("c_c", 0.70),
        ]

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Unused; nothing here writes."""

    async def list_collections(self) -> list[CollectionInfo]:
        """Unused."""
        return []

    async def drop_collection(self, name: str) -> bool:
        """Unused."""
        return False

    async def snapshot(self, collection: str) -> str:
        """Unused."""
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        """Unused."""
        return []

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Unused."""
        return False

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Unused."""

    async def set_alias(self, alias: str, collection: str) -> None:
        """Unused."""

    async def alias_target(self, alias: str) -> str | None:
        """Unused."""
        return None

    async def delete_alias(self, alias: str) -> bool:
        """Unused."""
        return False

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Unused."""
        return UpsertResult(upserted=0)

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield nothing; this double holds no scannable state."""
        empty: list[Point] = []
        for item in empty:
            yield item

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Answer the dense leg and return nothing for the keyword leg."""
        return [] if query.sparse is not None else list(self.dense)

    async def update(self, updates: list[PointUpdate]) -> None:
        """Unused."""

    async def delete(self, selector: PointSelector) -> None:
        """Unused."""

    async def health(self) -> HealthStatus:
        """The backend is healthy throughout; only the reranker fails."""
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        """Nothing to release."""


class Loader:
    """A patched ``load_cross_encoder`` that records every call it is given."""

    def __init__(self, *, raises: Exception | None = None, model: Any | None = None) -> None:
        """Fail with ``raises`` if given one, otherwise hand back ``model``."""
        self.raises = raises
        self.model = model
        self.calls: list[str] = []

    def __call__(self, model: str) -> Any:
        """Record the request and either fail it or serve the scripted model."""
        self.calls.append(model)
        if self.raises is not None:
            raise self.raises
        return self.model


class ScoringModel:
    """A loaded cross-encoder that scores, or dies while scoring."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        """Score in reverse shortlist order unless told to fail."""
        self.raises = raises
        self.calls = 0

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return a descending score per pair, reversing the fused order."""
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return [float(index) for index in range(1, len(pairs) + 1)]


def point(point_id: str, score: float) -> ScoredPoint:
    """Return one dense hit carrying the payload the retrieval layer expects."""
    return ScoredPoint(
        point_id=point_id,
        score=score,
        payload={
            "text": f"body of {point_id}",
            "document_id": "d_1",
            "source_uri": "handbook.md",
            "point_id": point_id,
        },
    )


def chaos_settings(model: str = UNLOADABLE_MODEL) -> Settings:
    """Return settings with reranking on and a shortlist wider than ``top_k``."""
    return Settings.model_validate(
        {
            "vector_db": {"provider": "qdrant", "mode": "external", "api_key_env": None},
            "embeddings": {"provider": "huggingface"},
            "llm": {"provider": "ollama", "api_key_env": None},
            "retrieval": {
                "rerank": True,
                "top_k": 3,
                "rerank_top_n": 10,
                "reranker_model": model,
            },
        }
    )


def service_for(settings: Settings) -> RetrievalService:
    """Wire the real retrieval service to the real reranker over the scripted shortlist."""
    return RetrievalService(
        settings,
        ScriptedAdapter(settings),
        StubRouter(),  # type: ignore[arg-type]
        reranker=CrossEncoderReranker(settings),
    )


def patch_loader(monkeypatch: pytest.MonkeyPatch, loader: Loader) -> Loader:
    """Replace the one seam between this package and the model stack."""
    monkeypatch.setattr(LOADER, loader)
    return loader


def test_a_model_that_really_will_not_load_is_classified_rerank_failed() -> None:
    """The real library's own exception, not a double's, becomes a typed RERANK_FAILED."""
    pytest.importorskip(
        "sentence_transformers", reason="the rerank extra is not installed in this environment"
    )
    reranker = CrossEncoderReranker(chaos_settings())

    with pytest.raises(RetrievalError) as failure:
        reranker.score("what is the notice period?", [])

    assert failure.value.code is ErrorCode.RERANK_FAILED
    assert failure.value.retryable is False
    assert UNLOADABLE_MODEL in str(failure.value)
    assert failure.value.__cause__ is not None
    assert not isinstance(failure.value.__cause__, FasterRagError)


async def test_a_reranker_that_really_will_not_load_degrades_the_query() -> None:
    """Row 18 end to end: the real library fails and the query is still answered."""
    pytest.importorskip(
        "sentence_transformers", reason="the rerank extra is not installed in this environment"
    )
    service = service_for(chaos_settings())

    result = await service.search("what is the notice period?")

    assert result.mode == HYBRID_ONLY_MODE
    assert result.degraded is True
    assert [chunk.chunk_id for chunk in result.chunks] == ["c_a", "c_b", "c_c"]
    assert all(chunk.rerank_score is None for chunk in result.chunks)


async def test_a_model_that_will_not_fit_in_memory_degrades_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OOM half of row 18, which cannot be produced honestly on arbitrary hardware."""
    loader = patch_loader(monkeypatch, Loader(raises=RuntimeError(OOM)))
    service = service_for(chaos_settings())

    result = await service.search("what is the notice period?")

    assert loader.calls == [UNLOADABLE_MODEL]
    assert result.mode == HYBRID_ONLY_MODE
    assert result.degraded is True
    assert [chunk.chunk_id for chunk in result.chunks] == ["c_a", "c_b", "c_c"]


async def test_a_reranker_that_dies_mid_query_degrades_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other seam: the model loaded, and scoring is what blew up."""
    model = ScoringModel(raises=RuntimeError(OOM))
    loader = patch_loader(monkeypatch, Loader(model=model))
    service = service_for(chaos_settings())

    result = await service.search("what is the notice period?")

    assert loader.calls == [UNLOADABLE_MODEL]
    assert model.calls == 1
    assert result.mode == HYBRID_ONLY_MODE
    assert [chunk.chunk_id for chunk in result.chunks] == ["c_a", "c_b", "c_c"]


async def test_the_degraded_rung_holds_for_every_query_not_just_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed load is retried rather than cached, so the rung cannot silently become fatal."""
    loader = patch_loader(monkeypatch, Loader(raises=RuntimeError(OOM)))
    service = service_for(chaos_settings())

    first = await service.search("what is the notice period?")
    second = await service.search("what is the notice period?")

    assert len(loader.calls) == 2
    assert first.mode == HYBRID_ONLY_MODE
    assert second.mode == HYBRID_ONLY_MODE


async def test_a_working_reranker_reports_full_mode_over_the_same_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control. Without it every assertion above would pass on a service that never ranks."""
    model = ScoringModel()
    patch_loader(monkeypatch, Loader(model=model))
    service = service_for(chaos_settings())

    result = await service.search("what is the notice period?")

    assert model.calls == 1
    assert result.mode == FULL_MODE
    assert result.degraded is False
    assert [chunk.chunk_id for chunk in result.chunks] == ["c_c", "c_b", "c_a"]


async def test_a_missing_rerank_extra_degrades_too_which_is_a_limit_not_a_guarantee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned as built. ``ConfigError`` is a ``FasterRagError``, so the ladder swallows it.

    ``CrossEncoderReranker`` re-raises it unwrapped on purpose, which reads as fail-fast on a
    misconfiguration; the service's ``except FasterRagError`` absorbs it anyway and serves a
    degraded answer forever. Asserted as observed so that changing it is a visible diff.
    """
    patch_loader(monkeypatch, Loader(raises=ConfigError("install fasterrag[rerank]")))
    service = service_for(chaos_settings())

    result = await service.search("what is the notice period?")

    assert result.mode == HYBRID_ONLY_MODE
    assert result.degraded is True


def test_the_model_is_not_loaded_until_a_query_needs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row 18's detection column said "at startup or first query"; only the second happens."""
    loader = patch_loader(monkeypatch, Loader(raises=RuntimeError(OOM)))

    service_for(chaos_settings())

    assert loader.calls == []
