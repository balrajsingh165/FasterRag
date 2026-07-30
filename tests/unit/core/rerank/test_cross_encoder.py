from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.rerank import cross_encoder
from fasterrag.core.rerank.cross_encoder import CrossEncoderReranker
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ConfigError, ErrorCode, RetrievalError


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the chunk."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.raises: Exception | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.raises is not None:
            raise self.raises
        self.calls.append(list(pairs))
        return [
            float(sum(word in chunk.lower() for word in query.lower().split()))
            for query, chunk in pairs
        ]


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> FakeCrossEncoder:
    fake = FakeCrossEncoder()
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", lambda name: fake)
    return fake


def chunk(chunk_id: str, text: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk_id=chunk_id, text=text, rrf_score=1.0 / rank, final_rank=rank)


def settings(**overrides: Any) -> Settings:
    return Settings.model_validate(overrides)


def test_construction_loads_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", lambda name: loaded.append(name))

    CrossEncoderReranker(settings())

    assert loaded == []


async def test_the_model_is_loaded_once_and_reused(model: FakeCrossEncoder) -> None:
    reranker = CrossEncoderReranker(settings())
    candidates = [chunk("c_a", "notice period", 1)]

    await reranker.rerank("notice", candidates)
    await reranker.rerank("notice", candidates)

    assert reranker._model is model
    assert len(model.calls) == 2


async def test_a_better_match_is_promoted_above_a_worse_one(model: FakeCrossEncoder) -> None:
    candidates = [
        chunk("c_weak", "unrelated content about invoices", 1),
        chunk("c_strong", "the termination notice period is thirty days", 2),
    ]

    results = await CrossEncoderReranker(settings()).rerank("termination notice period", candidates)

    assert [result.chunk_id for result in results] == ["c_strong", "c_weak"]


async def test_every_result_carries_its_rerank_score_and_new_rank(
    model: FakeCrossEncoder,
) -> None:
    candidates = [chunk("c_a", "notice", 1), chunk("c_b", "termination notice", 2)]

    results = await CrossEncoderReranker(settings()).rerank("termination notice", candidates)

    assert all(result.rerank_score is not None for result in results)
    assert [result.final_rank for result in results] == [1, 2]


async def test_the_earlier_fusion_evidence_is_preserved(model: FakeCrossEncoder) -> None:
    candidates = [chunk("c_a", "notice", 1)]

    results = await CrossEncoderReranker(settings()).rerank("notice", candidates)

    assert results[0].rrf_score == pytest.approx(1.0)


async def test_the_query_is_paired_with_every_chunk(model: FakeCrossEncoder) -> None:
    candidates = [chunk("c_a", "one", 1), chunk("c_b", "two", 2)]

    await CrossEncoderReranker(settings()).rerank("a query", candidates)

    assert model.calls[0] == [("a query", "one"), ("a query", "two")]


async def test_an_empty_shortlist_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(name: str) -> object:
        raise AssertionError("an empty shortlist must not load a model")

    monkeypatch.setattr(cross_encoder, "load_cross_encoder", refuse)

    assert await CrossEncoderReranker(settings()).rerank("q", []) == []


async def test_a_scoring_failure_is_a_typed_rerank_error(model: FakeCrossEncoder) -> None:
    model.raises = RuntimeError("out of memory")

    with pytest.raises(RetrievalError, match="failed to score") as caught:
        await CrossEncoderReranker(settings()).rerank("q", [chunk("c_a", "text", 1)])

    assert caught.value.code is ErrorCode.RERANK_FAILED
    assert caught.value.retryable is False


def test_the_configured_model_is_used() -> None:
    configured = settings(retrieval={"reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})

    assert CrossEncoderReranker(configured).model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_a_missing_extra_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises((ConfigError, AttributeError, ImportError, TypeError)):
        cross_encoder.load_cross_encoder("any-model")
