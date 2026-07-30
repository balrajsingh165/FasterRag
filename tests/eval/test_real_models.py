"""Verification against real downloaded models.

Every other test in the suite substitutes a fake for the embedder and the reranker, which
proves the plumbing but never the integration. These run the actual models, so the things a
fake cannot check are checked here: that the configured model loads at all, that it reports
the dimensions the collection will be built with, and that reranking genuinely reorders by
meaning.

Marked ``eval`` because they download and load model weights. Run them with
``pytest -m eval`` after ``pip install "fasterrag[rerank]"``; they skip cleanly without it.
"""

import warnings
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk

pytestmark = pytest.mark.eval

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMENSIONS = 384
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def require_models() -> None:
    """Skip unless the model stack is installed."""
    pytest.importorskip(
        "sentence_transformers",
        reason='install with: pip install "fasterrag[rerank]"',
    )


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "embeddings": {"provider": "huggingface", "model": EMBED_MODEL},
        "retrieval": {"reranker_model": RERANK_MODEL},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


async def test_the_default_embedding_model_loads_and_reports_its_dimensions() -> None:
    require_models()
    from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder

    embedder = HuggingFaceEmbedder(settings())
    try:
        result = await embedder.embed_documents(["Either party may terminate the agreement."])

        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == EMBED_DIMENSIONS
        assert embedder.dimensions == EMBED_DIMENSIONS
        assert result.model == EMBED_MODEL
    finally:
        await embedder.close()


async def test_the_dimension_accessor_works_without_a_deprecation_warning() -> None:
    require_models()
    from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder

    embedder = HuggingFaceEmbedder(settings())
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            await embedder.embed_documents(["text"])

        renames = [
            warning
            for warning in recorded
            if "get_sentence_embedding_dimension" in str(warning.message)
        ]
        assert renames == []
        assert embedder.dimensions == EMBED_DIMENSIONS
    finally:
        await embedder.close()


async def test_a_real_embedder_places_related_text_closer_than_unrelated() -> None:
    require_models()
    from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder

    embedder = HuggingFaceEmbedder(settings())
    try:
        result = await embedder.embed_documents(
            [
                "Either party may terminate this agreement with thirty days notice.",
                "How do I end the contract?",
                "The office is on the third floor.",
            ]
        )
    finally:
        await embedder.close()

    contract, question, unrelated = result.vectors

    def similarity(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    assert similarity(question, contract) > similarity(question, unrelated)


async def test_the_real_reranker_promotes_the_answer_despite_no_shared_words() -> None:
    require_models()
    from fasterrag.core.rerank import CrossEncoderReranker

    candidates = [
        ScoredChunk(
            chunk_id="c_payment", text="Invoices are payable within 45 days.", final_rank=1
        ),
        ScoredChunk(
            chunk_id="c_termination",
            text="Either party may terminate this agreement with thirty days written notice.",
            final_rank=2,
        ),
        ScoredChunk(chunk_id="c_noise", text="The office is on the third floor.", final_rank=3),
    ]

    reordered = await CrossEncoderReranker(settings()).rerank(
        "how do I end the contract?", candidates
    )

    assert reordered[0].chunk_id == "c_termination"
    assert reordered[0].final_rank == 1
    assert reordered[0].rerank_score is not None
    assert reordered[-1].chunk_id == "c_noise"


async def test_the_reranker_scores_every_candidate() -> None:
    require_models()
    from fasterrag.core.rerank import CrossEncoderReranker

    candidates = [
        ScoredChunk(chunk_id=f"c_{index}", text=f"passage number {index}", final_rank=index + 1)
        for index in range(4)
    ]

    reordered = await CrossEncoderReranker(settings()).rerank("a query", candidates)

    assert len(reordered) == 4
    assert all(chunk.rerank_score is not None for chunk in reordered)
    assert [chunk.final_rank for chunk in reordered] == [1, 2, 3, 4]
