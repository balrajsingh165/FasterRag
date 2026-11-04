"""``ScoredChunk``: the per-leg evidence that makes a ranking explainable.

Covered here rather than only through ``QueryService`` because the defaults and the
agreement flag are read by the reranker, the context builder and ``--show-chunks``, while
the mutation run of ``core/retrieval`` (TASK-0300) runs only this package's own tests — and
found nothing in it defending any of them.
"""

from dataclasses import FrozenInstanceError

import pytest

from fasterrag.core.retrieval.models import ScoredChunk


def test_an_unfused_chunk_scores_zero_and_ranks_nowhere() -> None:
    """A non-zero default would put a chunk nothing retrieved ahead of ones that were."""
    chunk = ScoredChunk(chunk_id="c_a", text="body")

    assert chunk.rrf_score == 0.0
    assert chunk.final_rank == 0
    assert chunk.rerank_score is None
    assert chunk.dense_rank is None
    assert chunk.bm25_rank is None


def test_agreement_means_both_legs_and_never_merely_one() -> None:
    """``found_by_both_legs`` is what the trace and the dashboard call an agreement."""
    both = ScoredChunk(chunk_id="c", text="t", dense_rank=1, bm25_rank=4)
    dense_only = ScoredChunk(chunk_id="c", text="t", dense_rank=1)
    sparse_only = ScoredChunk(chunk_id="c", text="t", bm25_rank=4)
    neither = ScoredChunk(chunk_id="c", text="t")

    assert both.found_by_both_legs is True
    assert dense_only.found_by_both_legs is False
    assert sparse_only.found_by_both_legs is False
    assert neither.found_by_both_legs is False


def test_the_citation_fields_are_read_from_the_payload() -> None:
    chunk = ScoredChunk(
        chunk_id="c", text="t", payload={"document_id": "d_1", "source_uri": "contract.pdf"}
    )

    assert chunk.document_id == "d_1"
    assert chunk.source == "contract.pdf"


def test_a_payload_field_of_the_wrong_type_reads_as_absent() -> None:
    """A citation cannot be built from a non-string id, so the property must not offer one."""
    chunk = ScoredChunk(chunk_id="c", text="t", payload={"document_id": 7, "source_uri": []})

    assert chunk.document_id is None
    assert chunk.source is None


def test_a_chunk_cannot_be_rewritten_once_retrieval_has_scored_it() -> None:
    """Reranking and context assembly build new chunks; editing one in place would hide why."""
    chunk = ScoredChunk(chunk_id="c_a", text="body", rrf_score=0.5)

    with pytest.raises(FrozenInstanceError):
        # CRITICAL: mypy is right that this assignment is illegal, and that is the assertion.
        # Every stage after fusion builds a new chunk rather than editing this one, so the
        # per-leg evidence it carries stays the evidence retrieval actually produced. This
        # proves the dataclass refuses at runtime, not only at type-check time, so the
        # ignore cannot be dropped.
        chunk.rrf_score = 9.0  # type: ignore[misc]

    assert chunk.rrf_score == 0.5
