from typing import Any

import pytest

from fasterrag.core.chunking.models import EstimatingTokenCounter
from fasterrag.core.context import Span, assemble_context
from fasterrag.core.retrieval.models import ScoredChunk

COUNTER = EstimatingTokenCounter()


def chunk(chunk_id: str, text: str, rank: int = 1, **payload: Any) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        payload=payload,
        rrf_score=1.0 / rank,
        final_rank=rank,
    )


def words(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_chunks_are_packed_in_the_order_given() -> None:
    result = assemble_context(
        [chunk("c_a", "first passage", 1), chunk("c_b", "second passage", 2)],
        budget_tokens=1000,
    )

    assert result.text == "first passage\n\nsecond passage"
    assert [citation.chunk_id for citation in result.citations] == ["c_a", "c_b"]
    assert result.used == 2


def test_the_budget_stops_packing_and_the_overflow_is_counted() -> None:
    big = words(100)
    result = assemble_context(
        [chunk("c_a", big, 1), chunk("c_b", words(100, "other"), 2)],
        budget_tokens=COUNTER.count(big),
        counter=COUNTER,
    )

    assert result.used == 1
    assert result.dropped_budget == 1
    assert result.truncated is True


def test_the_least_relevant_material_is_what_gets_dropped() -> None:
    result = assemble_context(
        [
            chunk("c_best", "the termination clause", 1),
            chunk("c_worst", words(200, "filler"), 2),
        ],
        budget_tokens=10,
        counter=COUNTER,
    )

    assert [citation.chunk_id for citation in result.citations] == ["c_best"]


def test_a_context_within_budget_is_not_truncated() -> None:
    result = assemble_context([chunk("c_a", "short")], budget_tokens=1000)

    assert result.truncated is False
    assert result.dropped_budget == 0


def test_identical_chunks_are_deduplicated() -> None:
    result = assemble_context(
        [chunk("c_a", "the same text", 1), chunk("c_b", "the same text", 2)],
        budget_tokens=1000,
    )

    assert result.used == 1
    assert result.dropped_duplicate == 1


def test_near_duplicates_from_chunk_overlap_are_dropped() -> None:
    shared = words(40)
    result = assemble_context(
        [chunk("c_a", shared, 1), chunk("c_b", f"{shared} word40", 2)],
        budget_tokens=1000,
    )

    assert result.used == 1
    assert result.dropped_duplicate == 1


def test_genuinely_different_chunks_both_survive() -> None:
    result = assemble_context(
        [
            chunk("c_a", "termination requires thirty days written notice", 1),
            chunk("c_b", "invoices are payable within forty five days", 2),
        ],
        budget_tokens=1000,
    )

    assert result.used == 2


def test_deduplication_can_be_disabled() -> None:
    shared = words(40)
    result = assemble_context(
        [chunk("c_a", shared, 1), chunk("c_b", f"{shared} extra", 2)],
        budget_tokens=1000,
        similarity_threshold=1.0,
    )

    assert result.used == 2


def test_a_citation_carries_source_page_and_span() -> None:
    result = assemble_context(
        [
            chunk(
                "c_a",
                "body",
                source_uri="contracts/vendor.pdf",
                page=12,
                span={"start": 100, "end": 220},
            )
        ],
        budget_tokens=1000,
    )
    citation = result.citations[0]

    assert citation.source == "contracts/vendor.pdf"
    assert citation.page == 12
    assert citation.span == Span(start=100, end=220)


def test_a_citation_survives_a_chunk_with_no_structure() -> None:
    result = assemble_context([chunk("c_a", "body")], budget_tokens=1000)
    citation = result.citations[0]

    assert citation.chunk_id == "c_a"
    assert citation.source is None
    assert citation.page is None
    assert citation.span is None


def test_a_citation_serializes_without_unknown_fields() -> None:
    payload = assemble_context([chunk("c_a", "body")], budget_tokens=1000).citations[0].as_dict()

    assert payload == {"chunk_id": "c_a", "score": 1.0}


def test_the_rerank_score_is_preferred_when_present() -> None:
    reranked = ScoredChunk(chunk_id="c_a", text="body", rrf_score=0.01, rerank_score=8.5)

    citation = assemble_context([reranked], budget_tokens=1000).citations[0]

    assert citation.score == pytest.approx(8.5)


def test_a_malformed_span_is_ignored_rather_than_crashing() -> None:
    result = assemble_context([chunk("c_a", "body", span={"start": "oops"})], budget_tokens=1000)

    assert result.citations[0].span is None


def test_blank_chunks_are_skipped() -> None:
    result = assemble_context(
        [chunk("c_a", "   ", 1), chunk("c_b", "real content", 2)], budget_tokens=1000
    )

    assert result.used == 1
    assert result.citations[0].chunk_id == "c_b"


def test_no_chunks_assemble_to_an_empty_context() -> None:
    result = assemble_context([], budget_tokens=1000)

    assert result.empty is True
    assert result.text == ""
    assert result.tokens == 0


def test_a_zero_budget_admits_nothing() -> None:
    result = assemble_context([chunk("c_a", "anything")], budget_tokens=0)

    assert result.empty is True
    assert result.dropped_budget == 1


def test_the_reported_token_count_matches_what_was_packed() -> None:
    first, second = "the first passage here", "an entirely separate passage"
    result = assemble_context(
        [chunk("c_a", first, 1), chunk("c_b", second, 2)],
        budget_tokens=1000,
        counter=COUNTER,
    )

    assert result.tokens == COUNTER.count(first) + COUNTER.count(second)


def test_every_packed_chunk_has_exactly_one_citation() -> None:
    result = assemble_context(
        [chunk(f"c_{index}", f"passage number {index}", index + 1) for index in range(5)],
        budget_tokens=1000,
    )

    assert result.used == len(result.citations)
    assert len({citation.chunk_id for citation in result.citations}) == 5
