from fasterrag.core.context import AssembledContext, Citation, Span, assemble_context
from fasterrag.core.generation import (
    P1_SYSTEM_PROMPT,
    P1_TEMPLATE_VERSION,
    build_context_block,
    build_prompt,
    citation_marker,
    resolve_citations,
)
from fasterrag.core.retrieval.models import ScoredChunk


def chunk(chunk_id: str, text: str, **payload: object) -> ScoredChunk:
    return ScoredChunk(chunk_id=chunk_id, text=text, payload=payload)


def context_of(*chunks: ScoredChunk) -> tuple[AssembledContext, list[str]]:
    assembled = assemble_context(chunks, budget_tokens=10_000)
    return assembled, [c.text for c in chunks]


def test_the_template_is_versioned_so_a_change_is_visible() -> None:
    assert P1_TEMPLATE_VERSION


def test_the_system_prompt_forbids_outside_knowledge_and_guessing() -> None:
    assert "strictly from the provided context" in P1_SYSTEM_PROMPT
    assert "Do not use outside knowledge" in P1_SYSTEM_PROMPT
    assert "fabricated one is not" in P1_SYSTEM_PROMPT


def test_each_chunk_is_marked_and_attributed() -> None:
    assembled, texts = context_of(
        chunk("c_9f2", "Either party may terminate.", source_uri="vendor.pdf", page=12)
    )

    block = build_context_block(assembled, texts)

    assert "[^c_9f2] (source: vendor.pdf, page: 12)" in block
    assert "Either party may terminate." in block


def test_a_chunk_without_provenance_still_gets_a_marker() -> None:
    assembled, texts = context_of(chunk("c_a", "body"))

    assert build_context_block(assembled, texts).startswith("[^c_a]\nbody")


def test_the_question_goes_last_so_everything_above_it_can_be_cached() -> None:
    assembled, texts = context_of(chunk("c_a", "body"))

    prompt = build_prompt("what is the notice period?", assembled, texts)

    assert prompt.index("<context>") < prompt.index("Question:")
    assert prompt.rstrip().endswith("Question: what is the notice period?")


def test_every_packed_chunk_appears_in_the_prompt() -> None:
    assembled, texts = context_of(chunk("c_a", "first passage"), chunk("c_b", "second passage"))

    prompt = build_prompt("q", assembled, texts)

    assert "[^c_a]" in prompt
    assert "[^c_b]" in prompt
    assert "first passage" in prompt
    assert "second passage" in prompt


def test_the_marker_format_matches_what_the_resolver_reads() -> None:
    supplied = [Citation(chunk_id="c_9f2")]

    assert resolve_citations(f"Answer {citation_marker('c_9f2')}.", supplied) == supplied


def test_only_cited_chunks_are_returned() -> None:
    supplied = [Citation(chunk_id="c_a"), Citation(chunk_id="c_b")]

    resolved = resolve_citations("The answer is here [^c_b].", supplied)

    assert [citation.chunk_id for citation in resolved] == ["c_b"]


def test_citations_come_back_in_order_of_first_appearance() -> None:
    supplied = [Citation(chunk_id="c_a"), Citation(chunk_id="c_b")]

    resolved = resolve_citations("First [^c_b] then [^c_a].", supplied)

    assert [citation.chunk_id for citation in resolved] == ["c_b", "c_a"]


def test_a_chunk_cited_twice_appears_once() -> None:
    supplied = [Citation(chunk_id="c_a")]

    resolved = resolve_citations("One [^c_a] and again [^c_a].", supplied)

    assert len(resolved) == 1


def test_an_invented_marker_is_dropped() -> None:
    supplied = [Citation(chunk_id="c_real")]

    resolved = resolve_citations("Real [^c_real] and invented [^c_hallucinated].", supplied)

    assert [citation.chunk_id for citation in resolved] == ["c_real"]


def test_an_answer_citing_only_invented_chunks_returns_nothing() -> None:
    supplied = [Citation(chunk_id="c_real")]

    assert resolve_citations("Entirely invented [^c_nope].", supplied) == []


def test_an_answer_with_no_markers_returns_no_citations() -> None:
    supplied = [Citation(chunk_id="c_a")]

    assert resolve_citations("An uncited answer.", supplied) == []


def test_resolution_preserves_the_full_citation_including_its_span() -> None:
    supplied = [
        Citation(
            chunk_id="c_a",
            source="vendor.pdf",
            page=12,
            span=Span(start=10, end=40),
            score=0.91,
        )
    ]

    resolved = resolve_citations("Answer [^c_a].", supplied)

    assert resolved[0].source == "vendor.pdf"
    assert resolved[0].span == Span(start=10, end=40)
    assert resolved[0].page == 12


def test_nothing_supplied_resolves_to_nothing() -> None:
    assert resolve_citations("Answer [^c_a].", []) == []


def test_markers_inside_ordinary_prose_do_not_confuse_the_resolver() -> None:
    supplied = [Citation(chunk_id="c_a")]

    resolved = resolve_citations(
        "See section [1] and footnote [^c_a] but not [^ ] or [c_a].", supplied
    )

    assert [citation.chunk_id for citation in resolved] == ["c_a"]
