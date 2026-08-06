"""Context assembly invariants, generated rather than hand-picked.

Assembly decides what evidence reaches the model, so a failure here is an answer grounded in
something other than what the retriever chose. The properties worth pinning are the ones a
reader would assume without checking: the budget binds, the citations describe exactly what
was packed, order survives, and everything handed over is accounted for.

That last one was false — a chunk with no text was skipped without being counted, so the
totals did not reconcile and a retrieved chunk could vanish with nothing reporting it.
"""

from hypothesis import given
from hypothesis import strategies as st

from fasterrag.core.chunking.models import EstimatingTokenCounter
from fasterrag.core.context import assemble_context
from fasterrag.core.retrieval.models import ScoredChunk

COUNTER = EstimatingTokenCounter()

# Includes blank and whitespace-only text on purpose: those are the chunks the accounting
# used to lose, and a strategy that generated only real text could not have found it.
TEXTS = st.lists(
    st.one_of(
        st.text(min_size=0, max_size=80),
        st.just(""),
        st.just("   "),
    ),
    min_size=0,
    max_size=12,
)

BUDGET = st.integers(min_value=1, max_value=200)


def chunks(texts: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk_id=f"c{index}", text=text, final_rank=index)
        for index, text in enumerate(texts)
    ]


@given(texts=TEXTS, budget=BUDGET)
def test_every_chunk_is_accounted_for(texts: list[str], budget: int) -> None:
    """Used, duplicated, over budget, or blank — a retrieved chunk cannot simply vanish."""
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)

    accounted = (
        result.used + result.dropped_duplicate + result.dropped_budget + result.dropped_empty
    )

    assert accounted == len(texts)


@given(texts=TEXTS, budget=BUDGET)
def test_the_budget_binds(texts: list[str], budget: int) -> None:
    """Overrunning it is a prompt the model refuses, or silently truncates."""
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)

    assert result.tokens <= budget


@given(texts=TEXTS, budget=BUDGET)
def test_a_citation_per_packed_chunk(texts: list[str], budget: int) -> None:
    """A mismatch means an answer cites evidence that is not in its context, or vice versa."""
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)

    assert len(result.citations) == result.used


@given(texts=TEXTS, budget=BUDGET)
def test_citations_keep_relevance_order(texts: list[str], budget: int) -> None:
    """Chunks arrive best-first; reordering them would misrepresent what ranked where."""
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)
    ranks = [int(citation.chunk_id.removeprefix("c")) for citation in result.citations]

    assert ranks == sorted(ranks)


@given(texts=TEXTS, budget=BUDGET)
def test_nothing_blank_is_ever_packed(texts: list[str], budget: int) -> None:
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)

    assert all(part.strip() for part in result.text.split("\n\n") if part)


@given(texts=TEXTS, budget=BUDGET)
def test_empty_reports_whether_anything_survived(texts: list[str], budget: int) -> None:
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)

    assert result.empty is (result.used == 0)


@given(texts=TEXTS, budget=BUDGET)
def test_truncated_means_the_budget_ended_packing(texts: list[str], budget: int) -> None:
    """`truncated` must mean the budget bound, not that anything at all was dropped."""
    result = assemble_context(chunks(texts), budget_tokens=budget, counter=COUNTER)

    assert result.truncated is (result.dropped_budget > 0)


@given(texts=st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=6), budget=BUDGET)
def test_identical_chunks_are_packed_once(texts: list[str], budget: int) -> None:
    """A duplicate spends budget twice on one piece of evidence."""
    doubled = [text for text in texts for _ in range(2)]
    result = assemble_context(chunks(doubled), budget_tokens=budget, counter=COUNTER)

    packed = [part for part in result.text.split("\n\n") if part.strip()]

    assert len(packed) == len(set(packed))


@given(texts=TEXTS)
def test_a_budget_of_one_still_packs_something_or_nothing_cleanly(texts: list[str]) -> None:
    """The per-chunk floor must not let a single chunk exceed the whole budget."""
    result = assemble_context(chunks(texts), budget_tokens=1, counter=COUNTER)

    assert result.tokens <= 1
    assert result.used <= 1
