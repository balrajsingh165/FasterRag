"""Where a chunk boundary lands, and what its overlap carries.

``test_invariants.py`` proves a chunking run produced something *legal*: a gapless tiling,
offsets in bounds, nothing past the budget. None of that pins the boundary rules
themselves, and the first mutation run of ``core/chunking`` (TASK-0300, 2026-11-04) showed
what that costs — the overlap could be removed outright, the packing anchor moved, the
estimating counter made to return zero for every string, and the page lookup disabled,
with all 122 existing tests still green.

These assert positions: which offset a cut lands on, and which characters the next chunk
reaches back for. Boundaries are not a local concern — ``core/identity`` derives document
and chunk ids from them, so a boundary that moves silently re-ids a whole corpus.
"""

from collections.abc import Sequence
from itertools import pairwise

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.chunking import (
    Chunker,
    FixedChunker,
    LateChunker,
    LayoutChunker,
    RecursiveChunker,
    SemanticChunker,
    create_chunker,
)
from fasterrag.core.chunking.models import (
    EstimatingTokenCounter,
    TextChunk,
    assemble,
    hard_split,
    within_budget,
)
from fasterrag.core.chunking.recursive import pack, split_span
from fasterrag.core.chunking.semantic import _distance, _sentences, _threshold
from fasterrag.core.parsing import parse_markdown, parse_plaintext
from fasterrag.core.parsing.models import ParsedDocument
from fasterrag.errors import EmbedError, ErrorCode, IngestionError

WORDS = "alpha beta gamma delta"
DENSE_BODY = "検索拡張生成" * 200
DEFAULT_LIMIT_BODY = " ".join(f"w{index:03d}" for index in range(4000))


class DenseCounter:
    """Every non-space character is one token, as CJK very nearly is."""

    def count(self, text: str) -> int:
        return len(text.strip())

    @property
    def chars_per_token(self) -> int:
        return 4


class RecordingCounter:
    """Counts like the estimate, but records that it was the one asked."""

    def __init__(self) -> None:
        self.counted = False
        self.sized = False

    def count(self, text: str) -> int:
        self.counted = True
        return len(text.strip()) // 4

    @property
    def chars_per_token(self) -> int:
        self.sized = True
        return 4


class ShiftingEmbedder:
    """Reports a topic change exactly once, at the third sentence."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if index < 2 else [0.0, 1.0] for index in range(len(texts))]


def test_the_estimate_counts_characters_and_never_fewer_than_words() -> None:
    """Every chunk size in the system is this number; a zero here sizes nothing at all."""
    counter = EstimatingTokenCounter()

    assert counter.count(WORDS) == 5
    assert counter.count("a b c d e f") == 6
    assert counter.count("   \n ") == 0


def test_a_hard_split_cuts_at_the_last_space_inside_the_window() -> None:
    """Snapping to a word boundary is the rule; the limit only says where to look."""
    assert hard_split(WORDS, 0, 12) == [(0, 11), (11, 22)]


def test_a_trailing_window_that_already_fits_is_left_uncut() -> None:
    """Snapping the tail as well would end most documents on a spurious extra chunk."""
    assert hard_split("alpha beta gamma", 0, 20) == [(0, 16)]


def test_hard_split_offsets_stay_absolute_to_the_span_it_was_given() -> None:
    """A segment that loses its origin moves the text every chunk id is derived from."""
    assert hard_split(WORDS, 7, 12) == [(7, 18), (18, 29)]


def test_the_snap_never_reaches_back_past_half_the_window() -> None:
    """Looking further back lets one early space collapse a full window to five characters."""
    assert hard_split("alph bcdefghijklm nop", 0, 12) == [(0, 12), (12, 21)]


def test_packing_merges_neighbours_while_the_merged_span_still_fits() -> None:
    """Measuring from the wrong anchor disables packing, leaving chunks ragged but legal."""
    assert pack([(0, 10), (10, 20), (20, 30), (30, 40)], 25) == [(0, 20), (20, 40)]


def test_packing_accepts_a_merge_that_lands_exactly_on_the_limit() -> None:
    """Refusing the exact fit wastes part of every chunk for nothing."""
    assert pack([(0, 10), (10, 25)], 25) == [(0, 25)]


def test_a_span_that_already_fits_is_returned_uncut() -> None:
    assert split_span(WORDS, 0, len(WORDS), len(WORDS)) == [(0, len(WORDS))]


def test_the_cut_prefers_the_blank_line_over_the_line_break() -> None:
    """Separator order is the whole strategy: a paragraph break beats a line break."""
    text = "aaa\nbbb\n\nccc\nddd"

    assert split_span(text, 0, len(text), 10) == [(0, 9), (9, 16)]


def test_the_next_separator_is_used_when_the_coarsest_one_is_absent() -> None:
    text = "aaa\nbbb\nccc\nddd"

    assert split_span(text, 0, len(text), 10) == [(0, 4), (4, 8), (8, 12), (12, 15)]


def test_a_piece_that_still_does_not_fit_falls_through_to_the_next_separator() -> None:
    """Recursing on the wrong tail of the separator list drops straight to a hard cut."""
    text = "aaabbb\ncccddd\n\neee"

    assert split_span(text, 0, len(text), 8) == [(0, 7), (7, 14), (14, 15), (15, 18)]


def test_each_chunk_carries_exactly_the_configured_overlap_back() -> None:
    """Overlap is the only thing keeping a sentence alive across a boundary."""
    counter = EstimatingTokenCounter()
    overlap_tokens = 5
    document = parse_plaintext(" ".join(f"word{index:03d}" for index in range(200)).encode())

    chunks = FixedChunker(chunk_size=20, overlap=overlap_tokens, counter=counter).split(document)

    assert len(chunks) > 2
    for earlier, later in pairwise(chunks):
        carried = document.text[later.start : earlier.end]
        assert len(carried) == overlap_tokens * counter.chars_per_token
        assert carried == earlier.text[-len(carried) :]
        assert later.text.startswith(carried)


def test_a_dense_overlap_halves_down_to_the_budget_and_stops_on_it() -> None:
    """One halving too many drops the context the overlap exists to carry."""
    counter = DenseCounter()
    overlap_tokens = 8
    document = parse_plaintext(DENSE_BODY.encode())

    chunks = RecursiveChunker(chunk_size=40, overlap=overlap_tokens, counter=counter).split(
        document
    )

    assert len(chunks) > 2
    for earlier, later in pairwise(chunks):
        assert counter.count(document.text[later.start : earlier.end]) == overlap_tokens


DEFAULTED_CHUNKERS: list[Chunker] = [FixedChunker(), RecursiveChunker(), LateChunker()]


@pytest.mark.parametrize(
    "chunker",
    DEFAULTED_CHUNKERS,
    ids=[chunker.strategy for chunker in DEFAULTED_CHUNKERS],
)
def test_the_shipped_defaults_are_the_documented_window_and_overlap(chunker: Chunker) -> None:
    """A deployment that configures nothing gets 768 tokens with 64 of overlap, or the docs lie."""
    chunks = chunker.split(parse_plaintext(DEFAULT_LIMIT_BODY.encode()))

    assert [(chunk.start, chunk.end) for chunk in chunks[:2]] == [(0, 3070), (2814, 6140)]


def test_a_chunk_cannot_be_mutated_after_it_is_built() -> None:
    """Chunks travel between worker pools; a mutable one is a boundary that can move later."""
    chunk = TextChunk(text="a", start=0, end=1, chunk_index=0, token_count=1, strategy="fixed")

    with pytest.raises(AttributeError):
        chunk.start = 5  # type: ignore[misc]


def test_the_page_and_section_lookups_reach_every_chunk() -> None:
    """Page and section are how a span-level citation resolves back to a place in the source."""
    chunks = assemble(
        WORDS,
        [(0, 11), (11, 22)],
        overlap_chars=0,
        strategy="fixed",
        counter=EstimatingTokenCounter(),
        page_at=lambda offset: 1 + offset // 11,
        section_at=lambda offset: f"s{offset}",
    )

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.section for chunk in chunks] == ["s0", "s11"]


def test_the_breakpoint_threshold_is_the_percentile_of_the_observed_distances() -> None:
    """The percentile is the adaptive rule; a fixed index would ignore the document."""
    distances = [0.1, 0.2, 0.3, 0.4, 0.9]

    assert _threshold(distances, 0.8) == 0.9
    assert _threshold(distances, 0.0) == 0.1
    assert _threshold(distances, 1.0) == 0.9
    assert _threshold([], 0.5) == 1.0


def test_the_shipped_percentile_leaves_only_the_top_distances_above_the_cutoff() -> None:
    """The default decides how many boundaries every semantic document gets."""
    assert _threshold([float(index) for index in range(40)]) == 38.0


def test_sentence_tiles_cover_the_text_exactly_and_keep_their_separator() -> None:
    """A tile that drops its trailing whitespace puts a hole in the tiling."""
    assert _sentences("One. Two! Three?\n\nFour") == [(0, 5), (5, 10), (10, 18), (18, 22)]
    assert _sentences("One. Two. ") == [(0, 5), (5, 10)]


def test_the_distance_ignores_magnitude_and_separates_opposites() -> None:
    """Boundaries come from direction; letting length in would break on long sentences."""
    assert _distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert _distance([3.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert _distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert _distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)
    assert _distance([0.0, 0.0], [1.0, 0.0]) == 1.0
    assert _distance([1.0, 0.0], [3.0, 0.0]) == pytest.approx(0.0)
    assert _distance([3.0, 4.0], [0.0, 2.0]) == pytest.approx(0.2)


def test_a_distance_between_vectors_of_different_widths_is_refused() -> None:
    """A prefix distance is a plausible number, so the widths have to be compared.

    Measured 2026-11-08: over the common prefix these two vectors score 0.0, identical
    topics, and the boundary the chunker then skips is invisible everywhere downstream.
    """
    with pytest.raises(EmbedError) as caught:
        _distance([1.0, 0.0, 0.0], [1.0, 0.0])

    assert "3-dimension" in caught.value.detail
    assert "2-dimension" in caught.value.detail
    assert caught.value.code is ErrorCode.EMBED_PROVIDER_ERROR
    assert caught.value.retryable is False
    assert isinstance(caught.value, IngestionError)


def test_a_distance_is_refused_whichever_vector_is_the_short_one() -> None:
    """Only the left vector is reached by indexing, so the mirrored case needs its own case."""
    with pytest.raises(EmbedError, match="2-dimension vector beside a 3-dimension one"):
        _distance([1.0, 0.0], [1.0, 0.0, 0.0])


class RaggedEmbedder:
    """Returns one narrower vector mid-document, as a truncated provider reply would."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if index != 2 else [1.0] for index in range(len(texts))]


def test_a_ragged_embedder_fails_the_document_instead_of_chunking_it_anyway() -> None:
    """The guard has to hold on the path a corpus actually takes, not only in isolation.

    The worker pool dead-letters an IngestionError, so the document is reported; a
    prefix distance would instead have chunked it at the wrong places, silently.
    """
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")

    with pytest.raises(EmbedError, match="different widths"):
        SemanticChunker(
            RaggedEmbedder(), chunk_size=200, overlap=0, counter=EstimatingTokenCounter()
        ).split(document)


class ShortReplyEmbedder:
    """Answers every sentence but the last two, as a truncated provider reply does."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in range(len(texts) - 2)]


class ExtraReplyEmbedder:
    """Answers with one vector more than it was asked for."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in range(len(texts) + 1)]


def test_a_short_embedder_reply_fails_the_document_instead_of_chunking_its_tail_blind() -> None:
    """An unanswered sentence used to merge into the last chunk with no error at all.

    Measured 2026-11-11: the two unanswered sentences were merged into the last chunk,
    ``[(0, 11), (11, 45)]``, because an out-of-range distance read as "no shift here".
    """
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")

    with pytest.raises(EmbedError) as caught:
        SemanticChunker(
            ShortReplyEmbedder(), chunk_size=200, overlap=0, counter=EstimatingTokenCounter()
        ).split(document)

    assert "2 vectors" in caught.value.detail
    assert "4 sentences" in caught.value.detail
    assert caught.value.code is ErrorCode.EMBED_PROVIDER_ERROR
    assert caught.value.retryable is False
    assert isinstance(caught.value, IngestionError)


def test_an_embedder_reply_with_a_vector_too_many_is_refused_as_well() -> None:
    """The count is compared, not bounded: an extra vector means the reply is misaligned.

    Without the mirrored case a ``!=`` weakened to ``<`` keeps passing, and a provider
    that answers for a sentence nobody sent goes on placing boundaries.
    """
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")

    with pytest.raises(EmbedError, match="5 vectors for 4 sentences"):
        SemanticChunker(
            ExtraReplyEmbedder(), chunk_size=200, overlap=0, counter=EstimatingTokenCounter()
        ).split(document)


def test_the_semantic_boundary_lands_on_the_sentence_the_distance_spikes_at() -> None:
    """Which sentence opens the next chunk is the feature; a chunk count is not."""
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")

    chunks = SemanticChunker(
        ShiftingEmbedder(), chunk_size=200, overlap=0, counter=EstimatingTokenCounter()
    ).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 23), (23, 45)]


def test_a_semantic_group_stops_growing_when_it_reaches_the_character_limit() -> None:
    """The size guard is measured from the group's own start, not from the previous end."""
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")

    chunks = SemanticChunker(
        ShiftingEmbedder(), chunk_size=5, overlap=0, counter=EstimatingTokenCounter()
    ).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 11), (11, 23), (23, 34), (34, 45)]


def test_a_heading_opens_a_new_chunk_even_when_the_previous_one_has_room() -> None:
    """Breaking at headings is what layout awareness buys; a merge here erases it."""
    document = parse_markdown(b"# Title\n\nalpha\n\n## Section A\n\nbeta\n")

    chunks = LayoutChunker(chunk_size=500, overlap=0, counter=EstimatingTokenCounter()).split(
        document
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 14), (14, 29)]


def test_layout_packs_following_blocks_against_the_chunk_it_is_filling() -> None:
    """Measuring from the previous block's end instead swallows the whole document."""
    source = "## S\n\n" + "\n\n".join(letter * 20 for letter in "abcd") + "\n"
    document = parse_markdown(source.encode())

    chunks = LayoutChunker(chunk_size=15, overlap=0, counter=EstimatingTokenCounter()).split(
        document
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 47), (47, 89)]


@pytest.mark.xfail(
    strict=True,
    reason="TASK-0300: LayoutChunker never runs the packing pass, so a block larger than "
    "chunk_size degrades to one chunk per separator occurrence",
)
def test_layout_packs_an_oversized_block_back_up_to_the_chunk_size() -> None:
    """Measured 2026-11-04: 121 chunks of ~50 characters where recursive produces 3 full ones."""
    budget = 768
    prose = "The meal allowance is forty-one pounds per day for UK travel. " * 100
    document = parse_markdown(("## Policy\n\n" + prose).encode())

    chunks = LayoutChunker(chunk_size=budget, overlap=64, counter=EstimatingTokenCounter()).split(
        document
    )

    assert min(chunk.token_count for chunk in chunks[1:-1]) > budget // 2


def test_an_injected_counter_reaches_the_chunker_the_factory_builds() -> None:
    """A counter the factory drops leaves chunk_size measured by something nobody chose."""
    settings = Settings.model_validate({"chunking": {"strategy": "fixed"}})
    counter = RecordingCounter()

    chunks = create_chunker(settings, counter=counter).split(
        parse_plaintext(b"alpha beta gamma delta epsilon")
    )

    assert chunks
    assert counter.counted
    assert counter.sized


def test_the_semantic_strategy_ships_the_same_window_and_overlap_as_the_others() -> None:
    """Semantic takes its own defaults, and a deployment that configures nothing gets them.

    The body carries no sentence punctuation, so the whole document is one tile and the
    boundaries below are the 768-token limit and the 64-token reach-back, nothing else.
    """
    document = parse_plaintext(DEFAULT_LIMIT_BODY.encode())

    chunks = SemanticChunker(ShiftingEmbedder()).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks[:2]] == [(0, 3070), (2814, 6140)]


def test_a_semantic_group_that_lands_exactly_on_the_limit_still_takes_the_next_sentence() -> None:
    """Refusing the exact fit cuts a sentence off every chunk that happens to land on it."""
    document = parse_plaintext(b"AAAA. BBBB. CCCC. DDDD.")

    chunks = SemanticChunker(
        ShiftingEmbedder(), chunk_size=3, overlap=0, counter=EstimatingTokenCounter()
    ).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 12), (12, 23)]


def test_a_blank_segment_is_absorbed_by_its_neighbour_not_by_the_first_chunk() -> None:
    """Absorbing into the first chunk makes the last one restate the whole document."""
    text = "alphabeta   "

    chunks = assemble(
        text,
        [(0, 5), (5, 9), (9, 12)],
        overlap_chars=0,
        strategy="fixed",
        counter=EstimatingTokenCounter(),
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 5), (5, 12)]
    assert [chunk.text for chunk in chunks] == ["alpha", "beta   "]


def test_a_leading_blank_segment_is_given_to_the_chunk_that_follows_it() -> None:
    """A blank segment with no predecessor has nowhere to merge back to, only forward.

    Dropping it opens a hole at offset zero, which breaks invariant 1: the chunks no
    longer reconstruct the source text.
    """
    text = "    alpha   beta "

    chunks = assemble(
        text,
        [(0, 4), (4, 9), (9, 17)],
        overlap_chars=0,
        strategy="fixed",
        counter=EstimatingTokenCounter(),
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 9), (9, 17)]
    assert "".join(chunk.text for chunk in chunks) == text


def test_a_document_that_opens_on_blank_segments_still_starts_at_its_first_character() -> None:
    """Reachable from a shipped strategy, so the leading hole is not a theoretical concern.

    Two blank segments precede the text here: holding only the first one's start is what
    keeps the chunk anchored at offset zero rather than at the second blank.
    """
    text = " " * 33 + "alpha beta gamma delta epsilon zeta eta"
    document = ParsedDocument(text=text, blocks=(), mime_type="text/plain", parser="test")

    chunks = FixedChunker(chunk_size=4, overlap=0, counter=EstimatingTokenCounter()).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 44), (44, 56), (56, 72)]
    assert "".join(chunk.text for chunk in chunks) == text


def test_a_strategy_that_emits_a_blank_segment_keeps_its_tiling() -> None:
    """Reachable from a shipped strategy, so the merge anchor is not a theoretical concern.

    ``RecursiveChunker`` splits the run of spaces off as a segment of its own here; anchoring
    the merge on the first chunk instead of the previous one makes chunk 1 restate chunk 0.
    """
    document = parse_plaintext(b"Rate. Limit.   Policy. Notes.")

    chunks = RecursiveChunker(chunk_size=2, overlap=0, counter=EstimatingTokenCounter()).split(
        document
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 6), (6, 15), (15, 23), (23, 29)]


class GlyphCounter:
    """Charges two tokens per character, as a tokenizer does on an unfamiliar script."""

    def count(self, text: str) -> int:
        return 2 * len(text.strip())

    @property
    def chars_per_token(self) -> int:
        return 4


def test_the_budget_pass_stops_halving_on_the_minimum_limit() -> None:
    """The floor is the only thing between a text no size satisfies and endless halving.

    Forty characters that count a token each against a budget of one: the pass halves
    40 to 20, then 20 to the floor, and must stop there rather than recurse on 16.
    """
    text = "a" * 40

    segments = within_budget(
        text, [(0, 40)], counter=DenseCounter(), budget=1, limit=40, split=split_span
    )

    assert segments == [(0, 16), (16, 20), (20, 36), (36, 40)]


def test_a_reach_back_the_budget_cannot_afford_is_halved_away_entirely() -> None:
    """Stopping one character short leaves an overlap the token budget has no room for."""
    chunks = assemble(
        "alphabeta",
        [(0, 5), (5, 9)],
        overlap_chars=4,
        overlap_tokens=1,
        strategy="fixed",
        counter=GlyphCounter(),
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 5), (5, 9)]
    assert [chunk.text for chunk in chunks] == ["alpha", "beta"]


class UnitCounter:
    """Counts one token per character, so a configured size is a character count."""

    def count(self, text: str) -> int:
        return len(text.strip())

    @property
    def chars_per_token(self) -> int:
        return 1


def test_a_document_without_blocks_is_chunked_from_its_first_character() -> None:
    """The fallback path takes its own start offset, and losing it shifts every chunk id.

    A parser that recovers no structure is the only way into this branch, so it needs a
    document built with no blocks rather than a plaintext one, which always has them.
    """
    document = ParsedDocument(
        text="alpha beta gamma delta", blocks=(), mime_type="text/plain", parser="stub"
    )

    chunks = LayoutChunker().split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 22)]
    assert chunks[0].text == "alpha beta gamma delta"


def test_layout_merges_a_block_that_lands_exactly_on_the_chunk_size() -> None:
    """Refusing the exact fit starts a new chunk for a block that was always going to fit."""
    document = parse_markdown(b"## S\n\naaaa\n\nbbbb\n\ncccc\n")

    chunks = LayoutChunker(chunk_size=9, overlap=0, counter=UnitCounter()).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 9), (9, 15), (15, 19)]


def test_the_layout_strategy_ships_the_same_window_and_overlap_as_the_others() -> None:
    """A deployment that configures nothing gets 768 tokens with 64 of overlap here too.

    Both sections fit the default window, so the only boundary the defaults decide is
    the second chunk's 256-character reach-back into the first.
    """
    body = " ".join(["constitutional"] * 30)
    source = f"## Section A\n\n{body}\n\n## Section B\n\n{body}\n"

    chunks = LayoutChunker().split(parse_markdown(source.encode()))

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 462), (206, 922)]


def test_a_lookup_that_is_not_callable_is_ignored_rather_than_called() -> None:
    """Both lookups arrive as plain objects, so the guard is all that stops a TypeError."""
    chunks = assemble(
        WORDS,
        [(0, 11), (11, 22)],
        overlap_chars=0,
        strategy="fixed",
        counter=EstimatingTokenCounter(),
        page_at=7,
        section_at="Section A",
    )

    assert [chunk.page for chunk in chunks] == [None, None]
    assert [chunk.section for chunk in chunks] == [None, None]


class CountingEmbedder(ShiftingEmbedder):
    """Shifts like its parent, and records whether it was consulted at all."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        return super().embed(texts)


def test_three_sentences_are_grouped_by_similarity_rather_than_packed() -> None:
    """Three is the shipped floor, and below it the embedder is never asked at all.

    Raising the floor turns a three-sentence document into one packed chunk with no
    topic boundary in it, and no failure anywhere to say the model was skipped.
    """
    embedder = CountingEmbedder()
    document = parse_plaintext(b"First one. Second one. Third one.")

    chunks = SemanticChunker(
        embedder, chunk_size=200, overlap=0, counter=EstimatingTokenCounter()
    ).split(document)

    assert embedder.calls == 1
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 23), (23, 33)]


class LateShiftingEmbedder:
    """Reports a topic change at the final sentence, where the loop's last index decides."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0]] * (len(texts) - 1) + [[0.0, 1.0]]


class TwoTileEmbedder:
    """Shifts at the third sentence, for documents that never reach three."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if index < 2 else [0.0, 1.0] for index in range(len(texts))]


def test_a_shift_at_the_last_sentence_still_opens_a_chunk() -> None:
    """The final tile is the one index the grouping loop can silently drop.

    Dropping it merges a changed topic back into the chunk before it; starting the
    loop one tile early instead restates the first tile and breaks the tiling.
    """
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")

    chunks = SemanticChunker(
        LateShiftingEmbedder(), chunk_size=200, overlap=0, counter=EstimatingTokenCounter()
    ).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 34), (34, 45)]


def test_a_document_below_the_sentence_floor_is_packed_from_its_own_tiles() -> None:
    """Packing the whole span instead loses the one sentence boundary the document has."""
    document = parse_plaintext(b"AAAAAAAAAAAAAAAAAAAA. B.")

    chunks = SemanticChunker(
        TwoTileEmbedder(), chunk_size=4, overlap=0, counter=EstimatingTokenCounter()
    ).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 16), (16, 22), (22, 24)]


def test_a_chunk_carries_no_attribute_dictionary() -> None:
    """Slots are what keep a chunk to its declared fields as it crosses a pool boundary."""
    chunk = TextChunk(text="a", start=0, end=1, chunk_index=0, token_count=1, strategy="fixed")

    assert not hasattr(chunk, "__dict__")
    assert hasattr(TextChunk, "__slots__")
