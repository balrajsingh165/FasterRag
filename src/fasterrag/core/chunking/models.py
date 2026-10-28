"""Chunk structures, token counting, and the shared assembler that enforces invariants.

Every strategy produces *segments* that tile the parsed text exactly, and this module
turns them into chunks. Centralizing that step is what makes the five invariants in
``docs/testing-strategy.md`` §1.2 hold for every chunker rather than for whichever one
was written most carefully:

1. Concatenating chunks, minus the configured overlap, reconstructs the source text.
2. No chunk is empty.
3. Character offsets are monotonic and in bounds.
4. The configured overlap is respected.
5. No chunk exceeds ``chunk_size`` beyond the tokenizer-boundary tolerance.

Because chunks are exact slices — ``chunk.text == document.text[chunk.start:chunk.end]``
— a citation's span resolves back to real characters in the parsed document, which is
what makes span-level citations (D5) possible at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

__all__ = [
    "CHARS_PER_TOKEN",
    "MINIMUM_LIMIT",
    "EstimatingTokenCounter",
    "Segment",
    "TextChunk",
    "TokenCounter",
    "assemble",
    "hard_split",
    "within_budget",
]

CHARS_PER_TOKEN: Final = 4

# The character limit below which the budget pass stops halving. Without a floor a text the
# counter never reports as fitting — a single glyph tokenizing to several tokens — would
# recurse until the segments were one character wide.
MINIMUM_LIMIT: Final = 16

Segment = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One chunk of a parsed document, before ids and embeddings are attached."""

    text: str
    start: int
    end: int
    chunk_index: int
    token_count: int
    strategy: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenCounter(Protocol):
    """Counts tokens in a string."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...

    @property
    def chars_per_token(self) -> int:
        """Return the average characters per token, used to size character windows."""
        ...


class EstimatingTokenCounter:
    """Approximates token counts at roughly four characters per token."""

    def __init__(self, chars_per_token: int = CHARS_PER_TOKEN) -> None:
        """Build the estimator."""
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Return the estimated token count, never less than the word count."""
        stripped = text.strip()
        if not stripped:
            return 0
        return max(len(stripped) // self._chars_per_token, len(stripped.split()))

    @property
    def chars_per_token(self) -> int:
        """Return the assumed characters per token."""
        return self._chars_per_token


def hard_split(text: str, start: int, limit: int) -> list[Segment]:
    """Cut an oversized run of text into ``limit``-sized pieces at whitespace where possible.

    The fallback that keeps invariant 5 true: an atom with no usable separator inside it
    — a long table row, an unbroken token stream — still has to fit.
    """
    segments: list[Segment] = []
    offset = 0
    length = len(text)

    while offset < length:
        end = min(offset + limit, length)
        if end < length:
            window = text.rfind(" ", offset + limit // 2, end)
            if window > offset:
                end = window + 1
        segments.append((start + offset, start + end))
        offset = end

    return segments


def assemble(
    text: str,
    segments: Sequence[Segment],
    *,
    overlap_chars: int,
    strategy: str,
    counter: TokenCounter,
    overlap_tokens: int | None = None,
    page_at: object = None,
    section_at: object = None,
) -> list[TextChunk]:
    """Turn tiling segments into chunks, applying overlap and dropping blank ones."""
    merged = _merge_blank(text, segments)
    chunks: list[TextChunk] = []

    for index, (start, end) in enumerate(merged):
        reach_back = (
            0 if index == 0 else _reach_back(text, start, overlap_chars, overlap_tokens, counter)
        )
        chunk_start = start - reach_back
        chunk_text = text[chunk_start:end]

        chunks.append(
            TextChunk(
                text=chunk_text,
                start=chunk_start,
                end=end,
                chunk_index=index,
                token_count=counter.count(chunk_text),
                strategy=strategy,
                page=_lookup_int(page_at, start),
                section=_lookup_str(section_at, start),
            )
        )

    return chunks


def _reach_back(
    text: str,
    start: int,
    overlap_chars: int,
    overlap_tokens: int | None,
    counter: TokenCounter,
) -> int:
    """Return how many characters a chunk should reach into its predecessor."""
    reach = min(overlap_chars, start)
    if overlap_tokens is None:
        return reach

    while reach > 0 and counter.count(text[start - reach : start]) > overlap_tokens:
        reach //= 2

    return reach


def _merge_blank(text: str, segments: Sequence[Segment]) -> list[Segment]:
    """Merge whitespace-only segments into their predecessor, preserving the tiling."""
    merged: list[Segment] = []
    for start, end in segments:
        if end <= start:
            continue
        if not text[start:end].strip() and merged:
            previous_start, _ = merged[-1]
            merged[-1] = (previous_start, end)
            continue
        if not text[start:end].strip():
            continue
        merged.append((start, end))
    return merged


def _lookup_int(lookup: object, offset: int) -> int | None:
    """Call an offset-to-page lookup if one was supplied."""
    if lookup is None or not callable(lookup):
        return None
    value = lookup(offset)
    return value if isinstance(value, int) else None


def _lookup_str(lookup: object, offset: int) -> str | None:
    """Call an offset-to-section lookup if one was supplied."""
    if lookup is None or not callable(lookup):
        return None
    value = lookup(offset)
    return value if isinstance(value, str) else None


def within_budget(
    text: str,
    segments: Sequence[Segment],
    *,
    counter: TokenCounter,
    budget: int,
    limit: int,
    split: Callable[[str, int, int, int], list[Segment]],
) -> list[Segment]:
    """Re-split any segment the counter says exceeds ``budget`` tokens."""

    def refine(segment: Segment, current: int) -> list[Segment]:
        start, end = segment
        if current <= MINIMUM_LIMIT or counter.count(text[start:end]) <= budget:
            return [segment]

        halved = max(current // 2, MINIMUM_LIMIT)
        return [
            piece for finer in split(text, start, end, halved) for piece in refine(finer, halved)
        ]

    return [piece for segment in segments for piece in refine(segment, limit)]
