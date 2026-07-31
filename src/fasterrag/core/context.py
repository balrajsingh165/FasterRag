"""Context assembly: pack retrieved chunks into a prompt, with citations.

The stage between retrieval and generation, and the one that decides what the model
actually sees. Three jobs, each addressing a documented failure:

* **Budgeting.** Chunks are packed in relevance order until the token budget is spent.
  Overflowing a context window does not fail loudly — providers truncate, usually from the
  middle, so the model silently loses the passage that mattered.
* **Deduplication.** Chunk overlap means neighbouring chunks share text by construction, and
  retrieval frequently returns several of them. Spending a budget on the same sentences
  three times is how a context window fills with nothing (pain point 18, context rot).
* **Citations.** Every chunk that reaches the prompt yields a citation carrying its source,
  page, and character span, so an answer can be traced to the exact text it came from.

Citations are built here rather than after generation on purpose: reconstructing which
passage an answer came from by re-reading the answer is guesswork, whereas recording what
was *sent* is fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from fasterrag.core.chunking.models import EstimatingTokenCounter, TokenCounter
from fasterrag.core.retrieval.models import ScoredChunk

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "AssembledContext",
    "Citation",
    "Span",
    "assemble_context",
]

DEFAULT_SIMILARITY_THRESHOLD: Final = 0.9

_SEPARATOR: Final = "\n\n"
_MINIMUM_TOKENS_PER_CHUNK: Final = 1


@dataclass(frozen=True, slots=True)
class Span:
    """A character range inside the parsed document."""

    start: int
    end: int

    def as_dict(self) -> dict[str, int]:
        """Return the serialized form."""
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class Citation:
    """Where one piece of the context came from."""

    chunk_id: str
    source: str | None = None
    page: int | None = None
    span: Span | None = None
    score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the response form, omitting what is not known."""
        payload: dict[str, Any] = {"chunk_id": self.chunk_id}
        if self.source is not None:
            payload["source"] = self.source
        if self.page is not None:
            payload["page"] = self.page
        if self.span is not None:
            payload["span"] = self.span.as_dict()
        if self.score is not None:
            payload["score"] = round(self.score, 6)
        return payload


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The prompt context and the provenance of everything in it."""

    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    tokens: int = 0
    used: int = 0
    dropped_duplicate: int = 0
    dropped_budget: int = 0

    @property
    def empty(self) -> bool:
        """Return whether nothing survived assembly."""
        return not self.citations

    @property
    def truncated(self) -> bool:
        """Return whether the budget, rather than the candidate list, ended packing."""
        return self.dropped_budget > 0


def _citation_for(chunk: ScoredChunk) -> Citation:
    """Build a citation from what the indexer recorded alongside the vector."""
    payload = chunk.payload
    span_payload = payload.get("span")
    span = None
    if isinstance(span_payload, dict):
        start, end = span_payload.get("start"), span_payload.get("end")
        if isinstance(start, int) and isinstance(end, int):
            span = Span(start=start, end=end)

    page = payload.get("page")
    score = chunk.rerank_score if chunk.rerank_score is not None else chunk.rrf_score

    return Citation(
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        page=page if isinstance(page, int) else None,
        span=span,
        score=score,
    )


def _terms(text: str) -> frozenset[str]:
    """Return a chunk's lowercased word set, the basis for near-duplicate detection."""
    return frozenset(text.lower().split())


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the Jaccard overlap of two term sets."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / len(left | right)


def assemble_context(
    chunks: Sequence[ScoredChunk],
    *,
    budget_tokens: int,
    counter: TokenCounter | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> AssembledContext:
    """Pack chunks into a context that fits the budget, dropping duplicates.

    Chunks are consumed in the order given, which is relevance order after reranking, so a
    budget that runs out drops the least relevant material rather than an arbitrary tail.

    Args:
        chunks: Retrieved chunks, best first.
        budget_tokens: Tokens available for context. The caller derives this from the
            model's window minus the space reserved for the answer; there is no
            configuration key for a provider's window size.
        counter: Token counter; defaults to the estimating counter.
        similarity_threshold: Term-overlap ratio above which a chunk is considered a
            near-duplicate of one already packed. ``1.0`` disables near-duplicate dropping
            and keeps only exact-text deduplication.

    Returns:
        The assembled context, with a citation per included chunk and counts of what was
        dropped and why.
    """
    tokens = counter or EstimatingTokenCounter()
    packed: list[str] = []
    citations: list[Citation] = []
    seen: list[frozenset[str]] = []
    seen_text: set[str] = set()

    used_tokens = 0
    duplicates = 0
    over_budget = 0

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue

        if text in seen_text:
            duplicates += 1
            continue

        terms = _terms(text)
        if any(_similarity(terms, other) >= similarity_threshold for other in seen):
            duplicates += 1
            continue

        cost = max(tokens.count(text), _MINIMUM_TOKENS_PER_CHUNK)
        if used_tokens + cost > budget_tokens:
            over_budget += 1
            continue

        packed.append(text)
        citations.append(_citation_for(chunk))
        seen.append(terms)
        seen_text.add(text)
        used_tokens += cost

    return AssembledContext(
        text=_SEPARATOR.join(packed),
        citations=citations,
        tokens=used_tokens,
        used=len(packed),
        dropped_duplicate=duplicates,
        dropped_budget=over_budget,
    )
