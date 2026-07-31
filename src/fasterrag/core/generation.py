"""P1 answer generation: prompt building and citation resolution.

Implements the P1 contract of ``docs/prompts.md``. Two halves:

**Building the prompt.** The stable system block goes first and the volatile context and
question last, so a provider's prompt cache can hit across queries. Each chunk carries a
``[^chunk_id]`` marker and its source, which is what lets the model cite precisely rather
than gesture at "the context".

**Resolving what came back.** The model emits inline markers; this maps them to real
citations. A marker that does not match a chunk actually supplied is **dropped and logged**
— that is invariant 2 of ``docs/data-model.md``: a citation can never reference content that
was not retrieved. A model that invents a plausible-looking marker is exactly the failure
citations exist to prevent, so an unresolvable one is never passed through.

The template carries a version. Prompt wording is quality-affecting, so a change to it runs
the regression gate like any other quality change, and the version lands in the trace so a
replay is meaningful (D8).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from fasterrag.core.context import AssembledContext, Citation
from fasterrag.observability.logging import get_logger

__all__ = [
    "P1_SYSTEM_PROMPT",
    "P1_TEMPLATE_VERSION",
    "build_context_block",
    "build_prompt",
    "citation_marker",
    "resolve_citations",
]

P1_TEMPLATE_VERSION: Final = "1.0.0"

P1_SYSTEM_PROMPT: Final = """\
You answer strictly from the provided context. If the context does not contain
enough information to answer, say so plainly instead of guessing — a partial or
absent answer is correct behavior, a fabricated one is not.

Cite the specific chunk that supports each factual claim using its marker, e.g. [^c_9f2].
Every factual sentence needs a marker. Do not cite chunks you did not use.
Do not use outside knowledge, even if you are confident it is correct.
Quote exact figures, dates, and identifiers rather than paraphrasing them."""

_MARKER = re.compile(r"\[\^([A-Za-z0-9_\-]+)\]")

_logger = get_logger(__name__)


def citation_marker(chunk_id: str) -> str:
    """Return the inline marker the model is asked to cite a chunk with."""
    return f"[^{chunk_id}]"


def _describe(citation: Citation) -> str:
    """Render a chunk's provenance line, omitting what is not known."""
    parts = []
    if citation.source:
        parts.append(f"source: {citation.source}")
    if citation.page is not None:
        parts.append(f"page: {citation.page}")
    return f" ({', '.join(parts)})" if parts else ""


def build_context_block(context: AssembledContext, texts: Sequence[str]) -> str:
    """Render the assembled context as marked, attributed passages.

    Args:
        context: The assembled context, whose citations are in packed order.
        texts: The packed chunk texts, in the same order.

    Returns:
        The block placed inside the prompt's ``<context>`` tags.
    """
    return "\n\n".join(
        f"{citation_marker(citation.chunk_id)}{_describe(citation)}\n{text}"
        for citation, text in zip(context.citations, texts, strict=False)
    )


def build_prompt(question: str, context: AssembledContext, texts: Sequence[str]) -> str:
    """Build the P1 user turn: the context block, then the question.

    The question goes last so everything above it is a cacheable prefix.
    """
    block = build_context_block(context, texts)
    return f"<context>\n{block}\n</context>\n\nQuestion: {question}"


def resolve_citations(answer: str, supplied: Sequence[Citation]) -> list[Citation]:
    """Return the citations the answer actually used, in order of first appearance.

    Args:
        answer: The generated text, containing inline markers.
        supplied: The citations that were placed in the prompt.

    Returns:
        Only citations whose markers appear in the answer and were genuinely supplied.
        A marker naming a chunk that was never in the context is dropped and logged, so a
        response can never cite content that was not retrieved.
    """
    by_id = {citation.chunk_id: citation for citation in supplied}
    resolved: list[Citation] = []
    seen: set[str] = set()
    invented: list[str] = []

    for match in _MARKER.finditer(answer):
        chunk_id = match.group(1)
        if chunk_id in seen:
            continue

        citation = by_id.get(chunk_id)
        if citation is None:
            invented.append(chunk_id)
            continue

        seen.add(chunk_id)
        resolved.append(citation)

    if invented:
        _logger.warning(
            "dropped citation markers that name chunks never supplied to the model",
            extra={"markers": invented, "supplied": len(supplied)},
        )

    return resolved
