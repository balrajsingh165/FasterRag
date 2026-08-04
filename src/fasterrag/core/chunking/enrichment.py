"""Contextual enrichment (P2, ``docs/prompts.md``).

Prepends a short document-level context to each chunk before it is embedded and indexed —
the Contextual Retrieval technique, which the source measures at -49% failed retrievals and
-67% combined with reranking (``docs/references.md`` R1). A chunk that says "the limit was
raised to £41" retrieves badly; the same chunk prefixed with "From the UK 2026 travel expense
policy, meal allowances" retrieves on the terms a question actually uses.

**The parent document leads every prompt, unchanged.** That is not incidental formatting: it
makes the document a stable cacheable prefix, so a provider's prompt cache absorbs it across
every chunk of that document. Enrichment is one call per chunk, and without that cache the
cost scales with document length times chunk count rather than with chunk count alone.

**A failed enrichment is never fatal.** The chunk is indexed unprefixed and flagged, because
a slightly worse chunk beats a dead-lettered document — the feature exists to improve
retrieval, and letting it decide whether a document survives inverts that.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Final

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import TextChunk
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import get_logger

__all__ = [
    "ENRICHMENT_FAILED_FLAG",
    "P2_SYSTEM_PROMPT",
    "P2_TEMPLATE_VERSION",
    "build_enrichment_prompt",
    "enrich_chunks",
    "system_prompt",
]

P2_TEMPLATE_VERSION: Final = "p2.1.0"


def system_prompt(context_tokens: int) -> str:
    """Return the P2 system prompt, sized by ``chunking.context_tokens``.

    The target is interpolated rather than hardcoded because the key exists and is
    documented as valid from 25 to 150. A setting the prompt ignores is a setting that
    silently does nothing, which is worse than not offering it.
    """
    return (
        "You write a short situating context for a chunk of a larger document, so the "
        "chunk\ncan be retrieved and understood on its own.\n\n"
        "Output the context only — no preamble, no explanation, no quotation marks.\n"
        f"Target about {context_tokens} tokens. State what section this is from, what "
        "entity or subject it\nconcerns, and any referent a pronoun or shorthand in the "
        "chunk depends on.\n"
        "Do not summarize the chunk itself and do not add information absent from the "
        "document."
    )


P2_SYSTEM_PROMPT: Final = system_prompt(75)

ENRICHMENT_FAILED_FLAG: Final = "enrichment_failed"

# CRITICAL: a ceiling on how much of the parent document is sent. A very large document would
# otherwise be resent in full for every one of its chunks — the prompt cache makes that cheap
# on the provider's side but not free, and a document beyond the model's window fails every
# chunk rather than one.
_MAX_DOCUMENT_CHARS: Final = 60_000

_logger = get_logger(__name__)


def build_enrichment_prompt(document: str, chunk: str) -> str:
    """Return the P2 user prompt.

    The document comes first and verbatim. Reordering it, or interpolating anything before
    it, breaks the cacheable prefix and turns a cache hit into a full re-read on every chunk.
    """
    parent = document[:_MAX_DOCUMENT_CHARS]
    return f"<document>\n{parent}\n</document>\n\n<chunk>\n{chunk}\n</chunk>"


def _clean(raw: str) -> str:
    """Strip the preamble and quoting a model adds despite being told not to.

    Instructions reduce this; they do not eliminate it. An unstripped ``"Here is the
    context:"`` becomes part of the embedded text on every chunk, which is noise added to
    the exact field the feature exists to improve.
    """
    openers = ("Here is the context:", "Situating context:", "Context:", "context:")
    text = raw.strip()

    # Looped, because the two forms nest: a model writes `Context: "…"` as readily as
    # `"Context: …"`, and stripping in one fixed order leaves the other's residue behind —
    # a stray quote that then travels into every embedding.
    for _ in range(4):
        before = text
        for opener in openers:
            if text.startswith(opener):
                text = text[len(opener) :].strip()
                break
        text = text.strip('"').strip("'").strip()
        if text == before:
            break

    return " ".join(text.split())


async def enrich_chunks(
    chunks: list[TextChunk],
    document: str,
    llm: LLMAdapter,
    settings: Settings,
) -> list[TextChunk]:
    """Return the chunks with a situating context prepended to each.

    Args:
        chunks: The document's chunks, in order.
        document: The whole parent document, used as the cacheable prompt prefix.
        llm: The model writing the contexts.
        settings: Validated configuration; ``chunking.context_tokens`` sizes the output.

    Returns:
        New chunks whose ``text`` carries the prefix and whose ``metadata`` records the
        unprefixed original plus the prefix itself. A chunk whose enrichment failed is
        returned unchanged apart from a flag.

    The calls run concurrently: enrichment is one provider round trip per chunk, and doing
    them in sequence would make ingestion latency the sum of every chunk's call.
    """
    if not chunks:
        return []

    instructions = system_prompt(settings.chunking.context_tokens)
    results = await asyncio.gather(
        *(_context_for(chunk, document, llm, instructions) for chunk in chunks),
        return_exceptions=False,
    )

    enriched: list[TextChunk] = []
    failures = 0
    for chunk, context in zip(chunks, results, strict=True):
        if not context:
            failures += 1
            enriched.append(
                replace(chunk, metadata={**chunk.metadata, ENRICHMENT_FAILED_FLAG: True})
            )
            continue

        enriched.append(
            replace(
                chunk,
                text=f"{context}\n\n{chunk.text}",
                metadata={
                    **chunk.metadata,
                    "context_prefix": context,
                    "original_text": chunk.text,
                    "enrichment_template": P2_TEMPLATE_VERSION,
                },
            )
        )

    _logger.info(
        "enriched a document's chunks",
        extra={"chunks": len(chunks), "failed": failures, "template": P2_TEMPLATE_VERSION},
    )
    return enriched


async def _context_for(chunk: TextChunk, document: str, llm: LLMAdapter, instructions: str) -> str:
    """Return one chunk's context, or an empty string when the call fails.

    # CRITICAL: every failure returns empty rather than raising. Enrichment improves a chunk;
    # it must never decide whether the chunk exists. A provider outage during ingestion would
    # otherwise dead-letter an entire corpus that would have indexed perfectly well unenriched.
    """
    try:
        completion = await llm.complete(
            build_enrichment_prompt(document, chunk.text), system=instructions
        )
    except FasterRagError as exc:
        _logger.warning(
            "contextual enrichment failed; the chunk is indexed without a prefix",
            extra={"chunk_index": chunk.chunk_index, "code": exc.code.value},
        )
        return ""

    return _clean(completion.text)
