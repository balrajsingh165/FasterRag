"""Preflight cost estimation (D9).

Answers "what will this corpus cost to ingest?" *before* anything is embedded. Everyone
else discovers embedding costs on the invoice; making spend visible in advance is what
turns it into a decision rather than a surprise (``docs/differentiators.md`` D9).

The estimate parses and chunks for real rather than guessing from file size, because token
count depends on the chunker: overlap duplicates text, and contextual enrichment adds to
every chunk. Only embedding is skipped, which is the expensive part being estimated.

Two honesty rules govern what comes back:

* **Prices are list prices, not measurements.** Every entry carries its own source and the
  date it was read, and a rate is only ever applied to the provider it was published for. A
  model the tables do not know reports ``None`` rather than a plausible-looking number.
* **Wall-clock time is not projected.** Throughput has not been measured on any reference
  hardware yet, so there is no number to project from. Publishing one would be a claim
  without a measurement (``docs/benchmarks.md``).

The price table and ``price_for``/``price_generation`` live in :mod:`fasterrag.core.pricing`
since 2026-11-15 (TASK-0529), because the embedding workers price every batch and may not
import a service. They are re-exported here, as the same objects, so imports from this module
keep working.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.pricing import (
    EMBEDDING_PRICES,
    GENERATION_PRICES,
    LOCAL_PROVIDERS,
    PRICES_DATED,
    TokenPrice,
    price_for,
    price_generation,
)
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.workers.cpu_pool import CpuWorkerPool, parse_and_chunk

__all__ = [
    "EMBEDDING_PRICES",
    "GENERATION_PRICES",
    "LOCAL_PROVIDERS",
    "PRICES_DATED",
    "Estimate",
    "ProviderEstimate",
    "TokenPrice",
    "estimate_sources",
    "price_for",
    "price_generation",
    "require_estimator",
]


# CRITICAL: the estimator must not load a tokenizer. It runs before ingestion to answer
# "what will this cost", and pulling a multi-gigabyte model to answer that would make the
# cheap preflight the expensive step. Four characters per token is the usual English
# approximation and the basis string says the figure is an estimate.
_CHARS_PER_TOKEN: Final = 4

_logger = get_logger(__name__)


def require_estimator(settings: Settings) -> None:
    """Reject preflight estimation when ``cost.estimator`` has switched it off."""
    if not settings.cost.estimator:
        raise FasterRagError(
            "preflight cost estimation is disabled; set cost.estimator to true to enable it",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )


@dataclass(frozen=True, slots=True)
class ProviderEstimate:
    """What one provider would charge to embed the corpus."""

    provider: str
    model: str
    tokens: int
    cost_usd: float | None
    basis: str

    @property
    def known(self) -> bool:
        """Return whether a cost could be established at all."""
        return self.cost_usd is not None

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable form."""
        return {
            "provider": self.provider,
            "model": self.model,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class Estimate:
    """What ingesting a set of sources would involve."""

    documents: int
    unreadable: int
    chunks: int
    tokens: int
    bytes_read: int
    providers: list[ProviderEstimate] = field(default_factory=list)
    parse_seconds: float = 0.0
    projected_seconds: float | None = None
    projection_note: str = ""
    enrichment: EnrichmentEstimate | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable report, the shape `--json` and the API use."""
        return {
            "documents": self.documents,
            "unreadable": self.unreadable,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "bytes": self.bytes_read,
            "parse_seconds": round(self.parse_seconds, 3),
            "projected_seconds": self.projected_seconds,
            "projection_note": self.projection_note,
            "prices_dated": PRICES_DATED,
            "providers": [provider.as_dict() for provider in self.providers],
            "enrichment": self.enrichment.as_dict() if self.enrichment else None,
        }


@dataclass(frozen=True, slots=True)
class EnrichmentEstimate:
    """What contextual enrichment (P2) would add to an ingest."""

    calls: int
    prompt_tokens: int
    completion_tokens: int
    model: str
    cost_usd: float | None
    basis: str

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable form."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "basis": self.basis,
        }


def estimate_enrichment(
    settings: Settings, *, chunks: int, document_tokens: int, chunk_tokens: int
) -> EnrichmentEstimate:
    """Return what enrichment would cost for a corpus of this shape."""
    completion_tokens = chunks * settings.chunking.context_tokens
    prompt_tokens = document_tokens + chunk_tokens
    model = settings.llm.model
    provider = settings.llm.provider

    cost = price_generation(provider, model, prompt_tokens, completion_tokens)
    basis = (
        f"{chunks} call(s) at list price for {model!r}, uncached; prompt caching reduces "
        f"this by an amount that depends on the provider and is not estimated here"
        if cost is not None
        else f"no published generation price recorded for {model!r} on {provider!r}"
    )

    return EnrichmentEstimate(
        calls=chunks,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        cost_usd=cost,
        basis=basis,
    )


def _providers_to_price(settings: Settings, all_providers: bool) -> list[tuple[str, str]]:
    """Return the provider and model pairs to price."""
    configured: tuple[str, str] = (settings.embeddings.provider, settings.embeddings.model)
    if not all_providers:
        return [configured]

    pairs: list[tuple[str, str]] = [configured]
    for model, entry in sorted(EMBEDDING_PRICES.items()):
        if (entry.provider, model) != configured:
            pairs.append((entry.provider, model))
    return pairs


def estimate_sources(
    sources: Sequence[str],
    settings: Settings,
    *,
    all_providers: bool = False,
) -> Estimate:
    """Parse and chunk ``sources`` to report what ingesting them would involve."""
    tasks = CpuWorkerPool.tasks_for(list(sources))
    started = time.perf_counter()

    chunks = 0
    tokens = 0
    read = 0
    unreadable = 0
    enrichment_prompt_tokens = 0
    enrichment_chunk_tokens = 0

    for task in tasks:
        try:
            outcome = parse_and_chunk(task, settings)
        except FasterRagError as exc:
            unreadable += 1
            _logger.info(
                "source could not be estimated",
                extra={"source": task.source, "code": exc.code.value},
            )
            continue

        chunks += len(outcome.chunks)
        document_chunk_tokens = sum(payload.chunk.token_count for payload in outcome.chunks)
        tokens += document_chunk_tokens
        read += Path(task.source).stat().st_size

        if settings.chunking.contextual_enrichment:
            document_tokens = len(outcome.document_text) // _CHARS_PER_TOKEN
            enrichment_prompt_tokens += document_tokens * len(outcome.chunks)
            enrichment_chunk_tokens += document_chunk_tokens

    elapsed = time.perf_counter() - started
    providers = []
    for provider, model in _providers_to_price(settings, all_providers):
        cost, basis = price_for(provider, model, tokens)
        providers.append(
            ProviderEstimate(
                provider=provider, model=model, tokens=tokens, cost_usd=cost, basis=basis
            )
        )

    return Estimate(
        documents=len(tasks) - unreadable,
        unreadable=unreadable,
        chunks=chunks,
        tokens=tokens,
        bytes_read=read,
        providers=providers,
        parse_seconds=elapsed,
        projected_seconds=None,
        projection_note=(
            "wall-clock time is not projected: embedding throughput has not been measured "
            "on reference hardware yet, and an unmeasured projection would be a claim "
            "without a measurement (docs/benchmarks.md)"
        ),
        enrichment=(
            estimate_enrichment(
                settings,
                chunks=chunks,
                document_tokens=enrichment_prompt_tokens,
                chunk_tokens=enrichment_chunk_tokens,
            )
            if settings.chunking.contextual_enrichment and chunks
            else None
        ),
    )
