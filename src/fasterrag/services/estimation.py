"""Preflight cost estimation (D9).

Answers "what will this corpus cost to ingest?" *before* anything is embedded. Everyone
else discovers embedding costs on the invoice; making spend visible in advance is what
turns it into a decision rather than a surprise (``docs/differentiators.md`` D9).

The estimate parses and chunks for real rather than guessing from file size, because token
count depends on the chunker: overlap duplicates text, and contextual enrichment adds to
every chunk. Only embedding is skipped, which is the expensive part being estimated.

Two honesty rules govern what comes back:

* **Prices are list prices, not measurements.** They are dated, sourced, and only applied
  to models this table actually knows. An unknown model reports ``None`` rather than a
  plausible-looking number.
* **Wall-clock time is not projected.** Throughput has not been measured on any reference
  hardware yet, so there is no number to project from. Publishing one would be a claim
  without a measurement (``docs/benchmarks.md``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.workers.cpu_pool import CpuWorkerPool, parse_and_chunk

__all__ = [
    "GENERATION_PRICES_USD_PER_MILLION_TOKENS",
    "PRICES_DATED",
    "PRICES_SOURCE",
    "PRICES_USD_PER_MILLION_TOKENS",
    "Estimate",
    "ProviderEstimate",
    "estimate_sources",
    "price_for",
    "price_generation",
]

# Published list prices in USD per million input tokens. These are not measurements and
# they go stale: a model absent from this table reports an unknown cost rather than a guess.
PRICES_USD_PER_MILLION_TOKENS: Final[dict[str, float]] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
    "embed-english-v3.0": 0.10,
    "embed-multilingual-v3.0": 0.10,
    "embed-english-light-v3.0": 0.10,
}

# Published list prices in USD per million tokens for *generation*, as (input, output).
# Two rates, not one: a generation model charges differently for the prompt it reads and the
# answer it writes, so pricing a completion at the input rate understates every query. A
# model absent from this table contributes no cost rather than a guess — which is why the
# cost panel can read zero on a working system, and says so on the panel itself.
GENERATION_PRICES_USD_PER_MILLION_TOKENS: Final[dict[str, tuple[float, float]]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

PRICES_DATED: Final = "2026-07-30"
PRICES_SOURCE: Final = "provider public pricing pages"

LOCAL_PROVIDERS: Final[frozenset[str]] = frozenset({"huggingface", "ollama"})

_MILLION: Final = 1_000_000

_logger = get_logger(__name__)


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
        }


def price_for(provider: str, model: str, tokens: int) -> tuple[float | None, str]:
    """Return the cost of embedding ``tokens`` and the basis for that figure.

    Returns:
        The cost in USD and a short explanation. A local provider costs nothing; an
        unknown hosted model returns ``None`` rather than a fabricated price.
    """
    if provider in LOCAL_PROVIDERS:
        return 0.0, "runs locally, so no provider charge"

    rate = PRICES_USD_PER_MILLION_TOKENS.get(model)
    if rate is None:
        return None, f"no published price recorded for {model!r} as of {PRICES_DATED}"

    return (
        tokens / _MILLION * rate,
        f"{rate} USD per million tokens, {PRICES_SOURCE} {PRICES_DATED}",
    )


def price_generation(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return the list-price estimate for one generation call, or ``None`` if unpriced.

    Separate from :func:`price_for` because a generation model has two rates and an
    embedding model has one. Calling the embedding pricer on a completion would charge the
    answer at the prompt rate, which understates every query by the margin between them.

    Returns:
        The estimated cost in USD, ``0.0`` for a local provider, or ``None`` when no rate is
        recorded for the model — never a fabricated figure.
    """
    if provider in LOCAL_PROVIDERS:
        return 0.0

    rates = GENERATION_PRICES_USD_PER_MILLION_TOKENS.get(model)
    if rates is None:
        return None

    input_rate, output_rate = rates
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / _MILLION


def _providers_to_price(settings: Settings, all_providers: bool) -> list[tuple[str, str]]:
    """Return the provider and model pairs to price."""
    configured: tuple[str, str] = (settings.embeddings.provider, settings.embeddings.model)
    if not all_providers:
        return [configured]

    pairs: list[tuple[str, str]] = [configured]
    for model in sorted(PRICES_USD_PER_MILLION_TOKENS):
        provider = "cohere" if model.startswith("embed-") else "openai"
        if (provider, model) != configured:
            pairs.append((provider, model))
    return pairs


def estimate_sources(
    sources: Sequence[str],
    settings: Settings,
    *,
    all_providers: bool = False,
) -> Estimate:
    """Parse and chunk ``sources`` to report what ingesting them would involve.

    Args:
        sources: Paths to estimate.
        settings: Validated configuration; the chunker settings decide the token count.
        all_providers: Also price every model with a published rate, so the cost of
            switching provider is visible before committing to one.

    Returns:
        The estimate. Documents that cannot be parsed are counted as unreadable rather
        than raising, because an estimate over a messy corpus is still useful.
    """
    tasks = CpuWorkerPool.tasks_for(list(sources))
    started = time.perf_counter()

    chunks = 0
    tokens = 0
    read = 0
    unreadable = 0

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
        tokens += sum(payload.chunk.token_count for payload in outcome.chunks)
        read += Path(task.source).stat().st_size

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
    )
