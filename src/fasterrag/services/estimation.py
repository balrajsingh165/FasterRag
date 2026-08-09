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
]


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """One model's published list price, carrying the provenance that makes it checkable.

    A rate without a source and a date is not a price, it is a rumour — last quarter's number
    bills nothing today. Provenance therefore lives on the entry rather than on the table: a
    single table-wide date is only ever as true as its oldest row, so re-checking one vendor
    would silently re-date every stale row beside it. With the date on the row, a price nobody
    has re-verified says so, and the rest of the table stays trustworthy.

    ``provider`` is part of the identity, not decoration. The same model id served through a
    different provider is a different bill, so a rate is applied only to the provider it was
    published for; anything else is charged at a price its vendor never quoted.

    Attributes:
        provider: The configured provider value this rate was published for.
        input_usd_per_million: USD per million input (prompt) tokens.
        output_usd_per_million: USD per million output tokens, or ``None`` for an embedding
            model, which produces no billable output tokens.
        source: Where the figure came from, precisely enough to re-check it.
        checked: ISO date the figure was last read from ``source``.
        note: Any caveat that makes the bare number misleading on its own.
    """

    provider: str
    input_usd_per_million: float
    output_usd_per_million: float | None
    source: str
    checked: str
    note: str = ""


_OPENAI_SOURCE: Final = "OpenAI API pricing, developers.openai.com/api/docs/pricing"
_ANTHROPIC_SOURCE: Final = (
    "Anthropic models overview, platform.claude.com/docs/en/about-claude/models/overview"
)
_COHERE_SOURCE: Final = "Cohere pricing, cohere.com/pricing"


def _table(
    provider: str,
    source: str,
    checked: str,
    rates: dict[str, tuple[float, float | None]],
    notes: dict[str, str] | None = None,
) -> dict[str, TokenPrice]:
    """Expand one provider's published rates into entries that each carry their provenance."""
    notes = notes or {}
    return {
        model: TokenPrice(
            provider=provider,
            input_usd_per_million=rates_pair[0],
            output_usd_per_million=rates_pair[1],
            source=source,
            checked=checked,
            note=notes.get(model, ""),
        )
        for model, rates_pair in rates.items()
    }


# CRITICAL: every rate below is a *published list price*, not a measurement, and every entry
# must carry the source and date it was read. A model with no citable rate stays out of these
# tables: it is then counted by ``fasterrag_unpriced_tokens_total`` and visibly excluded from
# the cost total, which is strictly better than a fabricated number producing a confident
# wrong bill (docs/observability.md, and the provable-claims policy in CLAUDE.md).
EMBEDDING_PRICES: Final[dict[str, TokenPrice]] = {
    **_table(
        "openai",
        _OPENAI_SOURCE,
        "2026-08-09",
        {
            "text-embedding-3-small": (0.02, None),
            "text-embedding-3-large": (0.13, None),
            "text-embedding-ada-002": (0.10, None),
        },
    ),
    **_table(
        "cohere",
        _COHERE_SOURCE,
        "2026-07-30",
        {
            "embed-english-v3.0": (0.10, None),
            "embed-multilingual-v3.0": (0.10, None),
            "embed-english-light-v3.0": (0.10, None),
        },
        dict.fromkeys(
            ("embed-english-v3.0", "embed-multilingual-v3.0", "embed-english-light-v3.0"),
            "a re-check on 2026-08-09 found no per-token embed rate stated on the pricing "
            "page, so this figure stands at its original date rather than a fresher one",
        ),
    ),
}

# Two rates per generation model, not one: a model charges differently for the prompt it reads
# and the answer it writes, so pricing a completion at the input rate understates every query.
GENERATION_PRICES: Final[dict[str, TokenPrice]] = {
    **_table(
        "openai",
        _OPENAI_SOURCE,
        "2026-08-09",
        {
            "gpt-5.6-sol": (5.00, 30.00),
            "gpt-5.6-terra": (2.00, 12.00),
            "gpt-5.6-luna": (0.20, 1.20),
            "gpt-5.5": (5.00, 30.00),
            "gpt-5.5-pro": (30.00, 180.00),
            "gpt-5.4": (2.50, 15.00),
            "gpt-5.4-mini": (0.75, 4.50),
            "gpt-5.4-nano": (0.20, 1.25),
            "gpt-5.4-pro": (30.00, 180.00),
            "gpt-5.2": (1.75, 14.00),
            "gpt-5.2-pro": (21.00, 168.00),
            "gpt-5.1": (1.25, 10.00),
            "gpt-5": (1.25, 10.00),
            "gpt-5-mini": (0.25, 2.00),
            "gpt-5-nano": (0.05, 0.40),
            "gpt-5-pro": (15.00, 120.00),
            "gpt-4.1": (2.00, 8.00),
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1-nano": (0.10, 0.40),
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-2024-05-13": (5.00, 15.00),
            "gpt-4o-mini": (0.15, 0.60),
            "o1": (15.00, 60.00),
            "o1-pro": (150.00, 600.00),
            "o3": (2.00, 8.00),
            "o3-pro": (20.00, 80.00),
            "o3-mini": (1.10, 4.40),
            "o4-mini": (1.10, 4.40),
            "gpt-4-turbo-2024-04-09": (10.00, 30.00),
            "gpt-4-0613": (30.00, 60.00),
            "gpt-3.5-turbo": (0.50, 1.50),
            "gpt-3.5-turbo-0125": (0.50, 1.50),
            "gpt-3.5-turbo-1106": (1.00, 2.00),
        },
        {
            "gpt-5.5": "published rate covers requests under 272K context",
            "gpt-5.5-pro": "published rate covers requests under 272K context",
            "gpt-5.4": "published rate covers requests under 272K context",
            "gpt-5.4-pro": "published rate covers requests under 272K context",
        },
    ),
    **_table(
        "anthropic",
        _ANTHROPIC_SOURCE,
        "2026-08-09",
        {
            "claude-fable-5": (10.00, 50.00),
            "claude-mythos-5": (10.00, 50.00),
            "claude-opus-5": (5.00, 25.00),
            "claude-sonnet-5": (3.00, 15.00),
            "claude-haiku-4-5": (1.00, 5.00),
            "claude-haiku-4-5-20251001": (1.00, 5.00),
            "claude-opus-4-8": (5.00, 25.00),
            "claude-opus-4-7": (5.00, 25.00),
            "claude-opus-4-6": (5.00, 25.00),
            "claude-opus-4-5": (5.00, 25.00),
            "claude-opus-4-5-20251101": (5.00, 25.00),
            "claude-sonnet-4-6": (3.00, 15.00),
            "claude-sonnet-4-5": (3.00, 15.00),
            "claude-sonnet-4-5-20250929": (3.00, 15.00),
        },
        {
            "claude-sonnet-5": (
                "standard rate; introductory 2.00/10.00 applies through 2026-08-31, so this "
                "over-estimates until then"
            ),
            "claude-mythos-5": "published as sharing Claude Fable 5's pricing",
        },
    ),
    **_table(
        "cohere",
        _COHERE_SOURCE,
        "2026-08-09",
        {"command-r-plus-08-2024": (2.50, 10.00)},
    ),
}

PRICES_DATED: Final = min(
    entry.checked for entry in (*EMBEDDING_PRICES.values(), *GENERATION_PRICES.values())
)
"""Oldest per-entry check date: the staleness floor for the tables as a whole.

Derived rather than declared, so it cannot claim a freshness no entry has. The per-entry
``checked`` dates are authoritative; this is what a report shows when it has room for one date.
"""

# CRITICAL: zero is a deliberate, defensible price here, not a missing one. A locally served
# model incurs no provider charge, so recording it as free is a fact — whereas leaving these
# providers out of the tables would count every local token as unpriced and imply a hidden
# bill that does not exist. Hardware and electricity are real costs, but they are not a
# per-token rate fasterRag can know, so they are out of scope rather than invented.
LOCAL_PROVIDERS: Final[frozenset[str]] = frozenset({"huggingface", "ollama"})

_MILLION: Final = 1_000_000

# CRITICAL: the estimator must not load a tokenizer. It runs before ingestion to answer
# "what will this cost", and pulling a multi-gigabyte model to answer that would make the
# cheap preflight the expensive step. Four characters per token is the usual English
# approximation and the basis string says the figure is an estimate.
_CHARS_PER_TOKEN: Final = 4

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
    """What contextual enrichment (P2) would add to an ingest.

    Reported separately from the embedding cost rather than folded into it. Enrichment is a
    *generation* charge on a different model at different rates, and a single blended number
    would hide which knob to turn when the total looks wrong.
    """

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
    """Return what enrichment would cost for a corpus of this shape.

    One call per chunk, each sending the whole parent document plus the chunk. The document
    dominates, which is why P2 sends it as a cacheable prefix.

    # CRITICAL: this is the *uncached* figure and the basis says so. Prompt caching is what
    # makes enrichment affordable, but the discount depends on the provider, the cache
    # window, and how many chunks a document has — quoting a discounted number fasterRag
    # cannot verify would understate a real bill. An over-estimate an operator can reason
    # about beats an under-estimate they discover on an invoice.
    """
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


def _rate_for(table: dict[str, TokenPrice], provider: str, model: str) -> TokenPrice | None:
    """Return the published rate for ``model`` on ``provider``, or ``None`` if there is none.

    The provider has to match. A rate published by one vendor says nothing about what another
    charges for a model of the same name, and an OpenAI-compatible gateway sets its own prices
    entirely — billing either at OpenAI's rate would be inventing a figure, which is the one
    thing this module must never do.
    """
    entry = table.get(model)
    if entry is None or entry.provider != provider:
        return None
    return entry


def price_for(provider: str, model: str, tokens: int) -> tuple[float | None, str]:
    """Return the cost of embedding ``tokens`` and the basis for that figure.

    Returns:
        The cost in USD and a short explanation. A local provider costs nothing; an
        unknown hosted model returns ``None`` rather than a fabricated price.
    """
    if provider in LOCAL_PROVIDERS:
        return 0.0, "runs locally, so no provider charge"

    entry = _rate_for(EMBEDDING_PRICES, provider, model)
    if entry is None:
        return None, f"no published price recorded for {model!r} on {provider!r}"

    basis = (
        f"{entry.input_usd_per_million} USD per million tokens, "
        f"{entry.source}, checked {entry.checked}"
    )
    if entry.note:
        basis = f"{basis} ({entry.note})"
    return tokens / _MILLION * entry.input_usd_per_million, basis


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

    entry = _rate_for(GENERATION_PRICES, provider, model)
    if entry is None or entry.output_usd_per_million is None:
        return None

    charged = (
        prompt_tokens * entry.input_usd_per_million
        + completion_tokens * entry.output_usd_per_million
    )
    return charged / _MILLION


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

        # Enrichment sends the whole parent document once per chunk, so the prompt cost is
        # the document's own token count multiplied by how many chunks it produced — the
        # term that makes this expensive, and the one a per-chunk figure would hide.
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
