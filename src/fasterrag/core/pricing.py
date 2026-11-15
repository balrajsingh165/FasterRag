"""Published list prices for embedding and generation models, and the arithmetic over them.

Moved out of ``services/estimation.py`` on 2026-11-15 (TASK-0529). Two readers need the table:
the preflight estimator, a service, and the embedding workers, which price every batch they
embed. A worker may not import a service, so the table lives below both. ``services.estimation``
still re-exports every name here, so existing imports keep working.

**Prices are list prices, not measurements.** Every entry carries its own source and the date
it was read, and a rate is only ever applied to the provider it was published for. A model the
tables do not know reports ``None`` rather than a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "EMBEDDING_PRICES",
    "GENERATION_PRICES",
    "LOCAL_PROVIDERS",
    "PRICES_DATED",
    "TokenPrice",
    "price_for",
    "price_generation",
]


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """One model's published list price, carrying the provenance that makes it checkable."""

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
        "2026-11-15",
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
            "re-checks on 2026-08-09 and 2026-11-15 found no per-token embed rate stated on "
            "the pricing page, so this figure stands at its original date",
        ),
    ),
}

GENERATION_PRICES: Final[dict[str, TokenPrice]] = {
    **_table(
        "openai",
        _OPENAI_SOURCE,
        "2026-11-15",
        {
            "gpt-6-astra": (10.00, 50.00),
            "gpt-6.1-sol": (2.00, 10.00),
            "gpt-6.1-terra": (1.00, 5.00),
            "gpt-6-sol": (2.00, 10.00),
            "gpt-6-luna": (0.10, 0.50),
            "gpt-5.6-sol": (4.00, 20.00),
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
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "o3": (2.00, 8.00),
            "o3-pro": (20.00, 80.00),
        },
        {
            **dict.fromkeys(
                ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-sol", "gpt-6-luna"),
                "published rate covers prompts up to 272K tokens; a longer prompt costs more",
            ),
            **dict.fromkeys(
                ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
                "published rate covers prompts up to 272K tokens; a longer prompt costs more",
            ),
            "gpt-5.5": "published rate covers requests under 272K context",
            "gpt-5.5-pro": "published rate covers requests under 272K context",
            "gpt-5.4": "published rate covers requests under 272K context",
            "gpt-5.4-pro": "published rate covers requests under 272K context",
        },
    ),
    **_table(
        "openai",
        _OPENAI_SOURCE,
        "2026-08-09",
        {
            "gpt-4.1-nano": (0.10, 0.40),
            "gpt-4o-2024-05-13": (5.00, 15.00),
            "o1": (15.00, 60.00),
            "o1-pro": (150.00, 600.00),
            "o3-mini": (1.10, 4.40),
            "o4-mini": (1.10, 4.40),
            "gpt-4-turbo-2024-04-09": (10.00, 30.00),
            "gpt-4-0613": (30.00, 60.00),
            "gpt-3.5-turbo": (0.50, 1.50),
            "gpt-3.5-turbo-0125": (0.50, 1.50),
            "gpt-3.5-turbo-1106": (1.00, 2.00),
        },
        dict.fromkeys(
            (
                "gpt-4.1-nano",
                "gpt-4o-2024-05-13",
                "o1",
                "o1-pro",
                "o3-mini",
                "o4-mini",
                "gpt-4-turbo-2024-04-09",
                "gpt-4-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo-1106",
            ),
            "no longer on the pricing page at the 2026-11-15 re-check, so this rate stands "
            "at the date it was last read rather than a fresher one",
        ),
    ),
    **_table(
        "anthropic",
        _ANTHROPIC_SOURCE,
        "2026-11-03",
        {
            "claude-fable-5-2": (8.00, 40.00),
            "claude-mythos-5-2": (8.00, 40.00),
            "claude-fable-5-1": (10.00, 50.00),
            "claude-mythos-5-1": (10.00, 50.00),
            "claude-fable-5": (10.00, 50.00),
            "claude-mythos-5": (10.00, 50.00),
            "claude-opus-5-5": (4.00, 20.00),
            "claude-opus-5": (5.00, 25.00),
            "claude-sonnet-5-5": (2.00, 10.00),
            "claude-sonnet-5": (2.00, 10.00),
            "claude-haiku-5-5": (0.10, 0.50),
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
            "claude-haiku-5-5": (
                "published rate covers prompts up to 100K tokens; 0.50/2.50 applies beyond, "
                "so this under-estimates a longer prompt"
            ),
            "claude-mythos-5-2": "published as sharing Claude Fable 5.2's pricing",
            "claude-mythos-5-1": "published as sharing Claude Fable 5.1's pricing",
            "claude-mythos-5": "published as sharing Claude Fable 5's pricing",
        },
    ),
    **_table(
        "cohere",
        _COHERE_SOURCE,
        "2026-11-15",
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


def _rate_for(table: dict[str, TokenPrice], provider: str, model: str) -> TokenPrice | None:
    """Return the published rate for ``model`` on ``provider``, or ``None`` if there is none."""
    entry = table.get(model)
    if entry is None or entry.provider != provider:
        return None
    return entry


def price_for(provider: str, model: str, tokens: int) -> tuple[float | None, str]:
    """Return the cost of embedding ``tokens`` and the basis for that figure."""
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
    """Return the list-price estimate for one generation call, or ``None`` if unpriced."""
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
