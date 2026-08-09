import re
from collections.abc import AsyncIterator
from typing import Any

import pytest

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.observability import metrics
from fasterrag.services.estimation import (
    EMBEDDING_PRICES,
    GENERATION_PRICES,
    LOCAL_PROVIDERS,
    PRICES_DATED,
    price_for,
    price_generation,
)
from fasterrag.services.generation import GenerationService
from fasterrag.services.querying import FULL_MODE, Retrieval

PROMPT_TOKENS = 1000
COMPLETION_TOKENS = 500
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class StubLLM(LLMAdapter):
    """Answers with fixed text and fixed token usage so the cost arithmetic is checkable."""

    provider = "openai"

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        return Completion(
            text="Either party may terminate [^c_a].",
            model=self.model,
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            finish_reason="stop",
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        yield "Either party may terminate [^c_a]."

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class GatewayLLM(StubLLM):
    """An OpenAI-compatible gateway: same model names, prices its vendor never published."""

    provider = "openai_compatible"


class OllamaLLM(StubLLM):
    """A locally served model, which costs the operator nothing per token."""

    provider = "ollama"


class StubRetrieval:
    """Returns one fixed chunk so a query reaches the generator."""

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        chunk = ScoredChunk(
            chunk_id="c_a", text="Either party may terminate.", payload={}, rrf_score=0.5
        )
        return Retrieval(chunks=[chunk], mode=FULL_MODE)


def build(adapter: type[StubLLM] = StubLLM, model: str = "gpt-4o-mini") -> GenerationService:
    settings = Settings.model_validate({"llm": {"model": model}})
    return GenerationService(
        settings,
        StubRetrieval(),  # type: ignore[arg-type]
        adapter(settings),
    )


def test_prompt_and_completion_are_priced_at_their_own_rates() -> None:
    """One rate for both would charge the answer at the prompt rate on every query."""
    entry = GENERATION_PRICES["gpt-4o-mini"]

    cost = price_generation("openai", "gpt-4o-mini", 1_000_000, 1_000_000)

    assert entry.output_usd_per_million is not None
    assert cost == entry.input_usd_per_million + entry.output_usd_per_million


def test_a_published_anthropic_rate_is_applied() -> None:
    cost = price_generation("anthropic", "claude-opus-5", 1_000_000, 1_000_000)

    assert cost == pytest.approx(30.00)


def test_a_published_cohere_rate_is_applied() -> None:
    cost = price_generation("cohere", "command-r-plus-08-2024", 1_000_000, 1_000_000)

    assert cost == pytest.approx(12.50)


def test_the_output_rate_is_the_higher_of_the_two() -> None:
    """Every published generation price charges more for tokens written than read."""
    for model, entry in GENERATION_PRICES.items():
        assert entry.output_usd_per_million is not None, model
        assert entry.output_usd_per_million > entry.input_usd_per_million, model


def test_every_price_records_where_and_when_it_was_checked() -> None:
    """A rate with no source and no date is a rumour, and the policy calls that a defect."""
    for model, entry in {**EMBEDDING_PRICES, **GENERATION_PRICES}.items():
        assert entry.source.strip(), model
        assert ISO_DATE.match(entry.checked), model
        assert entry.provider, model


def test_the_table_date_cannot_claim_a_freshness_no_entry_has() -> None:
    """Derived from the oldest row, so re-checking one vendor never re-dates a stale one."""
    checked = [entry.checked for entry in {**EMBEDDING_PRICES, **GENERATION_PRICES}.values()]

    assert min(checked) == PRICES_DATED


def test_a_rate_is_not_applied_across_providers() -> None:
    """A gateway serving `gpt-4o` sets its own prices; OpenAI's rate would be invented."""
    assert price_generation("openai", "gpt-4o", 1000, 1000) is not None
    assert price_generation("openai_compatible", "gpt-4o", 1000, 1000) is None


def test_an_unpriced_model_costs_nothing_rather_than_a_guess() -> None:
    assert price_generation("openai", "some-model-nobody-priced", 1000, 1000) is None


def test_a_local_provider_is_free() -> None:
    assert price_generation("ollama", "llama3", 1_000_000, 1_000_000) == 0.0


def test_a_locally_served_model_is_free_by_decision_not_by_omission() -> None:
    """Zero is the recorded price for a local model, not a gap the unpriced counter should see."""
    assert set(LOCAL_PROVIDERS) == {"huggingface", "ollama"}
    assert price_for("ollama", "nomic-embed-text", 1_000_000) == (
        0.0,
        "runs locally, so no provider charge",
    )


def test_the_embedding_pricer_does_not_know_generation_models() -> None:
    """The two tables are separate on purpose; sharing one would misprice both."""
    cost, basis = price_for("openai", "gpt-4o-mini", 1_000_000)

    assert cost is None
    assert "no published price" in basis


def test_the_basis_cites_the_source_and_the_date() -> None:
    _, basis = price_for("openai", "text-embedding-3-small", 1_000_000)

    assert "developers.openai.com" in basis
    assert EMBEDDING_PRICES["text-embedding-3-small"].checked in basis


def test_a_caveat_on_a_rate_reaches_the_basis_an_operator_reads() -> None:
    """A figure the vendor has stopped restating should not read as freshly confirmed."""
    entry = EMBEDDING_PRICES["embed-english-v3.0"]
    _, basis = price_for("cohere", "embed-english-v3.0", 1_000_000)

    assert entry.note
    assert entry.note in basis


async def test_a_priced_model_adds_its_list_price_to_the_cost_total() -> None:
    before = metrics.COST.value(provider="openai", tenant="none")
    unpriced_before = metrics.UNPRICED_TOKENS.value(provider="openai", model="gpt-4o-mini")

    await build().answer("q")

    expected = (PROMPT_TOKENS * 0.15 + COMPLETION_TOKENS * 0.60) / 1_000_000
    assert metrics.COST.value(provider="openai", tenant="none") == pytest.approx(before + expected)
    assert metrics.UNPRICED_TOKENS.value(provider="openai", model="gpt-4o-mini") == unpriced_before


async def test_an_unknown_model_is_counted_as_unpriced_and_adds_no_guess() -> None:
    """The gap has to be visible: a silent zero makes the panel look healthy while it lies."""
    model = "some-model-nobody-priced"
    cost_before = metrics.COST.value(provider="openai", tenant="none")
    before = metrics.UNPRICED_TOKENS.value(provider="openai", model=model)

    await build(model=model).answer("q")

    after = metrics.UNPRICED_TOKENS.value(provider="openai", model=model)
    assert after == before + PROMPT_TOKENS + COMPLETION_TOKENS
    assert metrics.COST.value(provider="openai", tenant="none") == cost_before


async def test_a_known_model_on_an_unknown_gateway_is_counted_rather_than_priced() -> None:
    provider = "openai_compatible"
    before = metrics.UNPRICED_TOKENS.value(provider=provider, model="gpt-4o-mini")

    await build(GatewayLLM).answer("q")

    after = metrics.UNPRICED_TOKENS.value(provider=provider, model="gpt-4o-mini")
    assert after == before + PROMPT_TOKENS + COMPLETION_TOKENS


async def test_a_local_model_adds_to_neither_counter() -> None:
    """Free is an answer, so local traffic is neither a cost nor an unpriced gap."""
    cost_before = metrics.COST.value(provider="ollama", tenant="none")
    unpriced_before = metrics.UNPRICED_TOKENS.value(provider="ollama", model="llama3.1")

    await build(OllamaLLM, model="llama3.1").answer("q")

    assert metrics.COST.value(provider="ollama", tenant="none") == cost_before
    assert metrics.UNPRICED_TOKENS.value(provider="ollama", model="llama3.1") == unpriced_before
