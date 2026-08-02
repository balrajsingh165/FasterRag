from fasterrag.services.estimation import (
    GENERATION_PRICES_USD_PER_MILLION_TOKENS,
    price_for,
    price_generation,
)


def test_prompt_and_completion_are_priced_at_their_own_rates() -> None:
    """One rate for both would charge the answer at the prompt rate on every query."""
    input_rate, output_rate = GENERATION_PRICES_USD_PER_MILLION_TOKENS["gpt-4o-mini"]

    cost = price_generation("openai", "gpt-4o-mini", 1_000_000, 1_000_000)

    assert cost == input_rate + output_rate


def test_the_output_rate_is_the_higher_of_the_two() -> None:
    """Every published generation price charges more for tokens written than read."""
    for model, (input_rate, output_rate) in GENERATION_PRICES_USD_PER_MILLION_TOKENS.items():
        assert output_rate > input_rate, model


def test_an_unpriced_model_costs_nothing_rather_than_a_guess() -> None:
    assert price_generation("openai", "some-model-nobody-priced", 1000, 1000) is None


def test_a_local_provider_is_free() -> None:
    assert price_generation("ollama", "llama3", 1_000_000, 1_000_000) == 0.0


def test_the_embedding_pricer_does_not_know_generation_models() -> None:
    """The two tables are separate on purpose; sharing one would misprice both."""
    cost, basis = price_for("openai", "gpt-4o-mini", 1_000_000)

    assert cost is None
    assert "no published price" in basis
