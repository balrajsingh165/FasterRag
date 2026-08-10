from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.estimation import (
    EMBEDDING_PRICES,
    estimate_sources,
    price_for,
    require_estimator,
)

BODY = "# Vendor Agreement\n\n" + ("sentence body text here. " * 60)


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {"chunking": {"chunk_size": 64, "overlap": 8}}
    payload.update(overrides)
    return Settings.model_validate(payload)


@pytest.fixture
def corpus(tmp_path: Path) -> list[str]:
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(BODY, encoding="utf-8")
    return [str(tmp_path / "a.md"), str(tmp_path / "b.md")]


def test_a_local_provider_costs_nothing() -> None:
    cost, basis = price_for("huggingface", "BAAI/bge-small-en-v1.5", 1_000_000)

    assert cost == 0.0
    assert "locally" in basis


def test_a_known_model_is_priced_from_the_published_rate() -> None:
    cost, basis = price_for("openai", "text-embedding-3-small", 1_000_000)

    assert cost == pytest.approx(0.02)
    assert "per million tokens" in basis


def test_cost_scales_with_tokens() -> None:
    cost, _ = price_for("openai", "text-embedding-3-large", 500_000)

    assert cost == pytest.approx(
        EMBEDDING_PRICES["text-embedding-3-large"].input_usd_per_million / 2
    )


def test_an_unknown_model_reports_no_price_rather_than_guessing() -> None:
    cost, basis = price_for("openai", "some-future-model", 1_000_000)

    assert cost is None
    assert "no published price" in basis


def test_an_estimate_counts_documents_chunks_and_tokens(corpus: list[str]) -> None:
    estimate = estimate_sources(corpus, settings())

    assert estimate.documents == 2
    assert estimate.unreadable == 0
    assert estimate.chunks > 2
    assert estimate.tokens > 0
    assert estimate.bytes_read > 0


def test_the_token_count_follows_the_chunker(corpus: list[str]) -> None:
    without_overlap = estimate_sources(corpus, settings(chunking={"chunk_size": 64, "overlap": 0}))
    with_overlap = estimate_sources(corpus, settings(chunking={"chunk_size": 64, "overlap": 32}))

    assert with_overlap.tokens > without_overlap.tokens


def test_the_configured_provider_is_priced(corpus: list[str]) -> None:
    estimate = estimate_sources(corpus, settings())

    assert len(estimate.providers) == 1
    assert estimate.providers[0].provider == "huggingface"
    assert estimate.providers[0].cost_usd == 0.0


def test_every_priced_model_can_be_compared(corpus: list[str]) -> None:
    estimate = estimate_sources(corpus, settings(), all_providers=True)

    models = {provider.model for provider in estimate.providers}
    assert "text-embedding-3-small" in models
    assert "embed-english-v3.0" in models
    assert all(provider.tokens == estimate.tokens for provider in estimate.providers)


def test_a_hosted_provider_shows_a_real_cost(corpus: list[str]) -> None:
    configured = settings(
        embeddings={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
        }
    )

    estimate = estimate_sources(corpus, configured)

    assert estimate.providers[0].cost_usd is not None
    assert estimate.providers[0].cost_usd > 0
    assert estimate.providers[0].known is True


def test_an_unreadable_source_is_counted_not_raised(corpus: list[str], tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"PK\x03\x04 not a document")

    estimate = estimate_sources([*corpus, str(broken)], settings())

    assert estimate.unreadable == 1
    assert estimate.documents == 2


def test_an_empty_corpus_estimates_zero() -> None:
    estimate = estimate_sources([], settings())

    assert estimate.documents == 0
    assert estimate.tokens == 0
    assert estimate.chunks == 0


def test_wall_clock_time_is_not_projected_without_a_measurement(corpus: list[str]) -> None:
    estimate = estimate_sources(corpus, settings())

    assert estimate.projected_seconds is None
    assert "has not been measured" in estimate.projection_note


def test_the_parse_time_actually_spent_is_reported(corpus: list[str]) -> None:
    estimate = estimate_sources(corpus, settings())

    assert estimate.parse_seconds > 0


def test_the_report_serializes_for_json_output(corpus: list[str]) -> None:
    payload = estimate_sources(corpus, settings()).as_dict()

    assert payload["documents"] == 2
    assert payload["projected_seconds"] is None
    assert payload["prices_dated"]
    assert payload["providers"][0]["provider"] == "huggingface"


def test_nothing_is_embedded_while_estimating(
    corpus: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the estimator must not embed anything")

    monkeypatch.setattr("fasterrag.adapters.embeddings.huggingface.load_model", refuse)

    assert estimate_sources(corpus, settings()).tokens > 0


def test_a_disabled_estimator_is_refused_by_the_name_of_its_setting() -> None:
    """``cost.estimator`` was declared, documented, and read by nothing (TASK-0200).

    The refusal names the setting, because that is the only thing telling an operator which
    switch produced it; a bare "disabled" leaves them grepping for it.
    """
    with pytest.raises(FasterRagError) as caught:
        require_estimator(settings(cost={"estimator": False}))

    assert caught.value.code == ErrorCode.VALIDATION_FAILED
    assert "cost.estimator" in caught.value.detail


def test_an_enabled_estimator_passes_the_guard() -> None:
    """The gate must refuse the off position only; a guard that always raised would too."""
    require_estimator(settings())


def test_the_throughput_measurement_survives_the_estimator_being_off(corpus: list[str]) -> None:
    """The one deliberate exemption, pinned here so it stays deliberate.

    ``benchmark --suite ingest`` reuses this function to time parse-and-chunk and reports no
    cost at all, so switching cost estimation off must not take the benchmark harness with
    it. That is why the gate sits at the D9 entry points rather than inside this function.
    """
    estimate = estimate_sources(corpus, settings(cost={"estimator": False}))

    assert estimate.documents == 2
