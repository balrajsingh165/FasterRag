from pathlib import Path

from fasterrag.config.schema import Settings
from fasterrag.services.estimation import estimate_enrichment, estimate_sources


def corpus(tmp_path: Path, documents: int = 1) -> list[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(documents):
        source = tmp_path / f"doc{index}.md"
        source.write_text(f"# Policy {index}\n\n" + ("body text. " * 60), encoding="utf-8")
        paths.append(str(source))
    return paths


def settings(enabled: bool, **chunking: object) -> Settings:
    return Settings.model_validate({"chunking": {"contextual_enrichment": enabled, **chunking}})


def test_no_enrichment_block_when_the_toggle_is_off(tmp_path: Path) -> None:
    estimate = estimate_sources(corpus(tmp_path), settings(False))

    assert estimate.enrichment is None


def test_the_toggle_on_reports_the_extra_calls(tmp_path: Path) -> None:
    """D9's claim is that spend is visible before it is committed."""
    estimate = estimate_sources(corpus(tmp_path), settings(True))

    assert estimate.enrichment is not None
    assert estimate.enrichment.calls == estimate.chunks


def test_the_prompt_cost_scales_with_documents_times_chunks(tmp_path: Path) -> None:
    """The parent document is sent once per chunk; that is the expensive term."""
    one = estimate_sources(corpus(tmp_path / "a", 1), settings(True))
    three = estimate_sources(corpus(tmp_path / "b", 3), settings(True))

    assert one.enrichment is not None
    assert three.enrichment is not None
    assert three.enrichment.prompt_tokens > one.enrichment.prompt_tokens


def test_the_completion_budget_follows_the_configured_target() -> None:
    small = estimate_enrichment(
        settings(True, context_tokens=25), chunks=10, document_tokens=100, chunk_tokens=50
    )
    large = estimate_enrichment(
        settings(True, context_tokens=150), chunks=10, document_tokens=100, chunk_tokens=50
    )

    assert small.completion_tokens == 250
    assert large.completion_tokens == 1500


def test_the_basis_says_the_figure_is_uncached() -> None:
    """Prompt caching is what makes enrichment affordable, and its discount is unverifiable.

    Quoting a discounted number fasterRag cannot verify would understate a real bill; an
    over-estimate an operator can reason about is the safer error.
    """
    estimate = estimate_enrichment(settings(True), chunks=5, document_tokens=1000, chunk_tokens=200)

    assert "uncached" in estimate.basis


def test_an_unpriced_model_reports_unknown_rather_than_zero() -> None:
    """Zero would read as free, which is the one wrong answer worse than no answer."""
    unpriced = Settings.model_validate(
        {"chunking": {"contextual_enrichment": True}, "llm": {"model": "some-unlisted-model"}}
    )

    estimate = estimate_enrichment(unpriced, chunks=5, document_tokens=1000, chunk_tokens=200)

    assert estimate.cost_usd is None
    assert "no published generation price" in estimate.basis


def test_the_estimate_serialises_for_scripts(tmp_path: Path) -> None:
    payload = estimate_sources(corpus(tmp_path), settings(True)).as_dict()

    assert payload["enrichment"]["calls"] >= 1
    assert "basis" in payload["enrichment"]


def test_the_estimator_loads_no_tokenizer(tmp_path: Path) -> None:
    """A preflight that pulls a multi-gigabyte model is not a cheap preflight."""
    import sys

    before = set(sys.modules)
    estimate_sources(corpus(tmp_path), settings(True))

    assert "sentence_transformers" not in set(sys.modules) - before
