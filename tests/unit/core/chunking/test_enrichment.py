from typing import Any

from fasterrag.adapters.llm.base import Completion
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.enrichment import (
    ENRICHMENT_FAILED_FLAG,
    P2_TEMPLATE_VERSION,
    build_enrichment_prompt,
    enrich_chunks,
    system_prompt,
)
from fasterrag.core.chunking.models import TextChunk
from fasterrag.errors import GenerationError

DOCUMENT = "# UK travel policy 2026\n\nThe meal allowance is £41 per day."


def chunk(index: int = 0, text: str = "The meal allowance is £41 per day.") -> TextChunk:
    return TextChunk(
        text=text, start=0, end=len(text), chunk_index=index, token_count=9, strategy="fixed"
    )


class StubLLM:
    """Returns a fixed context, recording what it was asked."""

    def __init__(self, reply: str = "From the UK 2026 travel policy, on meal allowances.") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.prompts.append(prompt)
        self.systems.append(system or "")
        return Completion(text=self.reply, model="stub")


class FailingLLM:
    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        raise GenerationError("the provider is down", retryable=True)


def settings(**chunking: Any) -> Settings:
    return Settings.model_validate({"chunking": chunking} if chunking else {})


async def test_the_context_is_prepended() -> None:
    result = await enrich_chunks([chunk()], DOCUMENT, StubLLM(), settings())  # type: ignore[arg-type]

    assert result[0].text.startswith("From the UK 2026 travel policy")
    assert "meal allowance is £41" in result[0].text


async def test_the_original_text_is_kept() -> None:
    """A round trip, a re-chunk, or an export all need the unprefixed body."""
    result = await enrich_chunks([chunk()], DOCUMENT, StubLLM(), settings())  # type: ignore[arg-type]

    assert result[0].metadata["original_text"] == "The meal allowance is £41 per day."
    assert result[0].metadata["context_prefix"].startswith("From the UK")


async def test_the_template_version_is_recorded() -> None:
    """A prompt change alters every embedding; the version is how that becomes visible."""
    result = await enrich_chunks([chunk()], DOCUMENT, StubLLM(), settings())  # type: ignore[arg-type]

    assert result[0].metadata["enrichment_template"] == P2_TEMPLATE_VERSION


async def test_a_failed_call_indexes_the_chunk_anyway() -> None:
    """A slightly worse chunk beats a dead-lettered document."""
    result = await enrich_chunks([chunk()], DOCUMENT, FailingLLM(), settings())  # type: ignore[arg-type]

    assert result[0].text == "The meal allowance is £41 per day."
    assert result[0].metadata[ENRICHMENT_FAILED_FLAG] is True


async def test_a_failure_is_visible_rather_than_silent() -> None:
    """An unflagged failure looks identical to enrichment being switched off."""
    result = await enrich_chunks([chunk()], DOCUMENT, FailingLLM(), settings())  # type: ignore[arg-type]

    assert ENRICHMENT_FAILED_FLAG in result[0].metadata


async def test_one_failure_does_not_lose_the_others() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
            self.calls += 1
            if self.calls == 1:
                raise GenerationError("transient", retryable=True)
            return Completion(text="a good context", model="stub")

    result = await enrich_chunks(
        [chunk(0), chunk(1), chunk(2)],
        DOCUMENT,
        Flaky(),  # type: ignore[arg-type]
        settings(),
    )

    enriched = [item for item in result if "context_prefix" in item.metadata]
    assert len(enriched) == 2


async def test_the_document_leads_the_prompt() -> None:
    """It is the cacheable prefix; anything before it turns a cache hit into a re-read."""
    llm = StubLLM()

    await enrich_chunks([chunk()], DOCUMENT, llm, settings())  # type: ignore[arg-type]

    assert llm.prompts[0].startswith("<document>")


async def test_the_token_target_follows_configuration() -> None:
    """A setting the prompt ignores is a setting that silently does nothing."""
    llm = StubLLM()

    await enrich_chunks([chunk()], DOCUMENT, llm, settings(context_tokens=120))  # type: ignore[arg-type]

    assert "about 120 tokens" in llm.systems[0]


async def test_a_model_preamble_is_stripped() -> None:
    """Otherwise it becomes part of the embedded text on every single chunk."""
    llm = StubLLM(reply='Context: "From the policy on allowances."')

    result = await enrich_chunks([chunk()], DOCUMENT, llm, settings())  # type: ignore[arg-type]

    assert result[0].metadata["context_prefix"] == "From the policy on allowances."


async def test_an_empty_reply_counts_as_a_failure() -> None:
    """A blank prefix would add a leading newline and nothing else."""
    result = await enrich_chunks([chunk()], DOCUMENT, StubLLM(reply="   "), settings())  # type: ignore[arg-type]

    assert result[0].metadata[ENRICHMENT_FAILED_FLAG] is True


async def test_no_chunks_makes_no_calls() -> None:
    llm = StubLLM()

    assert await enrich_chunks([], DOCUMENT, llm, settings()) == []  # type: ignore[arg-type]
    assert llm.prompts == []


def test_a_very_large_document_is_truncated() -> None:
    """Beyond the model's window every chunk fails rather than one."""
    prompt = build_enrichment_prompt("x" * 200_000, "a chunk")

    assert len(prompt) < 100_000


def test_the_prompt_carries_both_parts() -> None:
    prompt = build_enrichment_prompt("the doc", "the chunk")

    assert "<document>\nthe doc\n</document>" in prompt
    assert "<chunk>\nthe chunk\n</chunk>" in prompt


def test_the_system_prompt_forbids_a_preamble() -> None:
    assert "no preamble" in system_prompt(75)


async def test_a_quoted_preamble_is_stripped_in_either_nesting() -> None:
    """A model writes `Context: "..."` as readily as `"Context: ..."`."""
    for reply in ('Context: "the body."', '"Context: the body."', "'the body.'"):
        result = await enrich_chunks([chunk()], DOCUMENT, StubLLM(reply=reply), settings())  # type: ignore[arg-type]
        assert result[0].metadata["context_prefix"] == "the body."
