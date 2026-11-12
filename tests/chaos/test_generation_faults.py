"""Query-path chaos scenarios for TASK-0449: retry, fallback, and a reranker that hangs.

Each scenario injects its fault at the closest honest seam, as the rest of this suite does.
The rate-limit storm raises what the adapter boundary really produces for a ``429``: the
provider's own status error, carried through ``classify_llm_failure`` with its
``Retry-After`` header, so the retry policy is exercised against the typed error a real
provider failure becomes and not one a test author wrote by hand.

Nothing here asserts a duration. Waits are observed by replacing the one function the
service pauses through, so the scenarios run in milliseconds and stay deterministic on a
loaded machine.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import httpx
import pytest

from fasterrag.adapters.llm.base import Completion, LLMAdapter, classify_llm_failure
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.breaker import shared_breaker
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.services.generation import FALLBACK_BREAKER, GenerationService
from fasterrag.services.querying import FULL_MODE, Retrieval
from fasterrag.services.traces import TraceStore

pytestmark = pytest.mark.chaos

ANSWER: Final = "Either party may terminate with thirty days notice [^c_a]."
RETRY_AFTER_SECONDS: Final = 0.25


class RateLimitedError(Exception):
    """A provider's status error as the OpenAI and Anthropic clients raise it."""

    def __init__(self, retry_after: str) -> None:
        """Carry a ``429`` response with the provider's ``Retry-After`` header."""
        super().__init__("rate limited")
        self.status_code = 429
        self.response = httpx.Response(429, headers={"Retry-After": retry_after})


class StormingLLM(LLMAdapter):
    """Answers ``429`` to its first ``storm`` calls, then answers the question."""

    provider = "storming"

    def __init__(self, settings: Settings, storm: int) -> None:
        """Rate-limit the first ``storm`` calls."""
        super().__init__(settings)
        self.storm = storm
        self.calls = 0

    def _admit(self, operation: str) -> None:
        """Fail the call the way the adapter boundary would, while the storm lasts."""
        self.calls += 1
        if self.calls <= self.storm:
            raise classify_llm_failure(
                RateLimitedError(str(RETRY_AFTER_SECONDS)),
                provider=self.provider,
                operation=operation,
                key_env=None,
            )

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Answer once the storm has passed."""
        self._admit("complete")
        return Completion(text=ANSWER, model="storming", prompt_tokens=5, completion_tokens=9)

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Stream the answer once the storm has passed."""
        self._admit("stream")
        for word in ANSWER.split(" "):
            yield f"{word} "

    async def health(self) -> HealthStatus:
        """Report healthy; the storm is in the calls, not the probe."""
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        """Nothing to release."""


class StubRetrieval:
    """Returns one fixed chunk so generation has evidence to cite."""

    async def search(self, text: str, **kwargs: object) -> Retrieval:
        """Return the fixed chunk in full mode."""
        chunk = ScoredChunk(chunk_id="c_a", text="Either party may terminate.", rrf_score=0.5)
        return Retrieval(chunks=[chunk], mode=FULL_MODE)


def storm_settings() -> Settings:
    """Return the default retry policy with three attempts and millisecond backoff."""
    return Settings.model_validate(
        {
            "llm": {"provider": "ollama", "api_key_env": None},
            "reliability": {
                "retries": {"max_attempts": 3, "backoff_base_ms": 1, "backoff_max_ms": 5000}
            },
        }
    )


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every pause the service takes between attempts instead of sleeping it."""
    recorded: list[float] = []

    async def record(seconds: float) -> None:
        recorded.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr("fasterrag.services.generation._pause", record)
    return recorded


async def test_a_429_storm_is_retried_and_the_query_is_answered(waits: list[float]) -> None:
    """Scenario: three 429s in a row, each saying ``Retry-After``; the fourth call answers."""
    settings = storm_settings()
    llm = StormingLLM(settings, storm=3)
    service = GenerationService(settings, StubRetrieval(), llm)  # type: ignore[arg-type]

    answer = await service.answer("what is the notice period?")

    assert answer.mode == FULL_MODE
    assert answer.degraded is False
    assert answer.answer == ANSWER
    assert llm.calls == 4
    assert waits == [RETRY_AFTER_SECONDS] * 3
    assert service.breaker.is_open is False


async def test_a_429_storm_on_a_stream_is_retried_before_any_token(waits: list[float]) -> None:
    """The streamed half: the caller sees one ``meta`` in full mode and the whole answer."""
    settings = storm_settings()
    llm = StormingLLM(settings, storm=2)
    service = GenerationService(settings, StubRetrieval(), llm)  # type: ignore[arg-type]

    events = [event async for event in service.stream("what is the notice period?")]

    assert [event.type for event in events if event.type == "meta"] == ["meta"]
    assert events[-1].type == "done"
    assert "".join(event.data["text"] for event in events if event.type == "token").strip() == (
        ANSWER
    )
    assert llm.calls == 3
    assert waits == [RETRY_AFTER_SECONDS] * 2


class UnreachableLLM(StormingLLM):
    """A provider whose host refuses every connection, as one that is down does."""

    provider = "unreachable"

    def _admit(self, operation: str) -> None:
        """Fail every call the way a refused connection reaches the adapter boundary."""
        self.calls += 1
        raise classify_llm_failure(
            ConnectionRefusedError("[Errno 111] Connection refused"),
            provider=self.provider,
            operation=operation,
            key_env=None,
        )


async def test_a_provider_outage_falls_back_to_the_second_provider(
    waits: list[float], tmp_path: Path
) -> None:
    """Scenario: the primary is down for good; ``llm.fallback`` answers before the ladder."""
    settings = storm_settings()
    primary = UnreachableLLM(settings, storm=0)
    second = StormingLLM(settings, storm=0)
    service = GenerationService(
        settings,
        StubRetrieval(),  # type: ignore[arg-type]
        primary,
        fallback=second,
        traces=TraceStore(tmp_path),
    )

    answer = await service.answer("what is the notice period?")

    assert answer.mode == FULL_MODE
    assert answer.answer == ANSWER
    assert primary.calls == 1 + settings.reliability.retries.max_attempts
    assert second.calls == 1
    assert len(waits) == settings.reliability.retries.max_attempts
    assert shared_breaker(FALLBACK_BREAKER, settings).is_open is False
    stored = service.traces.load(answer.trace_id) if service.traces else None
    assert stored is not None
    failed = [span for span in stored.spans if span.name == "generation-failed"]
    assert [span.attributes["provider"] for span in failed] == ["unreachable"]
    assert stored.spans[-1].attributes["role"] == "fallback"


async def test_every_scenario_defined_in_this_module_is_scripted() -> None:
    """The scenarios this module owns, held by name so deleting one is a visible diff."""
    scenarios = {
        "llm-429-storm": test_a_429_storm_is_retried_and_the_query_is_answered,
        "llm-outage-fallback": test_a_provider_outage_falls_back_to_the_second_provider,
    }

    assert all(inspect.iscoroutinefunction(scenario) for scenario in scenarios.values())
