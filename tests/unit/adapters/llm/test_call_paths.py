"""What each generator actually sends and how it reads the answer back.

The existing suite covers construction, the factory, and failure classification. It does not
call ``complete`` or ``stream``, so request building and response parsing — the two places a
provider integration is most often wrong, and the two that only fail against a live API —
went unexercised.

Clients are faked. The point is the mapping between fasterRag's contract and each vendor's
shape; a real call would test the vendor's uptime.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from fasterrag.adapters.llm.anthropic import AnthropicGenerator
from fasterrag.adapters.llm.base import Completion
from fasterrag.adapters.llm.ollama import OllamaGenerator
from fasterrag.adapters.llm.openai import OpenAIGenerator
from fasterrag.config.schema import Settings
from fasterrag.errors import GenerationError


def settings(provider: str = "openai", **llm: Any) -> Settings:
    payload: dict[str, Any] = {"provider": provider, "model": "test-model", **llm}
    payload.setdefault("api_key_env", "TEST_LLM_KEY")
    if provider == "ollama":
        payload["api_key_env"] = None
    return Settings.model_validate({"llm": payload})


class Choice:
    def __init__(self, content: str | None, finish_reason: str | None = "stop") -> None:
        self.message = type("m", (), {"content": content})()
        self.finish_reason = finish_reason


class Response:
    def __init__(self, content: str | None = "an answer", **usage: int) -> None:
        self.choices = [Choice(content)]
        self.usage = type("u", (), usage)() if usage else None


class Delta:
    def __init__(self, content: str | None) -> None:
        self.delta = type("d", (), {"content": content})()


class Event:
    def __init__(self, content: str | None, *, empty: bool = False) -> None:
        self.choices = [] if empty else [Delta(content)]


class FakeCompletions:
    """Records the request and returns whatever it was primed with."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.result = result
        self.error = error

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def openai_with(result: Any = None, error: Exception | None = None) -> Any:
    """Return an adapter wired to a fake client, plus the fake completions object."""
    adapter = OpenAIGenerator(settings())
    completions = FakeCompletions(result=result, error=error)
    adapter._client = type("c", (), {"chat": type("ch", (), {"completions": completions})()})()
    return adapter, completions


async def test_the_prompt_is_sent_as_a_user_message() -> None:
    adapter, completions = openai_with(Response())

    await adapter.complete("what is the allowance")

    assert completions.requests[0]["messages"] == [
        {"role": "user", "content": "what is the allowance"}
    ]


async def test_a_system_prompt_leads_the_messages() -> None:
    """A system message after the user message is ignored by the model, silently."""
    adapter, completions = openai_with(Response())

    await adapter.complete("the question", system="be terse")

    assert completions.requests[0]["messages"][0] == {"role": "system", "content": "be terse"}


async def test_no_system_prompt_sends_no_empty_message() -> None:
    adapter, completions = openai_with(Response())

    await adapter.complete("the question")

    assert all(m["role"] != "system" for m in completions.requests[0]["messages"])


async def test_the_configured_sampling_reaches_the_request() -> None:
    """Settings the request omits are settings that silently do nothing."""
    adapter = OpenAIGenerator(settings(temperature=0.7, max_tokens=256))
    completions = FakeCompletions(result=Response())
    adapter._client = type("c", (), {"chat": type("ch", (), {"completions": completions})()})()

    await adapter.complete("the question")

    assert completions.requests[0]["temperature"] == 0.7
    assert completions.requests[0]["max_tokens"] == 256
    assert completions.requests[0]["model"] == "test-model"


async def test_the_answer_is_read_back() -> None:
    adapter, _ = openai_with(Response(content="forty-one pounds"))

    answer = await adapter.complete("the question")

    assert answer.text == "forty-one pounds"


async def test_token_usage_is_carried() -> None:
    """The cost metric multiplies these; a dropped count reads as a free query."""
    adapter, _ = openai_with(Response(prompt_tokens=386, completion_tokens=51))

    answer = await adapter.complete("the question")

    assert answer.prompt_tokens == 386
    assert answer.completion_tokens == 51


async def test_absent_usage_does_not_fail_the_call() -> None:
    """Some gateways omit usage entirely; an answer without a receipt is still an answer."""
    adapter, _ = openai_with(Response())

    answer = await adapter.complete("the question")

    assert answer.prompt_tokens == 0


async def test_a_null_content_becomes_an_empty_answer() -> None:
    """The SDK types content as optional; None would crash every downstream consumer."""
    adapter, _ = openai_with(Response(content=None))

    assert (await adapter.complete("the question")).text == ""


async def test_a_truncated_answer_says_so() -> None:
    """Silently returning a cut-off answer as complete is the worst of both."""
    adapter, _ = openai_with(Response())
    adapter._client.chat.completions.result.choices[0].finish_reason = "length"

    assert (await adapter.complete("the question")).truncated is True


async def test_a_vendor_error_becomes_a_typed_error() -> None:
    """A vendor exception escaping means the degradation ladder cannot act on it."""
    adapter, _ = openai_with(error=RuntimeError("the socket died"))

    with pytest.raises(GenerationError):
        await adapter.complete("the question")


async def test_streaming_asks_for_a_stream() -> None:
    class Events:
        def __aiter__(self) -> AsyncIterator[Any]:
            async def gen() -> AsyncIterator[Any]:
                yield Event("hello")

            return gen()

    adapter, completions = openai_with(Events())

    async for _ in adapter.stream("the question"):
        pass

    assert completions.requests[0]["stream"] is True


async def test_streaming_asks_for_usage() -> None:
    """Without it a streamed answer reports zero tokens and costs nothing on the dashboard."""

    class Events:
        def __aiter__(self) -> AsyncIterator[Any]:
            async def gen() -> AsyncIterator[Any]:
                yield Event("hello")

            return gen()

    adapter, completions = openai_with(Events())

    async for _ in adapter.stream("the question"):
        pass

    assert completions.requests[0]["stream_options"] == {"include_usage": True}


async def test_streamed_text_arrives_in_order() -> None:
    class Events:
        def __aiter__(self) -> AsyncIterator[Any]:
            async def gen() -> AsyncIterator[Any]:
                for piece in ("forty", "-one", " pounds"):
                    yield Event(piece)

            return gen()

    adapter, _ = openai_with(Events())

    assert "".join([piece async for piece in adapter.stream("q")]) == "forty-one pounds"


async def test_an_empty_event_is_skipped_rather_than_yielded() -> None:
    """The final usage-only event carries no choices; yielding it would emit a blank token."""

    class Events:
        def __aiter__(self) -> AsyncIterator[Any]:
            async def gen() -> AsyncIterator[Any]:
                yield Event("text")
                yield Event(None, empty=True)
                yield Event(None)

            return gen()

    adapter, _ = openai_with(Events())

    assert [piece async for piece in adapter.stream("q")] == ["text"]


async def test_a_mid_stream_failure_becomes_a_typed_error() -> None:
    """The SSE contract has to emit an error event and close, not leak a vendor exception."""

    class Events:
        def __aiter__(self) -> AsyncIterator[Any]:
            async def gen() -> AsyncIterator[Any]:
                yield Event("partial")
                raise RuntimeError("the connection dropped")

            return gen()

    adapter, _ = openai_with(Events())

    with pytest.raises(GenerationError):
        async for _ in adapter.stream("q"):
            pass


async def test_health_reports_unreachable_rather_than_raising() -> None:
    """A health check that raises is useless exactly when the provider is down."""
    adapter, _ = openai_with(error=RuntimeError("no route to host"))

    status = await adapter.health()

    assert status.healthy is False


async def test_health_reports_reachable_on_a_successful_ping() -> None:
    adapter, _ = openai_with(Response())

    assert (await adapter.health()).healthy is True


def test_a_completion_at_the_token_ceiling_is_truncated() -> None:
    assert Completion(text="x", model="m", finish_reason="max_tokens").truncated is True
    assert Completion(text="x", model="m", finish_reason="stop").truncated is False


@pytest.mark.parametrize(
    ("generator", "provider"),
    [(AnthropicGenerator, "anthropic"), (OllamaGenerator, "ollama")],
)
def test_every_generator_declares_its_provider(generator: type, provider: str) -> None:
    """The provider name labels metrics and the circuit breaker; a wrong one mislabels both."""
    assert generator(settings(provider)).provider == provider
