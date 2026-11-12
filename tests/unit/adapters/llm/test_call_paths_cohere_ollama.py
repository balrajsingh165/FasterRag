"""What the Cohere and Ollama generators send and how they read the answer back.

TASK-0441. ``test_call_paths.py`` exercises ``complete`` and ``stream`` for OpenAI only, so
the Cohere and Ollama call paths ran in no test at all — which is how one of them came to
raise ``TypeError`` on every call and the other to return empty text. This is the same set
of behaviours, run against both.

Clients are faked so the suite runs with neither library installed, but the fakes are not
free-hand: every reply is ``parsed`` from a recorded wire body in
``tests/unit/adapters/recordings.py``, and ``test_real_clients.py`` fails if a fake carries
anything the real client's object does not. Each fake also mirrors how its client is called:
Cohere has a separate ``chat_stream`` that is iterated directly, Ollama has one ``chat``
that is awaited and, when streaming, hands back the iterator.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

import pytest

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.adapters.llm.cohere import CohereGenerator
from fasterrag.adapters.llm.ollama import OllamaGenerator
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, GenerationError
from tests.unit.adapters.recordings import (
    COHERE_CHAT,
    COHERE_CHAT_STREAM,
    OLLAMA_CHAT,
    OLLAMA_CHAT_STREAM,
    parsed,
)

PROVIDERS = ["cohere", "ollama"]


class FakeClient:
    """What both fakes share: a primed reply, primed stream parts, and a record of calls."""

    def __init__(
        self,
        reply: dict[str, Any],
        parts: list[dict[str, Any]],
        *,
        error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.parts = parts
        self.error = error
        self.stream_error = stream_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def request(self) -> dict[str, Any]:
        """Return the keyword arguments of the only call made."""
        assert len(self.calls) == 1
        return self.calls[0][1]

    async def _events(self) -> AsyncIterator[Any]:
        for part in self.parts:
            yield parsed(part)
        if self.stream_error is not None:
            raise self.stream_error


class FakeCohereV2(FakeClient):
    """Stands in for ``cohere.AsyncClientV2``."""

    async def chat(self, **request: Any) -> Any:
        self.calls.append(("chat", request))
        if self.error is not None:
            raise self.error
        return parsed(self.reply)

    async def chat_stream(self, **request: Any) -> AsyncIterator[Any]:
        self.calls.append(("chat_stream", request))
        if self.error is not None:
            raise self.error
        async for event in self._events():
            yield event


class FakeOllama(FakeClient):
    """Stands in for ``ollama.AsyncClient``."""

    async def chat(self, **request: Any) -> Any:
        self.calls.append(("chat", request))
        if self.error is not None:
            raise self.error
        return self._events() if request["stream"] else parsed(self.reply)


def wire(
    provider: str,
    *,
    reply: dict[str, Any] | None = None,
    parts: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
    stream_error: Exception | None = None,
    **llm: Any,
) -> tuple[LLMAdapter, FakeClient]:
    """Return the provider's adapter wired to its fake client, and the fake."""
    key_env = None if provider == "ollama" else "TEST_LLM_KEY"
    payload = {"provider": provider, "model": "test-model", "api_key_env": key_env, **llm}
    settings = Settings.model_validate({"llm": payload})
    adapter: LLMAdapter
    client: FakeClient
    if provider == "cohere":
        adapter = CohereGenerator(settings)
        client = FakeCohereV2(
            reply or COHERE_CHAT,
            COHERE_CHAT_STREAM if parts is None else parts,
            error=error,
            stream_error=stream_error,
        )
    else:
        adapter = OllamaGenerator(settings)
        client = FakeOllama(
            reply or OLLAMA_CHAT,
            OLLAMA_CHAT_STREAM if parts is None else parts,
            error=error,
            stream_error=stream_error,
        )
    adapter._client = client
    return adapter, client


def recorded_reply(provider: str) -> dict[str, Any]:
    """Return a private copy of the provider's recorded reply, safe to alter."""
    return copy.deepcopy(COHERE_CHAT if provider == "cohere" else OLLAMA_CHAT)


def sampling(provider: str, request: dict[str, Any]) -> tuple[float, int]:
    """Return the temperature and token limit a request carries, wherever it puts them."""
    if provider == "cohere":
        return request["temperature"], request["max_tokens"]
    return request["options"]["temperature"], request["options"]["num_predict"]


async def collect(stream: AsyncIterator[str], into: list[str]) -> None:
    """Append every piece of ``stream`` to ``into``, so a failure keeps what arrived first."""
    async for piece in stream:
        into.append(piece)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_the_prompt_is_sent_as_a_user_message(provider: str) -> None:
    adapter, client = wire(provider)

    await adapter.complete("what is the allowance")

    assert client.request["messages"] == [{"role": "user", "content": "what is the allowance"}]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_system_prompt_leads_the_messages(provider: str) -> None:
    """A system message after the user message is ignored by the model, silently."""
    adapter, client = wire(provider)

    await adapter.complete("the question", system="be terse")

    assert client.request["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "the question"},
    ]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_the_configured_sampling_reaches_the_request(provider: str) -> None:
    """Settings the request omits are settings that silently do nothing."""
    adapter, client = wire(provider, temperature=0.7, max_tokens=256)

    await adapter.complete("the question")

    assert sampling(provider, client.request) == (0.7, 256)
    assert client.request["model"] == "test-model"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_the_answer_is_read_back(provider: str) -> None:
    """Cohere raised TypeError here and Ollama returned empty text; neither was tested."""
    adapter, _ = wire(provider)

    answer = await adapter.complete("the question")

    assert answer.text == "forty-one pounds"
    assert answer.model == "test-model"
    assert answer.truncated is False
    assert adapter.provider == provider


@pytest.mark.parametrize(("provider", "expected"), [("cohere", (5, 3)), ("ollama", (26, 298))])
async def test_token_usage_is_carried(provider: str, expected: tuple[int, int]) -> None:
    """The cost metric multiplies these; a dropped count reads as a free query."""
    adapter, _ = wire(provider)

    answer = await adapter.complete("the question")

    assert (answer.prompt_tokens, answer.completion_tokens) == expected


async def test_cohere_falls_back_to_consumed_tokens_when_no_bill_is_stated() -> None:
    """A reply with no billed units still used tokens; zero would price it as free."""
    reply = recorded_reply("cohere")
    del reply["usage"]["billed_units"]
    adapter, _ = wire("cohere", reply=reply)

    answer = await adapter.complete("the question")

    assert (answer.prompt_tokens, answer.completion_tokens) == (71, 3)


@pytest.mark.parametrize(
    ("provider", "fields"),
    [("cohere", ["usage"]), ("ollama", ["prompt_eval_count", "eval_count"])],
)
async def test_absent_usage_does_not_fail_the_call(provider: str, fields: list[str]) -> None:
    """An answer without a receipt is still an answer."""
    reply = recorded_reply(provider)
    for field in fields:
        del reply[field]
    adapter, _ = wire(provider, reply=reply)

    answer = await adapter.complete("the question")

    assert answer.text == "forty-one pounds"
    assert (answer.prompt_tokens, answer.completion_tokens) == (0, 0)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_reply_without_content_becomes_an_empty_answer(provider: str) -> None:
    """None would crash every downstream consumer of the text."""
    reply = recorded_reply(provider)
    reply["message"]["content"] = [] if provider == "cohere" else None
    adapter, _ = wire(provider, reply=reply)

    assert (await adapter.complete("the question")).text == ""


async def test_cohere_joins_text_blocks_and_ignores_the_rest() -> None:
    """A reasoning model interleaves thinking blocks, which carry no ``text`` at all."""
    reply = recorded_reply("cohere")
    reply["message"]["content"] = [
        {"type": "thinking", "thinking": "the allowance table says"},
        {"type": "text", "text": "forty-one"},
        {"type": "text", "text": " pounds"},
    ]
    adapter, _ = wire("cohere", reply=reply)

    assert (await adapter.complete("the question")).text == "forty-one pounds"


@pytest.mark.parametrize(
    ("provider", "field", "reason"),
    [("cohere", "finish_reason", "MAX_TOKENS"), ("ollama", "done_reason", "length")],
)
async def test_a_truncated_answer_says_so(provider: str, field: str, reason: str) -> None:
    """Silently returning a cut-off answer as complete is the worst of both."""
    reply = recorded_reply(provider)
    reply[field] = reason
    adapter, _ = wire(provider, reply=reply)

    assert (await adapter.complete("the question")).truncated is True


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_vendor_error_becomes_a_typed_error(provider: str) -> None:
    """A vendor exception escaping means the degradation ladder cannot act on it."""
    adapter, _ = wire(provider, error=RuntimeError("the socket died"))

    with pytest.raises(GenerationError) as caught:
        await adapter.complete("the question")

    assert caught.value.retryable is True


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("mistake", [TypeError("unexpected keyword"), AttributeError("no get")])
async def test_a_client_the_adapter_cannot_drive_is_not_reported_as_an_outage(
    provider: str, mistake: Exception
) -> None:
    """A TypeError from the wrong client read as "unreachable, retryable" and was retried."""
    adapter, _ = wire(provider, error=mistake)

    with pytest.raises(GenerationError) as caught:
        await adapter.complete("the question")

    assert caught.value.code is ErrorCode.INTERNAL
    assert caught.value.retryable is False
    assert "unreachable" not in caught.value.detail


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_streaming_asks_for_a_stream(provider: str) -> None:
    adapter, client = wire(provider)

    async for _ in adapter.stream("the question", system="be terse"):
        pass

    method, request = client.calls[0]
    assert request["messages"][0] == {"role": "system", "content": "be terse"}
    if provider == "cohere":
        assert method == "chat_stream"
    else:
        assert request["stream"] is True


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_streamed_text_arrives_in_order(provider: str) -> None:
    adapter, _ = wire(provider)

    assert [piece async for piece in adapter.stream("q")] == ["forty", "-one pounds"]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_part_without_text_is_skipped_rather_than_yielded(provider: str) -> None:
    """The recorded streams open and close with parts that carry no text; none is a token."""
    adapter, client = wire(provider)

    pieces = [piece async for piece in adapter.stream("q")]

    assert len(client.parts) > len(pieces)
    assert all(pieces)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_mid_stream_failure_becomes_a_typed_error(provider: str) -> None:
    """The SSE contract has to emit an error event and close, not leak a vendor exception."""
    adapter, _ = wire(provider, stream_error=RuntimeError("the connection dropped"))
    received: list[str] = []

    with pytest.raises(GenerationError):
        await collect(adapter.stream("q"), received)

    assert received == ["forty", "-one pounds"]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_stream_that_cannot_start_becomes_a_typed_error(provider: str) -> None:
    adapter, _ = wire(provider, error=RuntimeError("no route to host"))

    with pytest.raises(GenerationError):
        async for _ in adapter.stream("q"):
            pass


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_health_reports_unreachable_rather_than_raising(provider: str) -> None:
    """A health check that raises is useless exactly when the provider is down."""
    adapter, _ = wire(provider, error=RuntimeError("no route to host"))

    assert (await adapter.health()).healthy is False


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_health_reports_reachable_on_a_successful_ping(provider: str) -> None:
    adapter, _ = wire(provider)

    assert (await adapter.health()).healthy is True
