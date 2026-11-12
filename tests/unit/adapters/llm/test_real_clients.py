"""The Cohere and Ollama generators, driven through the real client libraries.

TASK-0441. ``test_call_paths.py`` wires each adapter to a fake client, which proves the
adapter and the fake agree and nothing about the library. Here the adapter builds its own
client — the genuine ``cohere`` or ``ollama`` one — and a recorded reply is fed back at the
transport (``tests/unit/adapters/recordings.py``), so request building, the library's own
parsing and the adapter's reading of the parsed object all run. No socket is opened and no
credential is needed.

Each test skips, naming the extra to install, when its library is absent. Both libraries are
in the ``dev`` extra precisely so that these do not skip in CI. One test per library carries
``must_run``: every test for a library skips on the same condition, so one that ran proves
the library was installed in that lane, and the build fails if it ran in none.

The last two tests of each group are the fidelity pair: they hold the fakes that the
call-path suite uses against what the real client parsed from the same body.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from fasterrag.adapters.llm.cohere import CohereGenerator
from fasterrag.adapters.llm.ollama import OllamaGenerator
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, GenerationError
from tests.unit.adapters.recordings import (
    COHERE_CHAT,
    COHERE_CHAT_STREAM,
    COHERE_UNAUTHORIZED,
    OLLAMA_CHAT,
    OLLAMA_CHAT_STREAM,
    OLLAMA_MODEL_NOT_FOUND,
    Replay,
    assert_same_shape,
    json_reply,
    ndjson_reply,
    parsed,
    refusing_connections,
    require_library,
    sse_reply,
)

KEY_ENV = "TEST_LLM_KEY"
MESSAGES = [{"role": "user", "content": "the question"}]


def cohere_settings(**llm: Any) -> Settings:
    payload: dict[str, Any] = {"provider": "cohere", "model": "test-model", "api_key_env": KEY_ENV}
    return Settings.model_validate({"llm": {**payload, **llm}})


def ollama_settings(**llm: Any) -> Settings:
    payload: dict[str, Any] = {"provider": "ollama", "model": "test-model", "api_key_env": None}
    return Settings.model_validate({"llm": {**payload, **llm}})


@pytest.fixture
def cohere_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the library and supply a placeholder credential; nothing is sent anywhere."""
    require_library("cohere")
    monkeypatch.setenv(KEY_ENV, "recorded-not-a-key")


@pytest.fixture
def ollama_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the library and clear the one variable that would redirect the client."""
    require_library("ollama")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.mark.must_run
@pytest.mark.usefixtures("cohere_key")
async def test_cohere_generates_through_the_v2_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v1 client was built, and it refuses ``messages`` with a TypeError before sending."""
    from cohere import AsyncClientV2

    replay = Replay(monkeypatch, {"/v2/chat": json_reply(COHERE_CHAT)})
    adapter = CohereGenerator(cohere_settings())

    await adapter.complete("the question")

    assert isinstance(adapter._client, AsyncClientV2)
    assert replay.paths == ["/v2/chat"]


@pytest.mark.usefixtures("cohere_key")
async def test_cohere_reads_a_recorded_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    Replay(monkeypatch, {"/v2/chat": json_reply(COHERE_CHAT)})

    answer = await CohereGenerator(cohere_settings()).complete("the question")

    assert answer.text == "forty-one pounds"
    assert answer.model == "test-model"
    assert answer.truncated is False
    assert (answer.prompt_tokens, answer.completion_tokens) == (5, 3)
    assert type(answer.prompt_tokens) is int
    assert type(answer.completion_tokens) is int


@pytest.mark.usefixtures("cohere_key")
async def test_cohere_sends_the_request_the_v2_api_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = Replay(monkeypatch, {"/v2/chat": json_reply(COHERE_CHAT)})
    adapter = CohereGenerator(cohere_settings(temperature=0.7, max_tokens=256))

    await adapter.complete("the question", system="be terse")

    assert replay.bodies[0] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "the question"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    }


@pytest.mark.usefixtures("cohere_key")
async def test_a_cohere_reply_cut_at_the_token_limit_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cohere says ``MAX_TOKENS`` in upper case, which matched nothing ``truncated`` knows."""
    cut_off = {**COHERE_CHAT, "finish_reason": "MAX_TOKENS"}
    Replay(monkeypatch, {"/v2/chat": json_reply(cut_off)})

    answer = await CohereGenerator(cohere_settings()).complete("the question")

    assert answer.finish_reason == "max_tokens"
    assert answer.truncated is True


@pytest.mark.usefixtures("cohere_key")
async def test_cohere_streams_a_recorded_reply_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = Replay(monkeypatch, {"/v2/chat": sse_reply(COHERE_CHAT_STREAM)})
    adapter = CohereGenerator(cohere_settings())

    pieces = [piece async for piece in adapter.stream("the question")]

    assert pieces == ["forty", "-one pounds"]
    assert replay.bodies[0]["stream"] is True


@pytest.mark.usefixtures("cohere_key")
async def test_a_rejected_cohere_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    Replay(monkeypatch, {"/v2/chat": json_reply(COHERE_UNAUTHORIZED, status=401)})

    with pytest.raises(GenerationError) as caught:
        await CohereGenerator(cohere_settings()).complete("the question")

    assert caught.value.retryable is False
    assert KEY_ENV in caught.value.detail
    assert "recorded-not-a-key" not in caught.value.detail


@pytest.mark.usefixtures("cohere_key")
async def test_the_v1_cohere_client_still_refuses_a_message_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins why the adapter names the v2 client: the day this passes a list, say so here."""
    from cohere import AsyncClient

    replay = Replay(monkeypatch, {"/v2/chat": json_reply(COHERE_CHAT)})
    adapter = CohereGenerator(cohere_settings())
    adapter._client = AsyncClient(api_key="recorded-not-a-key")

    with pytest.raises(GenerationError) as caught:
        await adapter.complete("the question")

    assert caught.value.code is ErrorCode.INTERNAL
    assert caught.value.retryable is False
    assert replay.requests == []


@pytest.mark.usefixtures("cohere_key")
async def test_the_fake_cohere_reply_has_the_shape_the_real_client_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/v2/chat": json_reply(COHERE_CHAT)})
    client = CohereGenerator(cohere_settings())._connected()

    real = await client.chat(model="test-model", messages=copy.deepcopy(MESSAGES))

    assert_same_shape(real, parsed(COHERE_CHAT))


@pytest.mark.usefixtures("cohere_key")
async def test_the_fake_cohere_events_have_the_shape_the_real_client_yields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/v2/chat": sse_reply(COHERE_CHAT_STREAM)})
    client = CohereGenerator(cohere_settings())._connected()

    stream = client.chat_stream(model="test-model", messages=copy.deepcopy(MESSAGES))
    real = [event async for event in stream]

    assert_same_shape(real, parsed(COHERE_CHAT_STREAM))


@pytest.mark.must_run
@pytest.mark.usefixtures("ollama_library")
async def test_ollama_reads_a_recorded_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The typed reply is not a dict, so the text came back empty beside correct counts."""
    Replay(monkeypatch, {"/api/chat": json_reply(OLLAMA_CHAT)})

    answer = await OllamaGenerator(ollama_settings()).complete("the question")

    assert answer.text == "forty-one pounds"
    assert (answer.prompt_tokens, answer.completion_tokens) == (26, 298)
    assert answer.finish_reason == "stop"
    assert answer.truncated is False


@pytest.mark.usefixtures("ollama_library")
async def test_an_ollama_reply_cut_at_the_token_limit_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/api/chat": json_reply({**OLLAMA_CHAT, "done_reason": "length"})})

    answer = await OllamaGenerator(ollama_settings()).complete("the question")

    assert answer.truncated is True


@pytest.mark.usefixtures("ollama_library")
async def test_ollama_sends_the_request_the_chat_api_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = Replay(monkeypatch, {"/api/chat": json_reply(OLLAMA_CHAT)})
    server = "http://ollama.internal:11434"
    adapter = OllamaGenerator(ollama_settings(temperature=0.7, max_tokens=256, base_url=server))

    await adapter.complete("the question", system="be terse")

    sent = replay.bodies[0]
    assert replay.requests[0][0].host == "ollama.internal"
    assert sent["model"] == "test-model"
    assert sent["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "the question"},
    ]
    assert sent["stream"] is False
    assert sent["options"] == {"temperature": 0.7, "num_predict": 256}


@pytest.mark.usefixtures("ollama_library")
async def test_ollama_streams_a_recorded_reply_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every part is a typed object, so the stream yielded nothing at all."""
    replay = Replay(monkeypatch, {"/api/chat": ndjson_reply(OLLAMA_CHAT_STREAM)})
    adapter = OllamaGenerator(ollama_settings())

    pieces = [piece async for piece in adapter.stream("the question")]

    assert pieces == ["forty", "-one pounds"]
    assert replay.bodies[0]["stream"] is True


@pytest.mark.usefixtures("ollama_library")
async def test_a_model_ollama_does_not_have_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    Replay(monkeypatch, {"/api/chat": json_reply(OLLAMA_MODEL_NOT_FOUND, status=404)})

    with pytest.raises(GenerationError) as caught:
        await OllamaGenerator(ollama_settings()).complete("the question")

    assert caught.value.retryable is False
    assert "404" in caught.value.detail


@pytest.mark.usefixtures("ollama_library")
async def test_an_ollama_server_that_is_not_running_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/api/chat": refusing_connections})

    with pytest.raises(GenerationError) as caught:
        await OllamaGenerator(ollama_settings()).complete("the question")

    assert caught.value.retryable is True
    assert "unreachable" in caught.value.detail


@pytest.mark.usefixtures("ollama_library")
async def test_the_fake_ollama_reply_has_the_shape_the_real_client_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/api/chat": json_reply(OLLAMA_CHAT)})
    client = OllamaGenerator(ollama_settings())._connected()

    real = await client.chat(model="test-model", messages=copy.deepcopy(MESSAGES))

    assert_same_shape(real, parsed(OLLAMA_CHAT))


@pytest.mark.usefixtures("ollama_library")
async def test_the_fake_ollama_parts_have_the_shape_the_real_client_yields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/api/chat": ndjson_reply(OLLAMA_CHAT_STREAM)})
    client = OllamaGenerator(ollama_settings())._connected()

    stream = await client.chat(model="test-model", messages=copy.deepcopy(MESSAGES), stream=True)
    real = [part async for part in stream]

    assert_same_shape(real, parsed(OLLAMA_CHAT_STREAM))
