"""The Ollama and Cohere embedders, driven through the real client libraries.

TASK-0441. The embedding twin of ``tests/unit/adapters/llm/test_real_clients.py``: the
adapter builds its own genuine client and a recorded reply is fed back at the transport
(``tests/unit/adapters/recordings.py``). No socket is opened and no credential is needed.

The Ollama half is where the defect was — every embedding failed, because the client
returns a typed ``EmbedResponse`` and the adapter looked for a ``dict``. The Cohere half
found nothing wrong and stays as the proof of that: the v1 ``embed`` call the adapter makes
is accepted by the library, and the vectors do arrive on ``embeddings.float_``.

Each test skips, naming the extra to install, when its library is absent. One test per
library carries ``must_run``, which fails the build if it skipped in every CI lane.
"""

from __future__ import annotations

from typing import Any

import pytest

from fasterrag.adapters.embeddings.cohere import CohereEmbedder
from fasterrag.adapters.embeddings.ollama import OllamaEmbedder
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError
from tests.unit.adapters.recordings import (
    COHERE_EMBED,
    OLLAMA_EMBED,
    OLLAMA_MODEL_NOT_FOUND,
    OTHER_VECTOR,
    VECTOR,
    Replay,
    assert_same_shape,
    json_reply,
    parsed,
    refusing_connections,
    require_library,
)

KEY_ENV = "TEST_EMBEDDING_KEY"


def ollama_settings(llm: dict[str, Any] | None = None, **embeddings: Any) -> Settings:
    payload: dict[str, Any] = {
        "embeddings": {"provider": "ollama", "model": "test-model", **embeddings}
    }
    if llm is not None:
        payload["llm"] = llm
    return Settings.model_validate(payload)


def cohere_settings(**embeddings: Any) -> Settings:
    payload = {"provider": "cohere", "model": "test-model", "api_key_env": KEY_ENV, **embeddings}
    return Settings.model_validate({"embeddings": payload})


@pytest.fixture
def ollama_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the library and clear the one variable that would redirect the client."""
    require_library("ollama")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.fixture
def cohere_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the library and supply a placeholder credential; nothing is sent anywhere."""
    require_library("cohere")
    monkeypatch.setenv(KEY_ENV, "recorded-not-a-key")


@pytest.mark.must_run
@pytest.mark.usefixtures("ollama_library")
async def test_ollama_embeds_a_recorded_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The typed reply is not a dict, so every text failed as "an empty embedding"."""
    replay = Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_EMBED)})

    result = await OllamaEmbedder(ollama_settings()).embed_documents(["a", "b"])

    assert result.vectors == [VECTOR, OTHER_VECTOR]
    assert result.model == "test-model"
    assert result.total_tokens == 8
    assert replay.paths == ["/api/embed"]
    assert replay.bodies[0]["model"] == "test-model"
    assert replay.bodies[0]["input"] == ["a", "b"]


@pytest.mark.usefixtures("ollama_library")
async def test_an_ollama_reply_without_a_count_is_no_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uncounted = {key: value for key, value in OLLAMA_EMBED.items() if key != "prompt_eval_count"}
    Replay(monkeypatch, {"/api/embed": json_reply(uncounted)})

    result = await OllamaEmbedder(ollama_settings()).embed_documents(["a", "b"])

    assert result.total_tokens is None


@pytest.mark.usefixtures("ollama_library")
async def test_an_ollama_query_embeds_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = Replay(
        monkeypatch, {"/api/embed": json_reply({**OLLAMA_EMBED, "embeddings": [VECTOR]})}
    )

    vector = await OllamaEmbedder(ollama_settings()).embed_query("a question")

    assert vector == VECTOR
    assert replay.bodies[0]["input"] == ["a question"]


@pytest.mark.usefixtures("ollama_library")
async def test_ollama_refuses_a_reply_with_too_few_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zipping two vectors onto three chunks would index a passage under another's vector."""
    Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_EMBED)})

    with pytest.raises(EmbedError, match="2 usable embeddings for 3 texts") as caught:
        await OllamaEmbedder(ollama_settings()).embed_documents(["a", "b", "c"])

    assert caught.value.retryable is False


@pytest.mark.usefixtures("ollama_library")
async def test_the_embedder_shares_the_server_of_an_ollama_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_EMBED)})
    llm = {"provider": "ollama", "api_key_env": None, "base_url": "http://ollama.internal:11434"}

    await OllamaEmbedder(ollama_settings(llm)).embed_documents(["a", "b"])

    assert replay.requests[0][0].host == "ollama.internal"


@pytest.mark.usefixtures("ollama_library")
async def test_the_embedder_does_not_borrow_another_providers_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``llm.base_url`` was read whatever the LLM was, sending embeddings to its gateway."""
    replay = Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_EMBED)})
    llm = {
        "provider": "openai_compatible",
        "api_key_env": "GATEWAY_KEY",
        "base_url": "http://gateway.internal:8000/v1",
    }

    await OllamaEmbedder(ollama_settings(llm)).embed_documents(["a", "b"])

    url = replay.requests[0][0]
    assert (url.host, url.port, url.path) == ("127.0.0.1", 11434, "/api/embed")


@pytest.mark.usefixtures("ollama_library")
async def test_the_embedder_follows_ollama_host_when_the_llm_is_not_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_EMBED)})
    monkeypatch.setenv("OLLAMA_HOST", "http://embeddings.internal:11434")

    await OllamaEmbedder(ollama_settings()).embed_documents(["a", "b"])

    assert replay.requests[0][0].host == "embeddings.internal"


@pytest.mark.usefixtures("ollama_library")
async def test_a_model_ollama_does_not_have_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_MODEL_NOT_FOUND, status=404)})

    with pytest.raises(EmbedError) as caught:
        await OllamaEmbedder(ollama_settings()).embed_documents(["a"])

    assert caught.value.retryable is False
    assert "404" in caught.value.detail


@pytest.mark.usefixtures("ollama_library")
async def test_an_ollama_server_that_is_not_running_is_unhealthy_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/api/embed": refusing_connections})

    status = await OllamaEmbedder(ollama_settings()).health()

    assert status.healthy is False


@pytest.mark.usefixtures("ollama_library")
async def test_the_fake_ollama_embedding_has_the_shape_the_real_client_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/api/embed": json_reply(OLLAMA_EMBED)})
    client = OllamaEmbedder(ollama_settings())._connected()

    real = await client.embed(model="test-model", input=["a", "b"])

    assert_same_shape(real, parsed(OLLAMA_EMBED))


@pytest.mark.must_run
@pytest.mark.usefixtures("cohere_key")
async def test_cohere_embeds_a_recorded_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = Replay(monkeypatch, {"/v1/embed": json_reply(COHERE_EMBED)})

    result = await CohereEmbedder(cohere_settings()).embed_documents(["a", "b"])

    assert result.vectors == [VECTOR, OTHER_VECTOR]
    assert replay.paths == ["/v1/embed"]
    assert replay.bodies[0] == {
        "texts": ["a", "b"],
        "model": "test-model",
        "input_type": "search_document",
        "embedding_types": ["float"],
    }
    assert result.total_tokens == 2
    assert type(result.total_tokens) is int


@pytest.mark.usefixtures("cohere_key")
async def test_a_cohere_reply_that_states_no_bill_is_no_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client invents a ``meta`` whose billed units are all ``None``; that is not zero."""
    unbilled = {key: value for key, value in COHERE_EMBED.items() if key != "meta"}
    Replay(monkeypatch, {"/v1/embed": json_reply(unbilled)})

    result = await CohereEmbedder(cohere_settings()).embed_documents(["a", "b"])

    assert result.total_tokens is None
    assert len(result.vectors) == 2


@pytest.mark.usefixtures("cohere_key")
async def test_the_fake_cohere_embedding_has_the_shape_the_real_client_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Replay(monkeypatch, {"/v1/embed": json_reply(COHERE_EMBED)})
    client = CohereEmbedder(cohere_settings())._connected()

    real = await client.embed(
        texts=["a", "b"],
        model="test-model",
        input_type="search_document",
        embedding_types=["float"],
    )

    assert_same_shape(real, parsed(COHERE_EMBED))
