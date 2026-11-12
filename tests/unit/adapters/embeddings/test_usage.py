"""Billed-usage extraction, proved without contacting a provider.

Every response below is built to the shape the vendor's client hands back, because the point
of the test is that the adapter reads the right field — a fake shaped like the adapter
already behaves would pass whatever the adapter did. Provenance for each:

* **Cohere** — built from ``COHERE_EMBED`` in ``tests/unit/adapters/recordings.py``, the
  ``/v1/embed`` body, parsed the way ``cohere.AsyncClient.embed`` parses it: vectors on
  ``embeddings.float_`` and the billable text units on ``meta.billed_units.input_tokens``,
  as a float. The reference makes every level of ``meta`` optional, which is why the
  absent-meta case is exercised too, and allows a separate ``meta.tokens``, which one test
  adds as a decoy.
* **OpenAI** — the embeddings response carries ``data[].embedding`` and ``usage``, whose
  ``total_tokens`` is the billed figure. The same field the adapter already read.
* **Ollama** — built from ``OLLAMA_EMBED``, the ``/api/embed`` body, parsed the way
  ``ollama.AsyncClient.embed`` parses it: an object, not a ``dict``, holding one vector per
  input and the ``prompt_eval_count`` the server counted for the batch.

The Cohere and Ollama fakes are held to the real libraries by the fidelity tests in
``test_real_clients.py`` beside this file (TASK-0441): each parses the same recorded body
through the installed client and fails if the fake carries anything the real object does
not. The live probes in ``tests/eval/test_live_providers.py`` are what check the recording
itself against the vendor, and they are not run here.

The last two tests are the anti-vacuity pair. The live probes only run with a credential, so
a ``reports_token_usage`` flag left behind by an adapter change would otherwise be caught by
nothing; these run in the job that cannot skip.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, sum_reported
from fasterrag.adapters.embeddings.cohere import CohereEmbedder
from fasterrag.adapters.embeddings.ollama import OllamaEmbedder
from fasterrag.adapters.embeddings.openai import OpenAIEmbedder
from fasterrag.config.schema import Settings
from tests.eval.test_live_providers import EMBEDDING_CHECKS, LiveCheck
from tests.unit.adapters.recordings import COHERE_EMBED, OLLAMA_EMBED, VECTOR, parsed


def settings_for(provider: str, **embeddings: Any) -> Settings:
    """Return settings pointing embedding at ``provider``."""
    payload: dict[str, Any] = {"provider": provider, "model": "test-model", **embeddings}
    if provider != "ollama":
        payload.setdefault("api_key_env", f"{provider.upper()}_API_KEY")
    return Settings.model_validate({"embeddings": payload})


def _obj(**fields: Any) -> Any:
    """Return a plain attribute bag, as a parsed SDK model presents itself."""
    return type("response", (), fields)()


def cohere_response(
    count: int, *, input_tokens: int | None, meta: bool = True, counted: int | None = None
) -> Any:
    """Return what the Cohere client hands back for ``count`` texts and their billed units."""
    vectors = [list(VECTOR) for _ in range(count)]
    reply = parsed({**COHERE_EMBED, "texts": ["a"] * count, "embeddings": {"float": vectors}})
    if not meta:
        reply.meta = None
        return reply
    billed = None if input_tokens is None else float(input_tokens)
    reply.meta.billed_units = SimpleNamespace(input_tokens=billed)
    reply.meta.tokens = None if counted is None else SimpleNamespace(input_tokens=counted)
    return reply


def openai_response(count: int, *, total_tokens: int | None) -> Any:
    """Return an OpenAI embeddings response, optionally without a usage block."""
    return _obj(
        data=[_obj(embedding=list(VECTOR)) for _ in range(count)],
        usage=None if total_tokens is None else _obj(total_tokens=total_tokens),
    )


class FakeCohereClient:
    """Returns primed Cohere embed responses, one per call."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def embed(self, **request: Any) -> Any:
        self.calls.append(request)
        return self.responses.pop(0)


class FakeOpenAIClient:
    """Returns primed OpenAI embeddings responses, one per call."""

    def __init__(self, *responses: Any) -> None:
        self.embeddings = self
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.calls.append(request)
        return self.responses.pop(0)


class FakeOllamaClient:
    """Answers ``embed`` as the Ollama client does: one typed reply for the whole batch."""

    def __init__(self, *, counted: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.counted = counted

    async def embed(self, **request: Any) -> Any:
        self.calls.append(request)
        texts = request["input"]
        body = {**OLLAMA_EMBED, "embeddings": [list(VECTOR) for _ in texts]}
        body["prompt_eval_count"] = 4 * len(texts)
        if not self.counted:
            del body["prompt_eval_count"]
        return parsed(body)


def cohere_with(*responses: Any, **overrides: Any) -> CohereEmbedder:
    """Return a Cohere adapter wired to a fake client."""
    adapter = CohereEmbedder(settings_for("cohere", **overrides))
    adapter._client = FakeCohereClient(*responses)
    return adapter


async def test_cohere_reports_the_units_it_says_it_billed() -> None:
    adapter = cohere_with(cohere_response(2, input_tokens=17))

    result = await adapter.embed_documents(["a", "b"])

    assert result.total_tokens == 17
    assert len(result.vectors) == 2


async def test_cohere_bills_from_billed_units_not_from_counted_tokens() -> None:
    """The invoice is charged on billed_units; meta.tokens is a different number."""
    adapter = cohere_with(cohere_response(1, input_tokens=17, counted=999))

    assert (await adapter.embed_documents(["a"])).total_tokens == 17


async def test_cohere_without_billed_units_reports_no_receipt() -> None:
    adapter = cohere_with(cohere_response(1, input_tokens=None))

    assert (await adapter.embed_documents(["a"])).total_tokens is None


async def test_cohere_without_meta_reports_no_receipt() -> None:
    adapter = cohere_with(cohere_response(1, input_tokens=None, meta=False))

    assert (await adapter.embed_documents(["a"])).total_tokens is None


async def test_cohere_sums_the_units_of_every_batch() -> None:
    adapter = cohere_with(
        cohere_response(2, input_tokens=10),
        cohere_response(1, input_tokens=4),
        batch_size=2,
    )

    result = await adapter.embed_documents(["a", "b", "c"])

    assert result.total_tokens == 14
    assert len(result.vectors) == 3


async def test_one_unreported_batch_makes_the_whole_total_unreported() -> None:
    """A partial sum under-counts, and an under-count reconciles wrong against an invoice."""
    adapter = cohere_with(
        cohere_response(2, input_tokens=10),
        cohere_response(1, input_tokens=None),
        batch_size=2,
    )

    assert (await adapter.embed_documents(["a", "b", "c"])).total_tokens is None


async def test_a_cohere_query_still_embeds_as_a_search_query() -> None:
    adapter = CohereEmbedder(settings_for("cohere"))
    client = FakeCohereClient(cohere_response(1, input_tokens=3))
    adapter._client = client

    vector = await adapter.embed_query("a question")

    assert vector == VECTOR
    assert client.calls[0]["input_type"] == "search_query"


async def test_openai_reports_the_usage_block_it_is_given() -> None:
    adapter = OpenAIEmbedder(settings_for("openai"))
    adapter._client = FakeOpenAIClient(openai_response(1, total_tokens=12))

    assert (await adapter.embed_documents(["a"])).total_tokens == 12


async def test_openai_without_a_usage_block_reports_no_receipt() -> None:
    """A gateway that omits usage has not told us the call was free."""
    adapter = OpenAIEmbedder(settings_for("openai"))
    adapter._client = FakeOpenAIClient(openai_response(1, total_tokens=None))

    assert (await adapter.embed_documents(["a"])).total_tokens is None


def ollama_with(client: FakeOllamaClient, **overrides: Any) -> OllamaEmbedder:
    """Return an Ollama adapter wired to a fake client."""
    adapter = OllamaEmbedder(settings_for("ollama", **overrides))
    adapter._client = client
    return adapter


async def test_ollama_reads_the_typed_reply_the_client_returns() -> None:
    """The reply is an object, not a dict; reading it as one failed every embedding."""
    adapter = ollama_with(FakeOllamaClient())

    result = await adapter.embed_documents(["a", "b"])

    assert result.vectors == [VECTOR, VECTOR]
    assert result.total_tokens == 8


async def test_ollama_sends_a_whole_batch_in_one_request() -> None:
    """The superseded route took one text per request, so a batch cost a round trip each."""
    client = FakeOllamaClient()

    await ollama_with(client).embed_documents(["a", "b", "c"])

    assert client.calls == [{"model": "test-model", "input": ["a", "b", "c"]}]


async def test_ollama_sums_the_count_of_every_batch() -> None:
    client = FakeOllamaClient()

    result = await ollama_with(client, batch_size=2).embed_documents(["a", "b", "c"])

    assert [call["input"] for call in client.calls] == [["a", "b"], ["c"]]
    assert result.total_tokens == 12
    assert len(result.vectors) == 3


async def test_ollama_without_a_count_reports_no_receipt() -> None:
    adapter = ollama_with(FakeOllamaClient(counted=False))

    assert (await adapter.embed_documents(["a"])).total_tokens is None


async def test_an_ollama_query_embeds_as_a_batch_of_one() -> None:
    client = FakeOllamaClient()

    vector = await ollama_with(client).embed_query("a question")

    assert vector == VECTOR
    assert client.calls[0]["input"] == ["a question"]


@pytest.mark.parametrize(
    ("llm_provider", "key_env", "expected"),
    [("ollama", None, "http://llm.internal:11434"), ("openai_compatible", "GATEWAY_KEY", None)],
)
def test_the_ollama_embedder_borrows_only_an_ollama_llms_endpoint(
    llm_provider: str, key_env: str | None, expected: str | None
) -> None:
    """``llm.base_url`` was read whatever the LLM was, sending embeddings to its gateway."""
    llm = {
        "provider": llm_provider,
        "api_key_env": key_env,
        "base_url": "http://llm.internal:11434",
    }
    configured = Settings.model_validate(
        {"embeddings": {"provider": "ollama", "model": "test-model"}, "llm": llm}
    )

    assert OllamaEmbedder(configured)._host == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], 0), ([3], 3), ([3, 4], 7), ([3, None], None), ([None], None)],
)
def test_an_incomplete_set_of_receipts_is_no_receipt(
    values: list[int | None], expected: int | None
) -> None:
    assert sum_reported(values) == expected


def _adapter_for(check: LiveCheck) -> EmbeddingAdapter:
    """Return the adapter for ``check``, wired to a fake that does report usage."""
    if check.provider == "cohere":
        return cohere_with(cohere_response(1, input_tokens=5))
    if check.provider == "openai":
        adapter = OpenAIEmbedder(settings_for("openai"))
        adapter._client = FakeOpenAIClient(openai_response(1, total_tokens=5))
        return adapter
    if check.provider == "ollama":
        ollama = OllamaEmbedder(settings_for("ollama"))
        ollama._client = FakeOllamaClient()
        return ollama
    raise AssertionError(f"{check.name} has no recorded response in this module")


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda check: check.name)
async def test_the_live_flag_matches_what_the_adapter_actually_does(check: LiveCheck) -> None:
    """Pin ``reports_token_usage`` against the adapter, in a job that cannot skip.

    Every provider is fed a response that does carry the vendor's usage fields, so a
    ``False`` can only come from the adapter.
    """
    result = await _adapter_for(check).embed_documents(["a"])

    assert (result.total_tokens is not None) is check.reports_token_usage


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda check: check.name)
def test_a_provider_that_reports_nothing_has_to_say_why(check: LiveCheck) -> None:
    assert bool(check.usage_note.strip()) is not check.reports_token_usage
