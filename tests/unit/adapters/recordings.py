"""Cohere and Ollama replies in each vendor's wire format, and the means to replay them.

TASK-0441. The Cohere and Ollama adapters were written against hand-built fakes shaped like
the adapters already behaved, so three of them were wrong against the libraries they wrap
and no test could say so. Everything here exists to stop a fake and a client library from
drifting apart again:

* The bodies below are what each server sends, in the format its API reference documents:
  Cohere's ``POST /v2/chat`` (whole reply, and the server-sent events of a streamed one) and
  ``POST /v1/embed``; Ollama's ``POST /api/chat`` (whole reply, and the newline-delimited
  parts of a streamed one) and ``POST /api/embed``. They were written from the references,
  not captured from a billed call, and were replayed through ``cohere`` 7.3.0 and ``ollama``
  0.6.3 — the versions ``uv.lock`` resolves — on 2026-11-12. The billed leg that can falsify
  them against a live server is ``tests/eval/test_live_providers.py``.
* ``Replay`` answers the real client libraries from those bodies at the transport, so a test
  can drive an adapter through the genuine client with no socket and no credential.
* ``parsed`` turns a body into the attribute-only object a typed client hands back, which is
  what the fakes in the call-path suites return. ``assert_same_shape`` is the proof that the
  two agree: every attribute a fake carries must exist, equal, on what the real client
  parsed from the same body.

What the real clients returned, recorded here because it is why the fakes look as they do:

* ``cohere.AsyncClientV2.chat`` returns a ``V2ChatResponse``; ``message.content`` is a list
  of typed blocks, ``finish_reason`` is upper-case, and both ``usage.billed_units`` and
  ``usage.tokens`` hold floats. ``chat_stream`` yields one typed event per server-sent
  event, and only ``content-delta`` carries answer text. ``cohere.AsyncClient.chat`` — the
  v1 client — refuses ``messages`` with a ``TypeError`` before sending anything.
* ``cohere.AsyncClient.embed`` returns the vectors on ``embeddings.float_``: the JSON key
  ``float`` is renamed, because it would shadow the builtin. The client splits the texts
  into requests of its own and merges the replies, rebuilding ``meta`` as it does: the
  merged ``meta`` always exists, always has a ``billed_units`` object (whose
  ``input_tokens`` is ``None`` when no reply stated one) and never has ``tokens``, whatever
  the server sent.
* ``ollama.AsyncClient.chat`` returns a ``ChatResponse`` and, streaming, an async iterator
  of them. ``embed`` returns an ``EmbedResponse``. None of them is a ``dict``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any, Final

import httpx
import pytest

Route = Callable[[dict[str, Any]], httpx.Response]

VECTOR: Final = [0.1, 0.2, 0.3]
OTHER_VECTOR: Final = [0.4, 0.5, 0.6]

COHERE_CHAT: Final[dict[str, Any]] = {
    "id": "c14c80c3-18eb-4519-9460-6c92edd8cfb4",
    "finish_reason": "COMPLETE",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "forty-one pounds"}],
    },
    "usage": {
        "billed_units": {"input_tokens": 5, "output_tokens": 3},
        "tokens": {"input_tokens": 71, "output_tokens": 3},
    },
}

COHERE_CHAT_STREAM: Final[list[dict[str, Any]]] = [
    {
        "type": "message-start",
        "id": "29f14a5a-11de-4cae-9800-25e4747408ea",
        "delta": {
            "message": {
                "role": "assistant",
                "content": [],
                "tool_plan": "",
                "tool_calls": [],
                "citations": [],
            }
        },
    },
    {
        "type": "content-start",
        "index": 0,
        "delta": {"message": {"content": {"type": "text", "text": ""}}},
    },
    {
        "type": "content-delta",
        "index": 0,
        "delta": {"message": {"content": {"text": "forty"}}},
    },
    {
        "type": "content-delta",
        "index": 0,
        "delta": {"message": {"content": {"text": "-one pounds"}}},
    },
    {"type": "content-end", "index": 0},
    {
        "type": "message-end",
        "delta": {
            "finish_reason": "COMPLETE",
            "usage": {
                "billed_units": {"input_tokens": 5, "output_tokens": 3},
                "tokens": {"input_tokens": 71, "output_tokens": 3},
            },
        },
    },
]

COHERE_EMBED: Final[dict[str, Any]] = {
    "id": "da6e531f-54c6-4a73-bf92-f60566d8d753",
    "texts": ["a", "b"],
    "embeddings": {"float": [VECTOR, OTHER_VECTOR]},
    "meta": {
        "api_version": {"version": "1"},
        "billed_units": {"input_tokens": 2},
    },
    "response_type": "embeddings_by_type",
}

COHERE_UNAUTHORIZED: Final[dict[str, Any]] = {"message": "invalid api token"}

OLLAMA_CHAT: Final[dict[str, Any]] = {
    "model": "llama3.2",
    "created_at": "2023-12-12T14:13:43.416799Z",
    "message": {"role": "assistant", "content": "forty-one pounds"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 5191566416,
    "load_duration": 2154458,
    "prompt_eval_count": 26,
    "prompt_eval_duration": 383809000,
    "eval_count": 298,
    "eval_duration": 4799921000,
}

OLLAMA_CHAT_STREAM: Final[list[dict[str, Any]]] = [
    {
        "model": "llama3.2",
        "created_at": "2023-08-04T08:52:19.385406455-07:00",
        "message": {"role": "assistant", "content": "forty"},
        "done": False,
    },
    {
        "model": "llama3.2",
        "created_at": "2023-08-04T08:52:19.385406455-07:00",
        "message": {"role": "assistant", "content": "-one pounds"},
        "done": False,
    },
    {
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 4883583458,
        "load_duration": 1334875,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 342546000,
        "eval_count": 282,
        "eval_duration": 4535599000,
    },
]

OLLAMA_EMBED: Final[dict[str, Any]] = {
    "model": "all-minilm",
    "embeddings": [VECTOR, OTHER_VECTOR],
    "total_duration": 14143917,
    "load_duration": 1019500,
    "prompt_eval_count": 8,
}

OLLAMA_MODEL_NOT_FOUND: Final[dict[str, Any]] = {
    "error": "model 'test-model' not found, try pulling it first"
}

_RENAMED_ATTRIBUTES: Final[Mapping[str, str]] = {"float": "float_"}


def parsed(body: Any) -> Any:
    """Return a wire body as the attribute-only object a typed client parses it into."""
    if isinstance(body, dict):
        return SimpleNamespace(
            **{_RENAMED_ATTRIBUTES.get(key, key): parsed(value) for key, value in body.items()}
        )
    if isinstance(body, list):
        return [parsed(item) for item in body]
    return body


def assert_same_shape(real: Any, fake: Any, path: str = "reply") -> None:
    """Fail unless everything the fake carries is on the real client's object, and equal."""
    assert not isinstance(real, dict), f"{path} is a dict on the real client, not an object"
    if isinstance(fake, SimpleNamespace):
        for name, expected in vars(fake).items():
            assert hasattr(real, name), f"the fake has {path}.{name}; the real object has not"
            _assert_same_value(getattr(real, name), expected, f"{path}.{name}")
        return
    _assert_same_value(real, fake, path)


def _assert_same_value(real: Any, fake: Any, path: str) -> None:
    """Compare one attribute, descending into nested objects and lists."""
    if isinstance(fake, SimpleNamespace):
        assert_same_shape(real, fake, path)
    elif isinstance(fake, list):
        assert isinstance(real, list), f"{path} is a list on the fake only"
        assert len(real) == len(fake), f"{path} holds {len(real)} items, the fake {len(fake)}"
        for index, (real_item, fake_item) in enumerate(zip(real, fake, strict=True)):
            _assert_same_value(real_item, fake_item, f"{path}[{index}]")
    else:
        assert real == fake, f"{path} is {real!r} on the real object and {fake!r} on the fake"


def require_library(package: str) -> None:
    """Skip the calling test, saying why, when a client library is not installed."""
    if find_spec(package) is None:
        pytest.skip(
            f"the {package} client library is not installed, so a recorded response cannot "
            f'be replayed through it; install it with pip install "fasterrag[{package}]"'
        )


def json_reply(body: Any, status: int = 200) -> Route:
    """Return a route answering every request with ``body`` as JSON."""
    return lambda _request: httpx.Response(status, json=body)


def sse_reply(events: Iterable[dict[str, Any]]) -> Route:
    """Return a route answering with server-sent events, as Cohere streams a chat."""
    payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return lambda _request: httpx.Response(
        200, content=payload.encode(), headers={"content-type": "text/event-stream"}
    )


def ndjson_reply(parts: Iterable[dict[str, Any]]) -> Route:
    """Return a route answering with newline-delimited JSON, as Ollama streams a chat."""
    payload = "".join(json.dumps(part) + "\n" for part in parts)
    return lambda _request: httpx.Response(
        200, content=payload.encode(), headers={"content-type": "application/x-ndjson"}
    )


def refusing_connections(_request: dict[str, Any]) -> httpx.Response:
    """Fail the way a server that is not running does."""
    raise httpx.ConnectError("All connection attempts failed")


class Replay:
    """Answers the real client libraries from recordings, at the transport, with no socket.

    It replaces ``httpx.AsyncHTTPTransport.handle_async_request``, the last call before a
    connection opens, so the adapter builds its own client and the library runs for real.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, routes: Mapping[str, Route]) -> None:
        self.routes = dict(routes)
        self.requests: list[tuple[httpx.URL, dict[str, Any]]] = []
        replay = self

        async def handle(_transport: Any, request: httpx.Request) -> httpx.Response:
            await request.aread()
            return replay.answer(request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)

    def answer(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer it from the route for its path."""
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.requests.append((request.url, body))
        route = self.routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, json={"message": f"no recording for {request.url.path}"})
        return route(body)

    @property
    def paths(self) -> list[str]:
        """Return the path of every request made, in order."""
        return [url.path for url, _ in self.requests]

    @property
    def bodies(self) -> list[dict[str, Any]]:
        """Return the JSON body of every request made, in order."""
        return [body for _, body in self.requests]
