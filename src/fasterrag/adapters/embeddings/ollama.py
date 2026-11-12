"""Ollama embedding provider — a local server, so no credential is required.

The adapter calls the client's ``embed()``, which is the ``/api/embed`` route: one request
carries a whole batch as ``input`` and the reply holds one vector per text, in order, plus
``prompt_eval_count`` — the tokens the server evaluated for that batch. Until TASK-0441 it
called ``embeddings()``, the superseded ``/api/embeddings`` route, once per text, and then
read the reply as a ``dict``; from ``ollama`` 0.4 the client returns a typed response
object, so every embedding failed with "ollama returned an empty embedding". The route
needs an Ollama server that serves ``/api/embed``, and the adapter does not fall back to
the superseded one.

``prompt_eval_count`` is surfaced as ``EmbeddingResult.total_tokens``. Ollama runs locally
and is priced at zero (``LOCAL_PROVIDERS`` in ``services.estimation``), so the figure moves
no bill; it is reported because a server that states a count has given a receipt, and
``None`` is reserved for a reply that states none.

**Where the server address comes from.** There is no ``embeddings.base_url`` key. When
``llm.provider`` is ``ollama`` the embedder uses ``llm.base_url``, because one Ollama
server normally serves both. When the LLM is any other provider, ``llm.base_url`` is that
provider's endpoint — an OpenAI-compatible gateway, say — and the embedder no longer
borrows it: the client is built with no host, so it applies its own rule, the
``OLLAMA_HOST`` environment variable and otherwise ``http://127.0.0.1:11434``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fasterrag.adapters.embeddings.base import (
    EmbeddingAdapter,
    EmbeddingResult,
    classify_provider_failure,
    require_extra,
    sum_reported,
)
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError

__all__ = ["OllamaEmbedder"]


class OllamaEmbedder(EmbeddingAdapter):
    """Embeds through a local Ollama server."""

    provider = "ollama"

    def __init__(self, settings: Settings) -> None:
        """Build the adapter without opening a connection."""
        super().__init__(settings)
        self._client: Any | None = None
        self._timeout = settings.reliability.timeouts.embeddings_ms / 1000
        self._host = settings.llm.base_url if settings.llm.provider == self.provider else None

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self.config.model

    @property
    def model_version(self) -> str:
        """Return the model version; the pulled model tag is the most specific available."""
        return self.config.model

    @property
    def dimensions(self) -> int | None:
        """Return the configured dimensionality override, if any."""
        return self.config.dimensions

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise require_extra(self.provider, "ollama", "ollama") from exc

            self._client = AsyncClient(host=self._host, timeout=self._timeout)
        return self._client

    async def _embed(self, texts: Sequence[str]) -> tuple[list[list[float]], int | None]:
        """Embed one batch in one request, returning vectors and the tokens Ollama counted."""
        try:
            response = await self._connected().embed(model=self.config.model, input=list(texts))
        except ConfigError:
            raise
        # CRITICAL: the catch stays broad because the SDK raises a deep hierarchy of
        # transport and API errors. Every one must become a typed EmbedError so the worker
        # pool can decide whether to retry instead of a vendor exception escaping.
        except Exception as exc:
            raise classify_provider_failure(
                exc,
                provider=self.provider,
                operation="embed",
                key_env=None,
            ) from exc

        embeddings = getattr(response, "embeddings", None) or []
        vectors = [[float(value) for value in vector] for vector in embeddings]
        usable = sum(1 for vector in vectors if vector)
        if usable != len(texts) or len(vectors) != len(texts):
            raise EmbedError(
                f"ollama returned {usable} usable embeddings for {len(texts)} texts",
                retryable=False,
            )
        tokens = getattr(response, "prompt_eval_count", None)
        return vectors, None if tokens is None else int(tokens)

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages, one request per batch."""
        vectors: list[list[float]] = []
        counted: list[int | None] = []
        for batch in self.batches(texts):
            batch_vectors, tokens = await self._embed(batch)
            vectors.extend(batch_vectors)
            counted.append(tokens)

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
            total_tokens=sum_reported(counted),
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        vectors, _ = await self._embed([text])
        return vectors[0]

    async def health(self) -> HealthStatus:
        """Report reachability with a minimal, non-destructive call."""
        try:
            await self._embed(["health check"])
        except (ConfigError, EmbedError) as exc:
            return HealthStatus(healthy=False, detail=exc.detail)
        return HealthStatus(healthy=True, detail=f"{self.provider} reachable")

    async def close(self) -> None:
        """Release the client."""
        self._client = None
