"""OpenAI embedding provider.

Requests are batched to ``embeddings.batch_size`` and every call carries the configured
``reliability.timeouts.embeddings_ms``, so no provider call is unbounded. Retries are the
worker pool's job; this adapter only classifies failures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fasterrag.adapters.embeddings.base import (
    EmbeddingAdapter,
    EmbeddingResult,
    classify_provider_failure,
    require_extra,
    require_key,
)
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError

__all__ = ["OpenAIEmbedder"]


class OpenAIEmbedder(EmbeddingAdapter):
    """Embeds through OpenAI's embeddings API."""

    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        """Build the adapter without opening a connection."""
        super().__init__(settings)
        self._client: Any | None = None
        self._timeout = settings.reliability.timeouts.embeddings_ms / 1000

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self.config.model

    @property
    def model_version(self) -> str:
        """Return the model version.

        OpenAI publishes no version endpoint for embedding models, so the pinned model id
        is the most specific value available; changing it is what drift detection sees.
        """
        return self.config.model

    @property
    def dimensions(self) -> int | None:
        """Return the configured dimensionality override, if any."""
        return self.config.dimensions

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise require_extra(self.provider, "openai", "openai") from exc

            self._client = AsyncOpenAI(
                api_key=require_key(self.config.api_key_env, self.provider),
                timeout=self._timeout,
                max_retries=0,
            )
        return self._client

    async def _embed(self, texts: Sequence[str]) -> tuple[list[list[float]], int]:
        """Embed one batch, returning vectors and the tokens the provider billed."""
        request: dict[str, Any] = {"model": self.config.model, "input": list(texts)}
        if self.config.dimensions is not None:
            request["dimensions"] = self.config.dimensions

        try:
            response = await self._connected().embeddings.create(**request)
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
                key_env=self.config.api_key_env,
            ) from exc

        vectors = [list(item.embedding) for item in response.data]
        tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
        return vectors, int(tokens)

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages in provider-sized batches."""
        vectors: list[list[float]] = []
        total = 0
        for batch in self.batches(texts):
            batch_vectors, tokens = await self._embed(batch)
            vectors.extend(batch_vectors)
            total += tokens

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
            total_tokens=total,
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
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
