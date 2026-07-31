"""Cohere generation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fasterrag.adapters.llm.base import (
    Completion,
    LLMAdapter,
    classify_llm_failure,
    require_llm_extra,
    require_llm_key,
)
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.errors import ConfigError, GenerationError

__all__ = ["CohereGenerator"]


class CohereGenerator(LLMAdapter):
    """Generates through Cohere's chat API."""

    provider = "cohere"

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from cohere import AsyncClient
            except ImportError as exc:
                raise require_llm_extra(self.provider, "cohere", "cohere") from exc

            self._client = AsyncClient(
                api_key=require_llm_key(self.config.api_key_env, self.provider),
                timeout=self.timeout,
            )
        return self._client

    def _request(self, prompt: str, system: str | None) -> dict[str, Any]:
        """Build the request body shared by both call shapes."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Generate a whole answer."""
        try:
            response = await self._connected().chat(**self._request(prompt, system))
        except ConfigError:
            raise
        # CRITICAL: the catch stays broad because the SDK raises a deep hierarchy of
        # transport and API errors. Every one must become a typed GenerationError so the
        # degradation ladder can act on it instead of a vendor exception escaping.
        except Exception as exc:
            raise classify_llm_failure(
                exc,
                provider=self.provider,
                operation="complete",
                key_env=self.config.api_key_env,
            ) from exc

        return Completion(
            text=_text_of(response),
            model=self.config.model,
            prompt_tokens=_usage_of(response, "input_tokens"),
            completion_tokens=_usage_of(response, "output_tokens"),
            finish_reason=getattr(response, "finish_reason", None),
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield answer text as it arrives."""
        try:
            events = self._connected().chat_stream(**self._request(prompt, system))
            async for event in events:
                delta = getattr(event, "delta", None)
                message = getattr(delta, "message", None) if delta else None
                content = getattr(message, "content", None) if message else None
                text = getattr(content, "text", None) if content else None
                if text:
                    yield text
        except ConfigError:
            raise
        # CRITICAL: see complete(); a mid-stream vendor error must reach the caller as a
        # typed error so the SSE contract can emit its error event and close the stream.
        except Exception as exc:
            raise classify_llm_failure(
                exc, provider=self.provider, operation="stream", key_env=self.config.api_key_env
            ) from exc

    async def health(self) -> HealthStatus:
        """Report reachability with a minimal generation."""
        try:
            await self.complete("ping")
        except (ConfigError, GenerationError) as exc:
            return HealthStatus(healthy=False, detail=exc.detail)
        return HealthStatus(healthy=True, detail=f"{self.provider} reachable")

    async def close(self) -> None:
        """Release the client."""
        self._client = None


def _text_of(response: Any) -> str:
    """Extract the answer text from a chat response."""
    message = getattr(response, "message", None)
    blocks = getattr(message, "content", None) if message else None
    if not blocks:
        return ""
    return "".join(getattr(block, "text", "") or "" for block in blocks)


def _usage_of(response: Any, field: str) -> int:
    """Extract a billed token count, defaulting to zero when absent."""
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "tokens", None) if usage else None
    value = getattr(tokens, field, None) if tokens else None
    return int(value or 0)
