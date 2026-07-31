"""Anthropic generation.

Anthropic takes the system prompt as a top-level parameter rather than as a message, which
is also what makes its prompt caching addressable: a stable system block can be cached
across calls while the user turn varies. That distinction is why the adapter keeps ``system``
separate all the way down rather than flattening it into the message list.
"""

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

__all__ = ["AnthropicGenerator"]


class AnthropicGenerator(LLMAdapter):
    """Generates through Anthropic's messages API."""

    provider = "anthropic"

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise require_llm_extra(self.provider, "anthropic", "anthropic") from exc

            self._client = AsyncAnthropic(
                api_key=require_llm_key(self.config.api_key_env, self.provider),
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def _request(self, prompt: str, system: str | None) -> dict[str, Any]:
        """Build the request body, keeping the system prompt out of the messages."""
        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        return request

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Generate a whole answer."""
        try:
            response = await self._connected().messages.create(**self._request(prompt, system))
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

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=self.config.model,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            finish_reason=getattr(response, "stop_reason", None),
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield answer text as it arrives."""
        try:
            client = self._connected()
            async with client.messages.stream(**self._request(prompt, system)) as events:
                async for text in events.text_stream:
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
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
