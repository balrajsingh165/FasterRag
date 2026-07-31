"""OpenAI and OpenAI-compatible generation.

One adapter serves both, because ``openai_compatible`` is the same wire protocol pointed at
a different host — vLLM, LM Studio, llama.cpp servers, TGI, and most hosted gateways speak
it (``docs/integrations.md`` §3). Treating them as two implementations would duplicate the
streaming logic for no benefit.
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

__all__ = ["OpenAICompatibleGenerator", "OpenAIGenerator"]


class OpenAIGenerator(LLMAdapter):
    """Generates through OpenAI's chat completions API."""

    provider = "openai"

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise require_llm_extra(self.provider, "openai", "openai") from exc

            self._client = AsyncOpenAI(
                api_key=require_llm_key(self.config.api_key_env, self.provider),
                base_url=self.config.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def _request(self, prompt: str, system: str | None, *, stream: bool) -> dict[str, Any]:
        """Build the request body shared by both call shapes."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        if stream:
            request["stream_options"] = {"include_usage": True}
        return request

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Generate a whole answer."""
        try:
            response = await self._connected().chat.completions.create(
                **self._request(prompt, system, stream=False)
            )
        except ConfigError:
            raise
        # CRITICAL: the catch stays broad because the SDK raises a deep hierarchy of
        # transport and API errors. Every one must become a typed GenerationError so the
        # degradation ladder can act on it instead of a vendor exception escaping.
        except Exception as exc:
            raise classify_llm_failure(
                exc, provider=self.provider, operation="complete", key_env=self.config.api_key_env
            ) from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.message.content or "",
            model=self.config.model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=choice.finish_reason,
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield answer text as it arrives."""
        try:
            events = await self._connected().chat.completions.create(
                **self._request(prompt, system, stream=True)
            )
            async for event in events:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
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


class OpenAICompatibleGenerator(OpenAIGenerator):
    """Generates through any endpoint speaking the OpenAI API shape.

    Configuration validation already requires ``llm.base_url`` for this provider, so a
    misconfigured endpoint fails at startup rather than on the first query.
    """

    provider = "openai_compatible"
