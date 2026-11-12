"""Ollama generation against a local server, so no credential is required.

From ``ollama`` 0.4 the client parses every reply into a typed response object
(``ChatResponse``, holding a ``Message``) rather than returning the decoded JSON as a
``dict``. This adapter reads the typed object by attribute. Until TASK-0441 it tested
``isinstance(response, dict)`` first, which is false for a typed response, so every answer
came back as empty text beside correct token counts and a stream yielded nothing — a
failure with no error anywhere. ``pyproject.toml`` pins the client to the versions that
return the typed shape.

A streamed reply ends with a part whose ``content`` is empty and whose ``done`` is true; it
carries the token counts, which ``stream`` does not report, and is skipped like any other
part without text.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fasterrag.adapters.llm.base import (
    Completion,
    LLMAdapter,
    classify_llm_failure,
    require_llm_extra,
)
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.errors import ConfigError, GenerationError

__all__ = ["OllamaGenerator"]


class OllamaGenerator(LLMAdapter):
    """Generates through a local Ollama server."""

    provider = "ollama"

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise require_llm_extra(self.provider, "ollama", "ollama") from exc

            self._client = AsyncClient(host=self.config.base_url, timeout=self.timeout)
        return self._client

    def _request(self, prompt: str, system: str | None, *, stream: bool) -> dict[str, Any]:
        """Build the request body shared by both call shapes."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Generate a whole answer."""
        try:
            response = await self._connected().chat(**self._request(prompt, system, stream=False))
        except ConfigError:
            raise
        # CRITICAL: the catch stays broad because the SDK raises a deep hierarchy of
        # transport and API errors. Every one must become a typed GenerationError so the
        # degradation ladder can act on it instead of a vendor exception escaping.
        except Exception as exc:
            raise classify_llm_failure(
                exc, provider=self.provider, operation="complete", key_env=None
            ) from exc

        return Completion(
            text=_text_of(response),
            model=self.config.model,
            prompt_tokens=int(getattr(response, "prompt_eval_count", None) or 0),
            completion_tokens=int(getattr(response, "eval_count", None) or 0),
            finish_reason=getattr(response, "done_reason", None),
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield answer text as it arrives."""
        try:
            events = await self._connected().chat(**self._request(prompt, system, stream=True))
            async for event in events:
                text = _text_of(event)
                if text:
                    yield text
        except ConfigError:
            raise
        # CRITICAL: see complete(); a mid-stream vendor error must reach the caller as a
        # typed error so the SSE contract can emit its error event and close the stream.
        except Exception as exc:
            raise classify_llm_failure(
                exc, provider=self.provider, operation="stream", key_env=None
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
    """Return the text of a chat response or stream part, empty when it carries none."""
    content = getattr(getattr(response, "message", None), "content", None)
    return content if isinstance(content, str) else ""
