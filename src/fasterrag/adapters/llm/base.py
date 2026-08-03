"""Vendor-neutral LLM provider contract.

Two call shapes, because they are genuinely different operations rather than one with a
flag: ``complete`` returns a whole answer and its token usage, while ``stream`` yields text
as it arrives so time-to-first-token stays low. A streamed call cannot report usage until it
ends, which is exactly why the SSE contract sends ``usage`` as a separate late event
(``docs/api-reference.md``).

Adapters classify failures and never retry. Retry policy lives with the caller so it exists
in one place, and a provider that refused a credential is never hammered
(``docs/reliability.md`` §2).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, ErrorCode, GenerationError

__all__ = [
    "Completion",
    "LLMAdapter",
    "classify_llm_failure",
    "require_llm_extra",
    "require_llm_key",
]

_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
_TIMEOUT_STATUSES: Final[frozenset[int]] = frozenset({408, 504})
_SERVER_ERROR_THRESHOLD: Final = 500
_RATE_LIMITED: Final = 429


@dataclass(frozen=True, slots=True)
class Completion:
    """A generated answer and what it cost."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """Return whether the provider stopped at the token limit rather than finishing."""
        return self.finish_reason in {"length", "max_tokens"}


def require_llm_extra(provider: str, package: str, extra: str) -> ConfigError:
    """Return the error explaining which optional install a provider needs."""
    return ConfigError(
        f"llm.provider is {provider!r}, which needs the {package} package; install it with "
        f"'pip install \"fasterrag[{extra}]\"'"
    )


def require_llm_key(name: str | None, provider: str) -> str:
    """Return the credential a provider needs, naming the variable if it is missing.

    Raises:
        ConfigError: If the variable is unset or blank. The value is never included.
    """
    if not name:
        raise ConfigError(f"llm.api_key_env must name a variable for provider {provider!r}")

    value = os.environ.get(name)
    if not value or not value.strip():
        raise ConfigError(f"the {name} environment variable is not set")
    return value


def classify_llm_failure(
    exc: BaseException,
    *,
    provider: str,
    operation: str,
    key_env: str | None,
) -> GenerationError:
    """Translate a provider failure into a typed, correctly-classified error."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    if status in _AUTH_STATUSES:
        named = key_env or "llm.api_key_env"
        return GenerationError(
            f"{provider} rejected the credentials during {operation}; check the key in the "
            f"{named} environment variable",
            retryable=False,
        )

    # A timeout gets its own code, mirroring the embedding side's split between
    # EMBED_PROVIDER_TIMEOUT and EMBED_PROVIDER_ERROR. The two call for different actions —
    # a timeout says raise reliability.timeouts.llm_ms or reduce the context, a hard failure
    # says look at the provider — and one code for both cannot say which.
    if status in _TIMEOUT_STATUSES or isinstance(exc, TimeoutError):
        return GenerationError(
            f"{provider} timed out during {operation}",
            code=ErrorCode.GENERATION_TIMEOUT,
            retryable=True,
        )

    if status == _RATE_LIMITED or (status is not None and status >= _SERVER_ERROR_THRESHOLD):
        return GenerationError(
            f"{provider} returned status {status} during {operation}", retryable=True
        )

    if status is not None:
        return GenerationError(
            f"{provider} returned status {status} during {operation}",
            code=ErrorCode.GENERATION_FAILED,
            retryable=False,
        )

    return GenerationError(
        f"{provider} was unreachable during {operation}: {type(exc).__name__}", retryable=True
    )


class LLMAdapter(ABC):
    """Vendor-neutral generation contract.

    Built-in providers and third-party ones registered through the ``fasterrag.llm`` entry
    point implement this identically (``docs/python-api.md`` §Extending).
    """

    provider: ClassVar[str]

    def __init__(self, settings: Settings) -> None:
        """Build the adapter from validated configuration, opening no connection.

        The client slot lives here because every provider needs one and every provider
        builds it lazily; keeping it in one place is what lets ``close`` mean the same
        thing everywhere.
        """
        self.settings = settings
        self.config = settings.llm
        self.timeout = settings.reliability.timeouts.llm_ms / 1000
        self._client: Any | None = None

    @property
    def model(self) -> str:
        """Return the model identifier recorded on every trace."""
        return self.config.model

    @abstractmethod
    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Generate a whole answer."""

    @abstractmethod
    def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield the answer as text deltas, first token as early as possible."""

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Report whether the provider is usable, without raising."""

    @abstractmethod
    async def close(self) -> None:
        """Release any client the adapter holds."""
