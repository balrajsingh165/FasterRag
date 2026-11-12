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
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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
    "retry_after_of",
]

_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
_TIMEOUT_STATUSES: Final[frozenset[int]] = frozenset({408, 504})
_SERVER_ERROR_THRESHOLD: Final = 500
_RATE_LIMITED: Final = 429
_PROGRAMMING_ERRORS: Final[tuple[type[BaseException], ...]] = (TypeError, AttributeError)
_MILLISECONDS: Final = 1000.0


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


def _headers_of(exc: BaseException) -> Mapping[str, str]:
    """Return the response headers a vendor error carries, lower-cased, or none.

    Cohere's ``ApiError`` holds them as ``headers``; OpenAI's and Anthropic's status errors
    hold the ``httpx`` response. Ollama's error carries no headers at all.
    """
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, Mapping):
        headers = getattr(getattr(exc, "response", None), "headers", None)
    if not isinstance(headers, Mapping):
        return {}
    return {str(name).lower(): str(value) for name, value in headers.items()}


def retry_after_of(exc: BaseException) -> float | None:
    """Return the seconds a provider asked to wait before the next call, if it said.

    ``retry-after-ms`` (OpenAI's and Anthropic's finer form) wins over ``Retry-After``,
    which may be seconds or an HTTP date. Anything unreadable or negative is no answer.
    """
    headers = _headers_of(exc)
    candidates = (
        (headers.get("retry-after-ms"), _MILLISECONDS),
        (headers.get("retry-after"), 1.0),
    )
    for raw, unit in candidates:
        if raw is None:
            continue
        try:
            seconds = float(raw) / unit
        except ValueError:
            seconds = _seconds_until(raw) if unit == 1.0 else -1.0
        if seconds >= 0:
            return seconds
    return None


def _seconds_until(moment: str) -> float:
    """Return the seconds from now until an HTTP date, or -1 if it is not one."""
    try:
        when = parsedate_to_datetime(moment)
    except (TypeError, ValueError):
        return -1.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


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

    if status is None and isinstance(exc, _PROGRAMMING_ERRORS):
        return GenerationError(
            f"the {provider} adapter raised {type(exc).__name__} during {operation}; this is "
            "a fault in the adapter or a client library version it was not written against, "
            "not a provider outage",
            code=ErrorCode.INTERNAL,
            retryable=False,
        )

    if status in _AUTH_STATUSES:
        named = key_env or "llm.api_key_env"
        return GenerationError(
            f"{provider} rejected the credentials during {operation}; check the key in the "
            f"{named} environment variable",
            retryable=False,
        )

    if status in _TIMEOUT_STATUSES or isinstance(exc, TimeoutError):
        return GenerationError(
            f"{provider} timed out during {operation}",
            code=ErrorCode.GENERATION_TIMEOUT,
            retryable=True,
            retry_after=retry_after_of(exc),
        )

    if status == _RATE_LIMITED or (status is not None and status >= _SERVER_ERROR_THRESHOLD):
        return GenerationError(
            f"{provider} returned status {status} during {operation}",
            retryable=True,
            retry_after=retry_after_of(exc),
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
        """Build the adapter from validated configuration, opening no connection."""
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
