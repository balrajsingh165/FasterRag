"""Vendor-neutral embedding provider contract.

Two calls, deliberately separated: documents and queries embed through different paths
because several models require different instruction prefixes for each, and conflating
them quietly costs retrieval quality.

Adapters report ``model`` and ``model_version``. Those two values are recorded on every
chunk and in the index lockfile, and they are the anchor drift detection compares against
(D1, ``docs/data-model.md``) — an adapter that reports a vague version makes drift
undetectable.

Adapters report billed usage on ``EmbeddingResult.total_tokens`` when the provider states
it, mirroring ``Completion.prompt_tokens`` on the generation side. It is the *provider's*
figure, not ours: the embedding pool prices it through ``services.estimation`` so a bill can
be reconciled against an invoice rather than against our own tokenizer's guess. A provider
that states nothing reports ``None``, never ``0``.

Adapters do not retry. They classify failures — setting ``retryable`` from what the
provider actually said — and the embedding worker pool owns backoff, so retry policy
lives in exactly one place (``docs/reliability.md`` §2).

Two refusals say more than "retry" or "do not" (TASK-0478). A request refused for its size —
``413``, or a token or batch limit named in the message — is an :class:`EmbedTooLargeError`:
the same request will fail forever, but a smaller one may not, so the pool halves the batch
instead of retrying it. A retryable refusal that names a pause in ``Retry-After`` (or
OpenAI's ``retry-after-ms``) is an :class:`EmbedThrottledError` carrying it, so the pool can
wait as long as it was asked rather than as long as its own backoff guesses. A plugin
adapter raises either one to get the same treatment.
"""

from __future__ import annotations

import math
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.utils import mktime_tz, parsedate_tz
from typing import ClassVar, Final

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError, ErrorCode

__all__ = [
    "ADAPTER_API_VERSION",
    "EmbedThrottledError",
    "EmbedTooLargeError",
    "EmbeddingAdapter",
    "EmbeddingResult",
    "classify_provider_failure",
    "require_extra",
    "require_key",
    "retry_after_seconds",
    "sum_reported",
]

ADAPTER_API_VERSION: Final = "1.0"

_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
_TIMEOUT_STATUSES: Final[frozenset[int]] = frozenset({408, 504})
_SERVER_ERROR_THRESHOLD: Final = 500
_RATE_LIMITED: Final = 429
_PROGRAMMING_ERRORS: Final[tuple[type[BaseException], ...]] = (TypeError, AttributeError)
_PAYLOAD_TOO_LARGE: Final = 413
_SIZE_STATUSES: Final[frozenset[int]] = frozenset({400, 413, 422, 500})
_SIZE_MESSAGE: Final = re.compile(
    r"context length|too many tokens|token limit|tokens per request|too large|too long"
    r"|number of (?:texts|inputs)|too many (?:texts|inputs)|max(?:imum)? input",
    re.IGNORECASE,
)
_SECONDS: Final = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")
_MILLISECONDS: Final = 1000.0


class EmbedTooLargeError(EmbedError):
    """The provider refused a request for its size: send less, not the same again."""

    def __init__(self, detail: str, *, trace_id: str | None = None) -> None:
        """Build the error; it is never retryable as sent."""
        super().__init__(detail, trace_id=trace_id, retryable=False)


class EmbedThrottledError(EmbedError):
    """A retryable refusal that named how long to wait before the next attempt."""

    def __init__(
        self, detail: str, *, retry_after_seconds: float, trace_id: str | None = None
    ) -> None:
        """Build the error with the pause the provider asked for, in seconds."""
        super().__init__(detail, trace_id=trace_id, retryable=True)
        self.retry_after_seconds = retry_after_seconds


def require_extra(provider: str, package: str, extra: str) -> ConfigError:
    """Return the error explaining which optional install a provider needs."""
    return ConfigError(
        f"embeddings.provider is {provider!r}, which needs the {package} package; "
        f"install it with 'pip install \"fasterrag[{extra}]\"'"
    )


def require_key(name: str | None, provider: str) -> str:
    """Return the credential a provider needs, or explain what is missing."""
    if not name:
        raise ConfigError(f"embeddings.api_key_env must name a variable for provider {provider!r}")

    value = os.environ.get(name)
    if not value or not value.strip():
        raise ConfigError(f"the {name} environment variable is not set")
    return value


def _headers(exc: BaseException) -> Mapping[str, str] | None:
    """Return the response headers a vendor error carries, wherever its SDK put them."""
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, Mapping):
        headers = getattr(getattr(exc, "response", None), "headers", None)
    if not isinstance(headers, Mapping):
        return None
    return {str(name).lower(): str(value) for name, value in headers.items()}


def _number(value: str | None) -> float | None:
    """Return a non-negative decimal header value, or ``None`` when it is not one."""
    if value is None or not _SECONDS.match(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Return the pause a refusal asked for, from ``retry-after-ms`` or ``Retry-After``.

    Seconds or an HTTP date; anything unreadable is ``None`` and a date already past is 0.
    """
    headers = _headers(exc)
    if headers is None:
        return None

    milliseconds = _number(headers.get("retry-after-ms"))
    if milliseconds is not None:
        return milliseconds / _MILLISECONDS

    raw = headers.get("retry-after")
    seconds = _number(raw)
    if seconds is not None or raw is None:
        return seconds

    parsed = parsedate_tz(raw)
    if parsed is None:
        return None
    return max(float(mktime_tz(parsed)) - time.time(), 0.0)


def _refused_for_size(exc: BaseException, status: int | None) -> bool:
    """Return whether the provider refused the request because of how much it carried."""
    if status == _PAYLOAD_TOO_LARGE:
        return True
    if status not in _SIZE_STATUSES:
        return False
    message = f"{exc} {getattr(exc, 'body', '')}"
    return _SIZE_MESSAGE.search(message) is not None


def classify_provider_failure(
    exc: BaseException,
    *,
    provider: str,
    operation: str,
    key_env: str | None,
) -> EmbedError:
    """Translate a provider failure into a typed, correctly-classified error."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    if status is None and isinstance(exc, _PROGRAMMING_ERRORS):
        return EmbedError(
            f"the {provider} adapter raised {type(exc).__name__} during {operation}; this is "
            "a fault in the adapter or a client library version it was not written against, "
            "not a provider outage",
            code=ErrorCode.INTERNAL,
            retryable=False,
        )

    if status in _AUTH_STATUSES:
        named = key_env or "embeddings.api_key_env"
        return EmbedError(
            f"{provider} rejected the credentials during {operation}; check the key in the "
            f"{named} environment variable",
            retryable=False,
        )

    if _refused_for_size(exc, status):
        return EmbedTooLargeError(
            f"{provider} refused the request as too large during {operation} (status {status}); "
            "the batch will be split, and an input refused on its own exceeds the model's limit"
        )

    if status in _TIMEOUT_STATUSES:
        return EmbedError(
            f"{provider} timed out during {operation}",
            code=ErrorCode.EMBED_PROVIDER_TIMEOUT,
            retryable=True,
        )

    if status == _RATE_LIMITED or (status is not None and status >= _SERVER_ERROR_THRESHOLD):
        pause = retry_after_seconds(exc)
        if pause is not None:
            return EmbedThrottledError(
                f"{provider} returned status {status} during {operation} and asked for a "
                f"{pause:.1f}s pause",
                retry_after_seconds=pause,
            )
        return EmbedError(
            f"{provider} returned status {status} during {operation}",
            retryable=True,
        )

    if status is not None:
        return EmbedError(
            f"{provider} returned status {status} during {operation}",
            retryable=False,
        )

    if isinstance(exc, TimeoutError):
        return EmbedError(
            f"{provider} timed out during {operation}",
            code=ErrorCode.EMBED_PROVIDER_TIMEOUT,
            retryable=True,
        )

    return EmbedError(
        f"{provider} was unreachable during {operation}: {type(exc).__name__}",
        retryable=True,
    )


def sum_reported(values: Sequence[int | None]) -> int | None:
    """Return the batch total, or ``None`` when any batch reported no usage.

    All-or-nothing on purpose: a partial sum is an under-count, and an under-count
    reconciles wrong against an invoice while looking like a receipt.
    """
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors for a batch, with the model that produced them."""

    vectors: list[list[float]]
    model: str
    model_version: str
    total_tokens: int | None = None
    """Tokens the provider said it billed, or ``None`` when it reported none.

    ``None`` and ``0`` are different facts and must stay different: ``0`` means nothing was
    billed (a fully cached batch), while ``None`` means there is no receipt to reconcile
    against an invoice. Collapsing the two prices unreported traffic as free.
    """


class EmbeddingAdapter(ABC):
    """Vendor-neutral embedding contract."""

    provider: ClassVar[str]

    def __init__(self, settings: Settings) -> None:
        """Build the adapter from validated configuration."""
        self.settings = settings
        self.config = settings.embeddings

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model identifier recorded on every chunk."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the model version, the anchor drift detection compares against."""

    @property
    @abstractmethod
    def dimensions(self) -> int | None:
        """Return the vector size, or None until it is known."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages for indexing, batched per ``embeddings.batch_size``."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Report whether the provider is usable, without raising."""

    @abstractmethod
    async def close(self) -> None:
        """Release any client or model the adapter holds."""

    def batches(self, texts: Sequence[str]) -> list[Sequence[str]]:
        """Split ``texts`` into request-sized batches.

        Batched embedding is far cheaper than one call per text, so every adapter
        batches; doing it here means no adapter can forget.
        """
        size = self.config.batch_size
        return [texts[start : start + size] for start in range(0, len(texts), size)]
