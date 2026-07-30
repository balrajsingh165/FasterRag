"""Vendor-neutral embedding provider contract.

Two calls, deliberately separated: documents and queries embed through different paths
because several models require different instruction prefixes for each, and conflating
them quietly costs retrieval quality.

Adapters report ``model`` and ``model_version``. Those two values are recorded on every
chunk and in the index lockfile, and they are the anchor drift detection compares against
(D1, ``docs/data-model.md``) — an adapter that reports a vague version makes drift
undetectable.

Adapters do not retry. They classify failures — setting ``retryable`` from what the
provider actually said — and the embedding worker pool owns backoff, so retry policy
lives in exactly one place (``docs/reliability.md`` §2).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Final

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError, ErrorCode

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingResult",
    "classify_provider_failure",
    "require_extra",
    "require_key",
]

_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
_TIMEOUT_STATUSES: Final[frozenset[int]] = frozenset({408, 504})
_SERVER_ERROR_THRESHOLD: Final = 500
_RATE_LIMITED: Final = 429


def require_extra(provider: str, package: str, extra: str) -> ConfigError:
    """Return the error explaining which optional install a provider needs."""
    return ConfigError(
        f"embeddings.provider is {provider!r}, which needs the {package} package; "
        f"install it with 'pip install \"fasterrag[{extra}]\"'"
    )


def require_key(name: str | None, provider: str) -> str:
    """Return the credential a provider needs, or explain what is missing.

    Configuration validation already guarantees the variable is present at startup, so
    reaching this error means the environment changed underneath a running process.

    Raises:
        ConfigError: If the variable is unset or blank. The value is never included.
    """
    if not name:
        raise ConfigError(f"embeddings.api_key_env must name a variable for provider {provider!r}")

    value = os.environ.get(name)
    if not value or not value.strip():
        raise ConfigError(f"the {name} environment variable is not set")
    return value


def classify_provider_failure(
    exc: BaseException,
    *,
    provider: str,
    operation: str,
    key_env: str | None,
) -> EmbedError:
    """Translate a provider failure into a typed, correctly-classified error.

    Retryability comes from what the provider actually reported: rate limits and server
    faults are worth retrying, a rejected credential never is. Getting this wrong means
    hammering a provider that has already refused you.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    if status in _AUTH_STATUSES:
        named = key_env or "embeddings.api_key_env"
        return EmbedError(
            f"{provider} rejected the credentials during {operation}; check the key in the "
            f"{named} environment variable",
            retryable=False,
        )

    if status in _TIMEOUT_STATUSES:
        return EmbedError(
            f"{provider} timed out during {operation}",
            code=ErrorCode.EMBED_PROVIDER_TIMEOUT,
            retryable=True,
        )

    if status == _RATE_LIMITED or (status is not None and status >= _SERVER_ERROR_THRESHOLD):
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


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors for a batch, with the model that produced them."""

    vectors: list[list[float]]
    model: str
    model_version: str
    total_tokens: int | None = None


class EmbeddingAdapter(ABC):
    """Vendor-neutral embedding contract.

    Built-in providers and third-party ones registered through the
    ``fasterrag.embeddings`` entry point implement this identically
    (``docs/python-api.md`` §Extending).
    """

    provider: ClassVar[str]

    def __init__(self, settings: Settings) -> None:
        """Build the adapter from validated configuration.

        The one-argument constructor is the registration contract, matching the vector
        database adapters. Construction stays cheap: models are loaded and connections
        opened on first use, so a worker that never embeds never pays for a model.
        """
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
