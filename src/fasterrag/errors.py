"""Typed error taxonomy shared by the API, the CLI, and the Python package.

Specified in ``docs/reliability.md`` §1; the ``code`` values are the stable,
machine-readable identifiers published in the ``docs/api-reference.md`` error table and
are never renamed once released. Every exception carries ``code``, ``trace_id``, and
``retryable`` so the three control surfaces report the same error identity: the API
renders it as an RFC 9457 problem document, the library raises the class, and the CLI
maps it to an exit code.

There is no bare ``except`` anywhere in this codebase and no exception is silently
swallowed: every caught error is either handled with its correlation id logged or
rethrown (``docs/CONTRIBUTING.md`` §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final

from fasterrag.observability.logging import current_trace_id

__all__ = [
    "PROBLEM_SPECS",
    "PROBLEM_TYPE_BASE",
    "CacheError",
    "ChunkError",
    "ConfigError",
    "EmbedError",
    "ErrorCode",
    "FasterRagError",
    "GenerationError",
    "IngestionError",
    "ParseError",
    "ProblemSpec",
    "ProviderError",
    "ProvisioningError",
    "RetrievalError",
    "problem_spec",
]

PROBLEM_TYPE_BASE: Final = "https://fasterrag.dev/problems/"


class ErrorCode(StrEnum):
    """Stable machine-readable error codes; renaming one is a breaking change."""

    CONFIG_INVALID = "CONFIG_INVALID"
    AUTH_MISSING = "AUTH_MISSING"
    AUTH_INVALID = "AUTH_INVALID"
    AUTH_SCOPE = "AUTH_SCOPE"
    TENANT_FORBIDDEN = "TENANT_FORBIDDEN"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    QUEUE_FULL = "QUEUE_FULL"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PARSE_FAILED = "PARSE_FAILED"
    CHUNK_FAILED = "CHUNK_FAILED"
    EMBED_PROVIDER_TIMEOUT = "EMBED_PROVIDER_TIMEOUT"
    EMBED_PROVIDER_ERROR = "EMBED_PROVIDER_ERROR"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    RERANK_FAILED = "RERANK_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    PROVISIONING_FAILED = "PROVISIONING_FAILED"
    CACHE_ERROR = "CACHE_ERROR"
    NOT_READY = "NOT_READY"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Transport metadata for one error code: how it renders as a problem document."""

    status: int
    slug: str
    title: str
    retryable: bool

    @property
    def type_uri(self) -> str:
        """Return the RFC 9457 ``type`` URI for this code."""
        return f"{PROBLEM_TYPE_BASE}{self.slug}"


PROBLEM_SPECS: Final[dict[ErrorCode, ProblemSpec]] = {
    ErrorCode.CONFIG_INVALID: ProblemSpec(500, "config-invalid", "Configuration is invalid", False),
    ErrorCode.AUTH_MISSING: ProblemSpec(401, "auth-missing", "Authentication is missing", False),
    ErrorCode.AUTH_INVALID: ProblemSpec(401, "auth-invalid", "Authentication is invalid", False),
    ErrorCode.AUTH_SCOPE: ProblemSpec(403, "auth-scope", "API key lacks the required scope", False),
    ErrorCode.TENANT_FORBIDDEN: ProblemSpec(
        403, "tenant-forbidden", "Key is not authorized for this tenant", False
    ),
    ErrorCode.VALIDATION_FAILED: ProblemSpec(
        422, "validation-failed", "Request failed schema validation", False
    ),
    ErrorCode.NOT_FOUND: ProblemSpec(404, "not-found", "Resource not found", False),
    ErrorCode.CONFLICT: ProblemSpec(409, "conflict", "Conflicting resource state", False),
    ErrorCode.PAYLOAD_TOO_LARGE: ProblemSpec(
        413, "payload-too-large", "Payload exceeds the configured limit", False
    ),
    ErrorCode.RATE_LIMITED: ProblemSpec(429, "rate-limited", "Rate limit exceeded", True),
    ErrorCode.QUEUE_FULL: ProblemSpec(429, "queue-full", "Ingestion queue is full", True),
    ErrorCode.BUDGET_EXCEEDED: ProblemSpec(402, "budget-exceeded", "Token budget exhausted", False),
    ErrorCode.PARSE_FAILED: ProblemSpec(422, "parse-failed", "Document could not be parsed", False),
    ErrorCode.CHUNK_FAILED: ProblemSpec(500, "chunk-failed", "Chunker invariant violated", False),
    ErrorCode.EMBED_PROVIDER_TIMEOUT: ProblemSpec(
        503, "provider-timeout", "Embedding provider timed out", True
    ),
    ErrorCode.EMBED_PROVIDER_ERROR: ProblemSpec(
        503, "provider-error", "Embedding provider failed", True
    ),
    ErrorCode.RETRIEVAL_FAILED: ProblemSpec(503, "retrieval-failed", "Retrieval failed", True),
    ErrorCode.RERANK_FAILED: ProblemSpec(503, "rerank-failed", "Reranking failed", True),
    ErrorCode.GENERATION_FAILED: ProblemSpec(503, "generation-failed", "Generation failed", True),
    ErrorCode.INSUFFICIENT_EVIDENCE: ProblemSpec(
        200, "insufficient-evidence", "Insufficient evidence to answer", False
    ),
    ErrorCode.CIRCUIT_OPEN: ProblemSpec(503, "circuit-open", "Circuit breaker is open", True),
    ErrorCode.PROVISIONING_FAILED: ProblemSpec(
        500, "provisioning-failed", "Provisioning step failed", False
    ),
    ErrorCode.CACHE_ERROR: ProblemSpec(500, "cache-error", "Cache backend failure", False),
    ErrorCode.NOT_READY: ProblemSpec(503, "not-ready", "Service is not ready", True),
    ErrorCode.INTERNAL: ProblemSpec(500, "internal", "Internal error", False),
}


def problem_spec(code: ErrorCode) -> ProblemSpec:
    """Return the transport metadata registered for ``code``."""
    return PROBLEM_SPECS[code]


class FasterRagError(Exception):
    """Root of the fasterRag exception hierarchy.

    Every subclass carries the same three members the API, CLI, and library all report:
    a stable ``code``, the ``trace_id`` that correlates logs and spans, and whether the
    caller may retry.
    """

    default_code: ClassVar[ErrorCode] = ErrorCode.INTERNAL

    def __init__(
        self,
        detail: str,
        *,
        code: ErrorCode | None = None,
        trace_id: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Build the error.

        Args:
            detail: Human-readable cause. Must never contain a secret value.
            code: Stable error code; defaults to the subclass's canonical code.
            trace_id: Correlation id; defaults to the id bound to the current context.
            retryable: Overrides the code's default retry semantics. Adapters set this
                at their boundary, where the provider's semantics are known (HTTP
                429/503 are retryable, 401 is not).
        """
        super().__init__(detail)
        self.detail = detail
        self.code = code if code is not None else self.default_code
        self.trace_id = trace_id if trace_id is not None else current_trace_id()
        spec = problem_spec(self.code)
        self.retryable = retryable if retryable is not None else spec.retryable

    @property
    def status(self) -> int:
        """Return the HTTP status this error maps to."""
        return problem_spec(self.code).status

    @property
    def title(self) -> str:
        """Return the short, human-readable summary of this error's kind."""
        return problem_spec(self.code).title

    def __str__(self) -> str:
        """Return ``CODE: detail`` so log lines identify the error without parsing."""
        return f"{self.code}: {self.detail}"


class ConfigError(FasterRagError):
    """Invalid configuration or a missing referenced environment variable.

    Raised at startup only — configuration is validated before anything serves traffic,
    so a running process is never misconfigured.
    """

    default_code: ClassVar[ErrorCode] = ErrorCode.CONFIG_INVALID


class IngestionError(FasterRagError):
    """Ingestion-pipeline failure.

    Concrete call sites pass the precise code from the error table (for example
    ``ErrorCode.QUEUE_FULL`` on bounded-queue overflow); an unclassified ingestion
    failure keeps the inherited ``INTERNAL`` code.
    """


class ParseError(IngestionError):
    """A document could not be parsed; the dead-letter reason code is ``PARSE_FAILED``."""

    default_code: ClassVar[ErrorCode] = ErrorCode.PARSE_FAILED


class ChunkError(IngestionError):
    """A chunker invariant was violated (offsets, overlap, or size bounds)."""

    default_code: ClassVar[ErrorCode] = ErrorCode.CHUNK_FAILED


class EmbedError(IngestionError):
    """The embedding stage failed; pass ``EMBED_PROVIDER_TIMEOUT`` for timeouts."""

    default_code: ClassVar[ErrorCode] = ErrorCode.EMBED_PROVIDER_ERROR


class RetrievalError(FasterRagError):
    """A retrieval leg, the fusion step, or reranking failed.

    Reranker failures pass ``ErrorCode.RERANK_FAILED``; when the degradation ladder is
    enabled the query is answered in ``hybrid_only`` mode instead of failing.
    """

    default_code: ClassVar[ErrorCode] = ErrorCode.RETRIEVAL_FAILED


class GenerationError(FasterRagError):
    """LLM generation failed after retries."""

    default_code: ClassVar[ErrorCode] = ErrorCode.GENERATION_FAILED


class ProviderError(FasterRagError):
    """External provider transport or API error.

    Call sites pass the precise code (``EMBED_PROVIDER_ERROR``, ``CIRCUIT_OPEN``) and set
    ``retryable`` from the provider's response semantics.
    """


class CacheError(FasterRagError):
    """Cache backend failure; the system degrades to cache-off rather than failing queries."""

    default_code: ClassVar[ErrorCode] = ErrorCode.CACHE_ERROR


class ProvisioningError(FasterRagError):
    """Doctor or provisioner failure, carrying a concrete fix-it hint."""

    default_code: ClassVar[ErrorCode] = ErrorCode.PROVISIONING_FAILED

    def __init__(
        self,
        detail: str,
        *,
        fix: str | None = None,
        code: ErrorCode | None = None,
        trace_id: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Build the error.

        Args:
            detail: Human-readable cause. Must never contain a secret value.
            fix: The actionable instruction shown to the operator; appended to the
                problem document's ``detail`` so the API and the CLI report the same
                remedy.
            code: Stable error code; defaults to ``PROVISIONING_FAILED``.
            trace_id: Correlation id; defaults to the id bound to the current context.
            retryable: Overrides the code's default retry semantics.
        """
        super().__init__(detail, code=code, trace_id=trace_id, retryable=retryable)
        self.fix = fix
