import pytest

from fasterrag.errors import (
    PROBLEM_SPECS,
    PROBLEM_TYPE_BASE,
    CacheError,
    ChunkError,
    ConfigError,
    EmbedError,
    ErrorCode,
    FasterRagError,
    GenerationError,
    IngestionError,
    ParseError,
    ProviderError,
    ProvisioningError,
    RetrievalError,
    problem_spec,
)
from fasterrag.observability.logging import use_trace_id

DOCUMENTED_STATUSES = {
    ErrorCode.CONFIG_INVALID: 500,
    ErrorCode.AUTH_MISSING: 401,
    ErrorCode.AUTH_INVALID: 401,
    ErrorCode.AUTH_SCOPE: 403,
    ErrorCode.TENANT_FORBIDDEN: 403,
    ErrorCode.VALIDATION_FAILED: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.QUEUE_FULL: 429,
    ErrorCode.BUDGET_EXCEEDED: 402,
    ErrorCode.PARSE_FAILED: 422,
    ErrorCode.CHUNK_FAILED: 500,
    ErrorCode.EMBED_PROVIDER_TIMEOUT: 503,
    ErrorCode.EMBED_PROVIDER_ERROR: 503,
    ErrorCode.RETRIEVAL_FAILED: 503,
    ErrorCode.RERANK_FAILED: 503,
    ErrorCode.GENERATION_FAILED: 503,
    ErrorCode.INSUFFICIENT_EVIDENCE: 200,
    ErrorCode.CIRCUIT_OPEN: 503,
    ErrorCode.PROVISIONING_FAILED: 500,
    ErrorCode.CACHE_ERROR: 500,
    ErrorCode.NOT_READY: 503,
    ErrorCode.INTERNAL: 500,
}

DEFAULT_CODES = [
    (FasterRagError, ErrorCode.INTERNAL),
    (ConfigError, ErrorCode.CONFIG_INVALID),
    (IngestionError, ErrorCode.INTERNAL),
    (ParseError, ErrorCode.PARSE_FAILED),
    (ChunkError, ErrorCode.CHUNK_FAILED),
    (EmbedError, ErrorCode.EMBED_PROVIDER_ERROR),
    (RetrievalError, ErrorCode.RETRIEVAL_FAILED),
    (GenerationError, ErrorCode.GENERATION_FAILED),
    (ProviderError, ErrorCode.INTERNAL),
    (CacheError, ErrorCode.CACHE_ERROR),
    (ProvisioningError, ErrorCode.PROVISIONING_FAILED),
]


def test_every_code_has_a_spec() -> None:
    assert set(PROBLEM_SPECS) == set(ErrorCode)


def test_statuses_match_the_documented_error_table() -> None:
    assert {code: spec.status for code, spec in PROBLEM_SPECS.items()} == DOCUMENTED_STATUSES


def test_type_uri_matches_the_documented_example() -> None:
    spec = problem_spec(ErrorCode.EMBED_PROVIDER_TIMEOUT)
    assert spec.type_uri == f"{PROBLEM_TYPE_BASE}provider-timeout"
    assert spec.title == "Embedding provider timed out"


def test_slugs_are_unique() -> None:
    slugs = [spec.slug for spec in PROBLEM_SPECS.values()]
    assert len(slugs) == len(set(slugs))


@pytest.mark.parametrize(("error_class", "expected"), DEFAULT_CODES)
def test_default_codes(error_class: type[FasterRagError], expected: ErrorCode) -> None:
    assert error_class("boom").code == expected


def test_taxonomy_hierarchy_matches_the_specification() -> None:
    assert issubclass(ParseError, IngestionError)
    assert issubclass(ChunkError, IngestionError)
    assert issubclass(EmbedError, IngestionError)
    for error_class, _ in DEFAULT_CODES:
        assert issubclass(error_class, FasterRagError)


def test_error_carries_code_trace_id_and_retryable() -> None:
    with use_trace_id("4bf92f3577b34da6a3ce929d0e0e4736"):
        error = EmbedError("provider call failed")

    assert error.code is ErrorCode.EMBED_PROVIDER_ERROR
    assert error.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert error.retryable is True
    assert error.status == 503


def test_trace_id_is_minted_when_no_context_is_bound() -> None:
    error = ConfigError("missing key")
    assert len(error.trace_id) == 32
    assert int(error.trace_id, 16) >= 0


def test_retryable_can_be_overridden_at_the_adapter_boundary() -> None:
    error = ProviderError("unauthorized", code=ErrorCode.EMBED_PROVIDER_ERROR, retryable=False)
    assert error.retryable is False


def test_explicit_code_overrides_the_class_default() -> None:
    error = IngestionError("queue is full", code=ErrorCode.QUEUE_FULL)
    assert error.code is ErrorCode.QUEUE_FULL
    assert error.status == 429
    assert error.retryable is True


def test_provisioning_error_carries_a_fix_hint() -> None:
    error = ProvisioningError("docker is not running", fix="Start Docker Desktop and retry.")
    assert error.fix == "Start Docker Desktop and retry."
    assert error.code is ErrorCode.PROVISIONING_FAILED


def test_str_includes_the_stable_code() -> None:
    assert str(ParseError("bad pdf")) == "PARSE_FAILED: bad pdf"
