"""Pydantic models mirroring ``docs/config-reference.md``.

Every key, default, bound, and cross-field rule in that document is enforced here, so
the reference and the schema cannot drift without a test failing. Sections reject
unknown keys (``extra="forbid"``) — a typo in ``config.yaml`` is a startup failure that
names the key, never a silently ignored setting. Models are frozen: configuration is
read-only once loaded.

Environment-variable *presence* (cross-field rule 9) is checked by
:mod:`fasterrag.config.loader`, which is the only layer that touches the environment.
"""

from __future__ import annotations

import ipaddress
import platform
import re
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from fasterrag.config.plugins import (
    BUILTIN_EMBEDDING_PROVIDERS,
    BUILTIN_LLM_PROVIDERS,
    BUILTIN_VECTOR_DB_PROVIDERS,
    EMBEDDINGS_GROUP,
    LLM_GROUP,
    PLANNED_VECTOR_DB_PROVIDERS,
    VECTOR_DB_GROUP,
    require_selectable,
)

__all__ = [
    "AppSettings",
    "AutopilotSettings",
    "CacheSettings",
    "ChromaSettings",
    "ChunkingSettings",
    "CostSettings",
    "EmbeddingsSettings",
    "EvalSettings",
    "GenerationSettings",
    "IndexSettings",
    "IngestionSettings",
    "LlmSettings",
    "ObservabilitySettings",
    "ParsingSettings",
    "PgvectorSettings",
    "ReliabilitySettings",
    "RetrievalSettings",
    "SecuritySettings",
    "Settings",
    "TierRule",
    "TracesSettings",
    "VectorDbSettings",
    "WorkersSettings",
]

_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HOSTNAME_RE = re.compile(
    r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$"
)
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PG_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_DOCKER_VOLUME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_HTTP_HEADER_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")
_PATH_LIKE_RE = re.compile(r"^(~|\.{1,2}[\\/]|[A-Za-z]:[\\/])|[\\/]")

_REMOVED_APP_WORKERS = (
    "app.workers was removed on 2026-11-12: it was validated and read by nothing, because "
    "'fasterrag serve' runs one server process whatever it said. Delete the line; the "
    "pipeline pools are sized by workers.cpu_pool_size and workers.embedding_pool_size, "
    "and a separate worker process is tracked as TASK-0130"
)

ChunkStrategy = Literal["fixed", "recursive", "semantic", "layout", "late"]
TokenCounterMode = Literal["auto", "estimate", "model"]

PROVIDERS_REQUIRING_EMBEDDING_KEY: frozenset[str] = frozenset({"openai", "cohere"})
PROVIDERS_NOT_REQUIRING_LLM_KEY: frozenset[str] = frozenset({"ollama"})
EMBEDDING_PROVIDERS_WITHOUT_KEY: frozenset[str] = frozenset({"huggingface", "ollama"})
VECTOR_DB_PROVIDERS_WITHOUT_KEY: frozenset[str] = frozenset({"chroma", "pgvector"})

REDIS_URL_SCHEMES: frozenset[str] = frozenset({"redis", "rediss", "unix"})
DEFAULT_REDIS_URL: str = "redis://localhost:6379/0"


def _is_ip_address(value: str) -> bool:
    """Return whether ``value`` parses as an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def is_windows_or_wsl() -> bool:
    """Return whether the host is Windows or WSL, where Docker bind mounts are unsafe.

    Qdrant's install documentation records file-system data loss with bind mounts on
    these hosts, so a named Docker volume is mandatory there (cross-field rule 7).
    """
    if platform.system() == "Windows":
        return True
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _validate_redis_url(value: str) -> str:
    """Validate that ``value`` is a Redis connection URL the client can parse.

    Checked here rather than at first use so a typo fails at startup naming the key, not
    hours later inside a cache write that degrades silently to cache-off.
    """
    scheme, separator, rest = value.partition("://")
    if not separator or scheme.lower() not in REDIS_URL_SCHEMES or not rest:
        supported = ", ".join(f"{name}://" for name in sorted(REDIS_URL_SCHEMES))
        raise ValueError(f"must be a redis connection URL starting with one of: {supported}")
    return value


def _validate_env_var_name(value: str | None) -> str | None:
    """Validate that ``value`` is a usable environment-variable name."""
    if value is None:
        return None
    if not _ENV_VAR_RE.match(value):
        raise ValueError(
            "must be an environment variable NAME such as 'OPENAI_API_KEY', not a secret "
            "value; config.yaml never contains credentials"
        )
    return value


class Section(BaseModel):
    """Base for every configuration section: unknown keys rejected, values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # CRITICAL: a plugin provider never inherits a built-in provider's default key name.
    # `llm.api_key_env` defaults to OPENAI_API_KEY and `vector_db.api_key_env` to
    # QDRANT_API_KEY, and an adapter reads its credential through whatever name the field
    # holds — so a config that only says `provider: acme` would hand a third party's code
    # the name of another vendor's secret, and rule 9 would then refuse to start until that
    # unrelated secret was present. The schema cannot ask the plugin whether it needs a key
    # (that would load third-party code during validation), so it asks the operator: the
    # field must be written, as a variable name or as null. `model_fields_set` is what tells
    # a written null from an absent key.
    def _require_written_plugin_key(self, section: str, provider: str) -> None:
        """Refuse a plugin provider whose ``api_key_env`` was left to a built-in's default."""
        if "api_key_env" in self.model_fields_set:
            return
        default = type(self).model_fields["api_key_env"].default
        raise ValueError(
            f"{section}.api_key_env must be written out for plugin provider {provider!r}: "
            "name the environment variable holding its key, or set it to null if it needs "
            f"none. The default, {default!r}, names a built-in provider's key and is never "
            "handed to a plugin"
        )


class AppSettings(Section):
    """API server process settings."""

    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    @model_validator(mode="before")
    @classmethod
    def _refuse_removed_keys(cls, data: Any) -> Any:
        """Name a removed key and say why, where ``extra="forbid"`` would only call it unknown."""
        if isinstance(data, dict) and "workers" in data:
            raise ValueError(_REMOVED_APP_WORKERS)
        return data

    @field_validator("host")
    @classmethod
    def _check_host(cls, value: str) -> str:
        """Require a valid IP address or hostname."""
        if not value.strip():
            raise ValueError("must not be empty")
        if _is_ip_address(value) or _HOSTNAME_RE.match(value):
            return value
        raise ValueError(f"{value!r} is not a valid IP address or hostname")


class DockerSettings(Section):
    """System-managed container settings for ``vector_db.mode: docker``."""

    image: str = "qdrant/qdrant:v1.18.1"
    volume: str = "fasterrag_qdrant_storage"

    @field_validator("image")
    @classmethod
    def _check_pinned_tag(cls, value: str) -> str:
        """Require a pinned image tag or digest; ``latest`` is not reproducible."""
        if not value.strip():
            raise ValueError("must not be empty")
        if "@sha256:" in value:
            return value
        _, separator, tag = value.rpartition(":")
        if not separator or "/" in tag:
            raise ValueError(f"{value!r} must pin a tag, e.g. 'qdrant/qdrant:v1.9.0'")
        if tag == "latest":
            raise ValueError("image tag 'latest' is not allowed; pin an explicit version")
        return value

    @field_validator("volume")
    @classmethod
    def _check_volume(cls, value: str) -> str:
        """Require a named Docker volume, rejecting bind-mount paths on Windows/WSL."""
        if not value.strip():
            raise ValueError("must not be empty")
        if _PATH_LIKE_RE.search(value):
            if is_windows_or_wsl():
                raise ValueError(
                    f"{value!r} is a bind-mount path; on Windows/WSL a named Docker volume "
                    "is required because bind mounts risk file-system data loss"
                )
            return value
        if not _DOCKER_VOLUME_RE.match(value):
            raise ValueError(f"{value!r} is not a valid Docker volume name")
        return value


class PgvectorSettings(Section):
    """PostgreSQL connection and layout settings for ``vector_db.provider: pgvector``."""

    # CRITICAL: this defaults to None, not to "PGVECTOR_DSN". Every populated `*_env` field
    # anywhere in the settings tree is collected by `referenced_env_vars` and then required
    # to be present at startup by cross-field rule 9 — so a default here would demand a
    # PostgreSQL DSN from every deployment, including the Qdrant ones that will never open a
    # Postgres connection. Rule 10 below asks for it only when the provider is pgvector.
    dsn_env: str | None = None
    db_schema: str = "fasterrag"
    pool_max_size: Annotated[int, Field(ge=1, le=256)] = 8

    # CRITICAL: this is the ceiling for schema changes, snapshot copies, and restores only.
    # Ordinary statements are bounded by reliability.timeouts.vector_db_ms; applying that
    # bound to a CREATE INDEX would abort the HNSW build on any collection large enough to
    # need one, which is the failure this second budget exists to prevent.
    maintenance_timeout_ms: Annotated[int, Field(ge=1000)] = 600000

    @field_validator("dsn_env")
    @classmethod
    def _check_dsn_env(cls, value: str | None) -> str | None:
        """Require an environment-variable name, never a DSN containing a password."""
        return _validate_env_var_name(value)

    @field_validator("db_schema")
    @classmethod
    def _check_db_schema(cls, value: str) -> str:
        """Require a plain lower-case PostgreSQL identifier for the fasterRag schema."""
        if not _PG_IDENTIFIER_RE.match(value):
            raise ValueError(
                f"{value!r} must be a lower-case PostgreSQL identifier matching "
                "^[a-z_][a-z0-9_]{0,62}$"
            )
        return value


class ChromaSettings(Section):
    """Embedded-store settings for ``vector_db.provider: chroma``."""

    path: str = ".fasterrag/chroma"
    batch_size: Annotated[int, Field(ge=1, le=5000)] = 1000

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        """Require a non-empty directory path; Chroma runs in-process, not over a socket."""
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class CollectionSettings(Section):
    """Defaults applied when a collection is created through an adapter."""

    default_name: str = "default"
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    shard_number: Annotated[int, Field(ge=1)] = 1
    replication_factor: Annotated[int, Field(ge=1)] = 1

    @field_validator("default_name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        """Require a collection name matching the documented pattern."""
        if not _COLLECTION_NAME_RE.match(value):
            raise ValueError(f"{value!r} must match ^[a-zA-Z0-9_-]{{1,64}}$")
        return value


class VectorDbSettings(Section):
    """Vector database selection and connection settings."""

    provider: str = "qdrant"
    mode: Literal["docker", "external"] = "docker"
    host: str = "localhost"
    port: Annotated[int, Field(ge=1, le=65535)] = 6333
    grpc_port: Annotated[int, Field(ge=1, le=65535)] = 6334
    prefer_grpc: bool = False
    https: bool = False
    api_key_env: str | None = "QDRANT_API_KEY"
    docker: DockerSettings = DockerSettings()
    pgvector: PgvectorSettings = PgvectorSettings()
    chroma: ChromaSettings = ChromaSettings()
    collection: CollectionSettings = CollectionSettings()

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        """Require a built-in vector database or one an installed plugin registers."""
        return require_selectable(
            value,
            builtin=BUILTIN_VECTOR_DB_PROVIDERS,
            group=VECTOR_DB_GROUP,
            planned=PLANNED_VECTOR_DB_PROVIDERS,
        )

    @field_validator("host")
    @classmethod
    def _check_host(cls, value: str) -> str:
        """Require a non-empty host or IP."""
        if not value.strip():
            raise ValueError("must not be empty")
        if _is_ip_address(value) or _HOSTNAME_RE.match(value):
            return value
        raise ValueError(f"{value!r} is not a valid hostname or IP")

    @field_validator("api_key_env")
    @classmethod
    def _check_api_key_env(cls, value: str | None) -> str | None:
        """Require an environment-variable name, never a key value."""
        return _validate_env_var_name(value)

    @model_validator(mode="after")
    def _check_ports_differ(self) -> Self:
        """Enforce cross-field rule 4: the REST and gRPC ports must differ."""
        if self.port == self.grpc_port:
            raise ValueError(
                f"vector_db.grpc_port ({self.grpc_port}) must differ from vector_db.port; "
                "both 6333 (REST) and 6334 (gRPC) must be reachable"
            )
        return self

    @model_validator(mode="after")
    def _check_pgvector_dsn(self) -> Self:
        """Enforce cross-field rule 11: pgvector needs a DSN, and only pgvector does."""
        if self.provider == "pgvector" and self.pgvector.dsn_env is None:
            raise ValueError(
                "vector_db.pgvector.dsn_env is required when vector_db.provider is "
                "'pgvector'; name the environment variable holding the libpq connection "
                "string, for example dsn_env: PGVECTOR_DSN"
            )
        return self

    @model_validator(mode="after")
    def _check_plugin_key(self) -> Self:
        """Enforce cross-field rule 3 for a plugin: its key name is written, never defaulted."""
        if self.provider not in BUILTIN_VECTOR_DB_PROVIDERS:
            self._require_written_plugin_key("vector_db", self.provider)
        return self


class EmbeddingCacheSettings(Section):
    """Embedding cache keyed by content hash plus model and version."""

    enabled: bool = True
    backend: Literal["memory", "disk", "redis"] = "disk"
    max_entries: Annotated[int, Field(ge=1)] = 10_000
    redis_url: str = DEFAULT_REDIS_URL

    @field_validator("redis_url")
    @classmethod
    def _check_redis_url(cls, value: str) -> str:
        """Require a parseable Redis URL; only read when the backend is ``redis``."""
        return _validate_redis_url(value)


class TierRule(Section):
    """One tiered-embedding routing rule; the first matching rule wins."""

    match: dict[str, Any]
    provider: str
    model: str

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        """Require a built-in embedding provider or one an installed plugin registers."""
        return require_selectable(
            value, builtin=BUILTIN_EMBEDDING_PROVIDERS, group=EMBEDDINGS_GROUP
        )

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        """Require a non-empty model identifier."""
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class TieringSettings(Section):
    """Tiered embedding: route document classes to cheaper or more precise models."""

    enabled: bool = False
    rules: list[TierRule] = []

    @model_validator(mode="after")
    def _check_rules_present(self) -> Self:
        """Enforce cross-field rule 8: enabling tiering requires at least one rule."""
        if self.enabled and not self.rules:
            raise ValueError("embeddings.tiering.rules must be non-empty when tiering is enabled")
        return self


class EmbeddingsSettings(Section):
    """Embedding provider, model, batching, cache, and tiering."""

    provider: str = "huggingface"
    model: str = "BAAI/bge-small-en-v1.5"
    api_key_env: str | None = None
    batch_size: Annotated[int, Field(ge=1, le=2048)] = 64
    dimensions: Annotated[int, Field(ge=8)] | None = None
    cache: EmbeddingCacheSettings = EmbeddingCacheSettings()
    tiering: TieringSettings = TieringSettings()

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        """Require a built-in embedding provider or one an installed plugin registers."""
        return require_selectable(
            value, builtin=BUILTIN_EMBEDDING_PROVIDERS, group=EMBEDDINGS_GROUP
        )

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        """Require a non-empty model identifier."""
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _check_api_key_env(cls, value: str | None) -> str | None:
        """Require an environment-variable name, never a key value."""
        return _validate_env_var_name(value)

    @model_validator(mode="after")
    def _check_key_required(self) -> Self:
        """Enforce cross-field rule 3: hosted embedding providers need a key reference.

        A plugin is asked for nothing here: the default is already null, so there is no
        built-in key name to inherit, and whether it needs one is the plugin's to enforce.
        """
        if self.provider in PROVIDERS_REQUIRING_EMBEDDING_KEY and self.api_key_env is None:
            raise ValueError(
                f"embeddings.api_key_env is required for provider {self.provider!r}; "
                "set it to the name of the environment variable holding the key"
            )
        return self


class LlmSettings(Section):
    """Generation provider, model, sampling, and streaming."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str | None = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.1
    max_tokens: Annotated[int, Field(ge=1, le=32768)] = 1024
    streaming: bool = True

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        """Require a built-in LLM provider or one an installed plugin registers."""
        return require_selectable(value, builtin=BUILTIN_LLM_PROVIDERS, group=LLM_GROUP)

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        """Require a non-empty model identifier."""
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _check_api_key_env(cls, value: str | None) -> str | None:
        """Require an environment-variable name, never a key value."""
        return _validate_env_var_name(value)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        """Require an absolute HTTP(S) URL when an endpoint override is set."""
        if value is None:
            return None
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"{value!r} must be an absolute http:// or https:// URL")
        return value

    @model_validator(mode="after")
    def _check_provider_requirements(self) -> Self:
        """Enforce cross-field rule 3 for LLMs and the ``openai_compatible`` endpoint.

        A plugin may be keyless or keyed, which the schema cannot know without loading it,
        so it is only required to say which: the key name is written, never defaulted.
        """
        if self.provider not in BUILTIN_LLM_PROVIDERS:
            self._require_written_plugin_key("llm", self.provider)
            return self
        if self.provider not in PROVIDERS_NOT_REQUIRING_LLM_KEY and self.api_key_env is None:
            raise ValueError(
                f"llm.api_key_env is required for provider {self.provider!r}; "
                "only local providers may omit it"
            )
        if self.provider == "openai_compatible" and self.base_url is None:
            raise ValueError("llm.base_url is required for provider 'openai_compatible'")
        return self


class ParsingSettings(Section):
    """Parser thresholds: the OCR trigger, the OCR render, headings, and row grouping."""

    minimum_chars_per_page: Annotated[int, Field(ge=0, le=10_000)] = 40
    ocr_resolution: Annotated[int, Field(ge=72, le=1200)] = 200
    heading_size_ratio: Annotated[float, Field(ge=1.0, le=4.0)] = 1.15
    max_heading_chars: Annotated[int, Field(ge=1, le=1000)] = 120
    rows_per_block: Annotated[int, Field(ge=1, le=1000)] = 20


class ChunkingSettings(Section):
    """Chunking strategy, sizing, and contextual enrichment."""

    strategy: ChunkStrategy = "recursive"
    chunk_size: Annotated[int, Field(ge=64, le=2500)] = 768
    overlap: Annotated[int, Field(ge=0)] = 64
    token_counter: TokenCounterMode = "auto"
    chars_per_token: Annotated[int, Field(ge=1, le=16)] = 4
    semantic_percentile: Annotated[float, Field(ge=0.50, le=0.99)] = 0.95
    contextual_enrichment: bool = False
    context_tokens: Annotated[int, Field(ge=25, le=150)] = 75

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        """Enforce cross-field rule 2: overlap must be smaller than the chunk size."""
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"chunking.overlap ({self.overlap}) must be less than "
                f"chunking.chunk_size ({self.chunk_size})"
            )
        return self


class RetrievalSettings(Section):
    """Hybrid retrieval, Reciprocal Rank Fusion, and cross-encoder reranking."""

    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    hybrid: bool = True
    bm25_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    dense_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    rrf_k: Annotated[float, Field(gt=0)] = 60
    bm25_k1: Annotated[float, Field(ge=0.0, le=3.0)] = 1.2
    bm25_b: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: Annotated[int, Field(ge=10, le=1000)] = 100

    @model_validator(mode="after")
    def _check_retrieval_rules(self) -> Self:
        """Enforce cross-field rule 1 plus the leg-weight and reranker-model rules."""
        if self.top_k > self.rerank_top_n:
            raise ValueError(
                f"retrieval.top_k ({self.top_k}) must be less than or equal to "
                f"retrieval.rerank_top_n ({self.rerank_top_n})"
            )
        if self.bm25_weight + self.dense_weight <= 0:
            raise ValueError(
                "retrieval.bm25_weight + retrieval.dense_weight must be greater than 0"
            )
        if self.rerank and not self.reranker_model.strip():
            raise ValueError("retrieval.reranker_model is required when retrieval.rerank is true")
        return self


class GenerationSettings(Section):
    """Grounding, faithfulness, citation policy, and the assembled-context budget."""

    grounded_or_refuse: bool = False
    faithfulness_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    citations: bool = True
    context_budget_tokens: Annotated[int, Field(ge=1)] = 4000

    @model_validator(mode="after")
    def _check_citations(self) -> Self:
        """Enforce cross-field rule 5: refusing without citations is not a valid mode."""
        if self.grounded_or_refuse and not self.citations:
            raise ValueError(
                "generation.citations cannot be false while generation.grounded_or_refuse is true"
            )
        return self


class CacheSettings(Section):
    """Semantic response cache keyed by query-embedding similarity."""

    semantic: bool = False
    similarity_threshold: Annotated[float, Field(ge=0.90, le=0.99)] = 0.95
    ttl: Annotated[int, Field(ge=1)] = 3600
    backend: Literal["memory", "disk", "redis"] = "memory"
    max_entries: Annotated[int, Field(ge=1)] = 10_000
    redis_url: str = DEFAULT_REDIS_URL

    @field_validator("redis_url")
    @classmethod
    def _check_redis_url(cls, value: str) -> str:
        """Require a parseable Redis URL; only read when the backend is ``redis``."""
        return _validate_redis_url(value)


class WorkersSettings(Section):
    """Pipeline worker pools and the bounded queue between them."""

    cpu_pool_size: Annotated[int, Field(ge=0)] = 0
    embedding_pool_size: Annotated[int, Field(ge=1)] = 1
    queue_depth: Annotated[int, Field(ge=10)] = 1000


class JournalSettings(Section):
    """Checkpointed ingestion journal enabling exact crash resume."""

    enabled: bool = True
    checkpoint_every: Annotated[int, Field(ge=1)] = 100
    retention_days: Annotated[int, Field(ge=0, le=36500)] = 0


class DlqSettings(Section):
    """Dead-letter queue for documents that fail after their retries."""

    enabled: bool = True
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3


class IngestionSettings(Section):
    """Deduplication, journaling, dead-lettering, and input limits."""

    dedup: bool = True
    journal: JournalSettings = JournalSettings()
    dlq: DlqSettings = DlqSettings()
    max_document_mb: Annotated[int, Field(ge=1, le=1024)] = 100
    parse_timeout_seconds: Annotated[float, Field(ge=0)] = 600

    # CRITICAL: this is an authorization boundary, not a convenience. A `path` source names
    # a file on the SERVER, so without confinement any caller holding the `ingest` scope can
    # ingest `.env`, `config.yaml` or an SSH key and read it back out with an ordinary query.
    # Empty means "unconfined", which is safe only when nobody but the operator can reach the
    # API — so `resolve_sources` refuses path sources outright when the list is empty AND
    # `security.auth` is on. Leaving it empty therefore cannot silently expose a served
    # deployment; it can only break one loudly, at the request, naming this setting.
    allowed_source_roots: list[str] = []


class ReindexSettings(Section):
    """Zero-downtime reindexing strategy and rollback retention."""

    strategy: Literal["blue_green", "in_place"] = "blue_green"
    eval_gate: bool = True
    rollback_retention_hours: Annotated[int, Field(ge=0)] = 72


class IndexSettings(Section):
    """Index lockfile and reindexing policy."""

    lockfile: bool = True
    reindex: ReindexSettings = ReindexSettings()


class TimeoutSettings(Section):
    """Explicit timeouts; no external call is ever unbounded."""

    vector_db_ms: Annotated[int, Field(ge=100)] = 5000
    embeddings_ms: Annotated[int, Field(ge=100)] = 30000
    llm_ms: Annotated[int, Field(ge=1000)] = 120000
    redis_ms: Annotated[int, Field(ge=10)] = 1000


class RetrySettings(Section):
    """Bounded retries with exponential backoff, applied only to retryable errors."""

    max_attempts: Annotated[int, Field(ge=0, le=10)] = 3
    backoff_base_ms: Annotated[int, Field(ge=1)] = 250
    backoff_max_ms: Annotated[int, Field(ge=1)] = 10000
    jitter: bool = True

    @model_validator(mode="after")
    def _check_backoff_bounds(self) -> Self:
        """Require the backoff ceiling to be at least the backoff base."""
        if self.backoff_max_ms < self.backoff_base_ms:
            raise ValueError(
                f"reliability.retries.backoff_max_ms ({self.backoff_max_ms}) must be greater "
                f"than or equal to backoff_base_ms ({self.backoff_base_ms})"
            )
        return self


class CircuitBreakerSettings(Section):
    """Per-provider circuit breakers whose state is exported as a metric."""

    enabled: bool = True
    failure_threshold: Annotated[int, Field(ge=1)] = 5
    reset_timeout_ms: Annotated[int, Field(ge=1000)] = 30000


class IdempotencySettings(Section):
    """``Idempotency-Key`` record lifetime and the switch that turns the feature off."""

    # CRITICAL: `enabled` defaults to **true**, deviating from the default-false rule for
    # risky features, and the deviation is the decision rather than an oversight. The rule
    # exists for features that start doing something — provisioning writes to external
    # systems, the dashboard opens a port. This one starts nothing: a request without the
    # header touches no store and writes no byte. Defaulting it off would instead make a
    # published contract false, since `docs/api-reference.md` tells clients the header is
    # honoured, and a client whose retry silently executed twice has no way to notice. The
    # flag is a kill switch for a store that is misbehaving, and a kill switch that ships
    # already pulled is not one. Turning it off is a deliberate, documented downgrade.
    enabled: bool = True
    ttl_seconds: Annotated[int, Field(ge=1, le=7 * 24 * 60 * 60)] = 24 * 60 * 60
    in_flight_lease_seconds: Annotated[int, Field(ge=1)] = 15 * 60

    @model_validator(mode="after")
    def _check_lease_fits_the_lifetime(self) -> Self:
        """Require the in-flight lease to be no longer than the record lifetime."""
        if self.in_flight_lease_seconds > self.ttl_seconds:
            raise ValueError(
                f"reliability.idempotency.in_flight_lease_seconds "
                f"({self.in_flight_lease_seconds}) must be less than or equal to "
                f"ttl_seconds ({self.ttl_seconds}); an unfinished request would otherwise "
                "hold its key longer than a completed one can be replayed"
            )
        return self


class ReliabilitySettings(Section):
    """Timeouts, retries, circuit breakers, idempotency, and the degradation ladder."""

    timeouts: TimeoutSettings = TimeoutSettings()
    retries: RetrySettings = RetrySettings()
    circuit_breaker: CircuitBreakerSettings = CircuitBreakerSettings()
    idempotency: IdempotencySettings = IdempotencySettings()
    degradation_ladder: bool = True
    shutdown_grace_seconds: Annotated[int, Field(ge=0, le=3600)] = 20
    stale_job_seconds: Annotated[int, Field(ge=1, le=86400)] = 120


class TracesSettings(Section):
    """Local trace persistence backing time-travel replay."""

    store: bool = True
    retention_days: Annotated[int, Field(ge=1)] = 30
    replay: bool = True


class CostSettings(Section):
    """Preflight estimation and runtime token budgets; 0 means unlimited."""

    estimator: bool = True
    per_query_token_budget: Annotated[int, Field(ge=0)] = 0
    per_tenant_token_budget: Annotated[int, Field(ge=0)] = 0


class AutopilotSettings(Section):
    """Eval-driven tuning that only ever suggests; it never writes ``config.yaml``."""

    enabled: bool = False
    golden_set_size: Annotated[int, Field(ge=10, le=10000)] = 100


class EvalSettings(Section):
    """Retrieval regression gate and its tolerances."""

    regression_gate: bool = False
    recall_tolerance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02
    ndcg_tolerance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02


class ObservabilitySettings(Section):
    """Read-only inspection surfaces; every integration toggle defaults to false."""

    dashboard: bool = False
    dashboard_port: Annotated[int, Field(ge=1, le=65535)] = 8080
    otel: bool = False
    otel_endpoint: str | None = None
    langfuse: bool = False
    grafana: bool = False

    @field_validator("otel_endpoint")
    @classmethod
    def _check_endpoint(cls, value: str | None) -> str | None:
        """Require an absolute URL when a collector endpoint is set."""
        if value is None:
            return None
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"{value!r} must be an absolute http:// or https:// URL")
        return value

    @model_validator(mode="after")
    def _check_otel_endpoint(self) -> Self:
        """Enforce cross-field rule 6: enabling OTel requires a collector endpoint."""
        if self.otel and self.otel_endpoint is None:
            raise ValueError(
                "observability.otel_endpoint is required when observability.otel is true"
            )
        return self


class SecuritySettings(Section):
    """Control-plane authentication, tenancy, and request limits."""

    auth: bool = False
    api_key_env: str = "FASTERRAG_API_KEY"
    multi_tenancy: bool = False
    tenant_header: str = "X-Tenant-ID"

    # CRITICAL: off by default because turning it on changes what an already-ingested
    # corpus answers. Chunks written before `multi_tenancy` was enabled carry no `tenant`
    # payload key at all, and an absent key never matches a filter, so enabling this over
    # unbackfilled data returns empty results rather than wrong ones. Default-off keeps
    # every existing deployment on exactly the behaviour it has today (TASK-0339).
    tenant_scoped_retrieval: bool = False
    rate_limit_per_minute: Annotated[int, Field(ge=1)] = 600
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    rate_limit_redis_url: str = DEFAULT_REDIS_URL
    max_request_mb: Annotated[int, Field(ge=1, le=1024)] = 25

    # CRITICAL: this is an authorization boundary, not a convenience. `POST /v1/admin/export`
    # names the file it writes and `POST /v1/admin/import` names the file it reads, both from
    # the request body, so without confinement an `admin` key writes anywhere the process can
    # write — over `config.yaml`, into a web root — and reads any archive on the host. It
    # lives beside `auth` rather than under a feature section because it bounds what a
    # credential may reach on the host, which is what the rest of this section decides.
    # Empty means "unconfined", which `guard_archive_path` refuses outright once `auth` is
    # on, so leaving it empty breaks a served deployment loudly instead of exposing it.
    allowed_archive_roots: list[str] = []

    @field_validator("rate_limit_redis_url")
    @classmethod
    def _check_rate_limit_redis_url(cls, value: str) -> str:
        """Require a parseable Redis URL; only read when the backend is ``redis``."""
        return _validate_redis_url(value)

    @field_validator("api_key_env")
    @classmethod
    def _check_api_key_env(cls, value: str) -> str:
        """Require an environment-variable name, never a key value."""
        validated = _validate_env_var_name(value)
        if validated is None:
            raise ValueError("security.api_key_env must name an environment variable")
        return validated

    @field_validator("tenant_header")
    @classmethod
    def _check_tenant_header(cls, value: str) -> str:
        """Require a syntactically valid HTTP header name."""
        if not _HTTP_HEADER_RE.match(value):
            raise ValueError(f"{value!r} is not a valid HTTP header name")
        return value

    @model_validator(mode="after")
    def _check_tenant_scoped_retrieval(self) -> Self:
        """Refuse retrieval scoping without the tenancy that supplies the tenant."""
        if self.tenant_scoped_retrieval and not (self.auth and self.multi_tenancy):
            raise ValueError(
                "security.tenant_scoped_retrieval requires security.auth and "
                "security.multi_tenancy, because the tenant it filters on is resolved by "
                "the authentication middleware; without them every retrieval is refused"
            )
        return self


class Settings(BaseSettings):
    """The validated whole of ``config.yaml``."""

    model_config = SettingsConfigDict(extra="forbid", frozen=True)

    app: AppSettings = AppSettings()
    vector_db: VectorDbSettings = VectorDbSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    llm: LlmSettings = LlmSettings()
    parsing: ParsingSettings = ParsingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    cache: CacheSettings = CacheSettings()
    workers: WorkersSettings = WorkersSettings()
    ingestion: IngestionSettings = IngestionSettings()
    index: IndexSettings = IndexSettings()
    reliability: ReliabilitySettings = ReliabilitySettings()
    traces: TracesSettings = TracesSettings()
    cost: CostSettings = CostSettings()
    autopilot: AutopilotSettings = AutopilotSettings()
    eval: EvalSettings = EvalSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    security: SecuritySettings = SecuritySettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Restrict configuration to the values passed in from the YAML source."""
        return (init_settings,)

    def referenced_env_vars(self) -> dict[str, str]:
        """Return every environment variable this configuration references.

        Keys are variable names, values are the dotted config paths that reference them,
        so a missing variable can be reported against the key that needs it.
        """
        found: dict[str, str] = {}
        self._collect_env_vars(self, (), found)
        return found

    def required_env_vars(self) -> dict[str, str]:
        """Return the referenced variables that a feature this configuration enables reads.

        Cross-field rule 9 requires these, and only these, at startup (TASK-0444).
        """
        found: dict[str, str] = {}
        self._collect_env_vars(self, (), found, skip=self._unread_env_keys())
        return found

    # CRITICAL: each exemption names a key that NOTHING reads under the stated condition —
    # never a key that is read later or optionally. `security.api_key_env` is read only by
    # the auth middleware; `vector_db.api_key_env` by every built-in vector database except
    # Chroma and pgvector, and by any plugin; `dsn_env` only by pgvector; the two provider
    # keys only by the providers outside the keyless sets. The LLM key has no condition of its
    # own: no setting disables generation, so a keyed provider is always in use. Widening this
    # list moves a missing secret from a startup failure to a failure mid-request.
    def _unread_env_keys(self) -> frozenset[str]:
        """Return the dotted ``*_env`` keys that no enabled feature reads."""
        unread: set[str] = set()
        if not self.security.auth:
            unread.add("security.api_key_env")
        if self.vector_db.provider in VECTOR_DB_PROVIDERS_WITHOUT_KEY:
            unread.add("vector_db.api_key_env")
        if self.vector_db.provider != "pgvector":
            unread.add("vector_db.pgvector.dsn_env")
        if self.embeddings.provider in EMBEDDING_PROVIDERS_WITHOUT_KEY:
            unread.add("embeddings.api_key_env")
        if self.llm.provider in PROVIDERS_NOT_REQUIRING_LLM_KEY:
            unread.add("llm.api_key_env")
        return frozenset(unread)

    @classmethod
    def _collect_env_vars(
        cls,
        model: BaseModel,
        path: tuple[str, ...],
        found: dict[str, str],
        skip: frozenset[str] = frozenset(),
    ) -> None:
        """Walk nested sections, recording every populated ``*_env`` field not in ``skip``."""
        for name in type(model).model_fields:
            value = getattr(model, name)
            current = (*path, name)
            if isinstance(value, BaseModel):
                cls._collect_env_vars(value, current, found, skip)
            elif name.endswith("_env") and isinstance(value, str):
                dotted = ".".join(current)
                if dotted not in skip:
                    found.setdefault(value, dotted)
