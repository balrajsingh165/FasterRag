"""The ``FasterRag`` facade: fasterRag as a library rather than a service.

The third control surface. REST, CLI, and this all call the same service layer, so behavior,
configuration, and errors are identical whichever one an application uses
(``docs/python-api.md``).

This is a thin composition layer and nothing else. It owns no retrieval, ingestion, or
generation logic — it builds the same services ``api/dependencies.py`` builds, from the same
validated ``Settings``, and hands them the same arguments. Any behavior that lived here would
be behavior the REST API does not have, which is exactly the divergence the shared service
layer exists to prevent.

**Resource-heavy objects are process-scoped, not per call.** The embedding router and the
semantic cache are built once when the facade starts and reused for every query. Building
either per call is not a small inefficiency: a local embedding model takes seconds to load,
and a cache constructed per call starts empty every time, so it can never hit while still
costing a query embedding. Both were real, measured regressions in the REST path.

The facade is an async context manager because those resources need closing. Using it without
``async with`` is a leak, so every entry point checks and says so rather than failing later
somewhere unrelated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from fasterrag.adapters.embeddings.tiering import TieringRouter, create_embedding_router
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.config.loader import DEFAULT_CONFIG_PATH, load_settings
from fasterrag.config.schema import Settings
from fasterrag.core.cache import create_semantic_store
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.estimation import Estimate, estimate_sources
from fasterrag.services.generation import Answer, GenerationService, QueryEvent
from fasterrag.services.journal import JobRecord, Journal, create_journal
from fasterrag.services.lockfile import IndexLock, create_lock_store
from fasterrag.services.querying import RetrievalService

__all__ = ["FasterRag"]

_logger = get_logger(__name__)


class FasterRag:
    """An embedded fasterRag instance.

    Construct with :meth:`from_config` or :meth:`from_settings`, then enter it as an async
    context manager::

        async with FasterRag.from_config("config.yaml") as rag:
            await rag.ingest(["./docs"])
            answer = await rag.query("What does the spec say about retries?")

    Construction validates configuration and builds nothing else; entering the context is
    what connects adapters and loads models. That split is deliberate — it means a
    configuration mistake is reported before any model is downloaded or any socket opened.
    """

    def __init__(self, settings: Settings) -> None:
        """Build a facade over validated settings without touching any backend.

        Prefer :meth:`from_config` or :meth:`from_settings`; this constructor is public only
        so the two classmethods have something to return.
        """
        self.settings = settings
        self._started = False
        self._vector_db: VectorDBAdapter | None = None
        self._embeddings: TieringRouter | None = None
        self._cache: SemanticCache | None = None
        self._journal: Journal | None = None
        self._generation: GenerationService | None = None

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Self:
        """Load and validate ``config.yaml``, then build a facade over it.

        Args:
            path: Path to the configuration file.

        Returns:
            An unstarted facade. No backend has been contacted.

        Raises:
            ConfigError: If the file is missing, the YAML is malformed, a key violates the
                schema, or a referenced environment variable is unset — the same fail-fast
                contract the API and CLI get, naming the offending key.
        """
        return cls(load_settings(path))

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build a facade over an already-validated ``Settings``.

        For applications that manage their own configuration and never keep a
        ``config.yaml`` on disk.
        """
        return cls(settings)

    async def __aenter__(self) -> Self:
        """Connect the adapter and build the process-scoped resources."""
        self._vector_db = create_vector_db_adapter(self.settings)
        self._embeddings = create_embedding_router(self.settings)
        self._cache = SemanticCache(self.settings, create_semantic_store(self.settings))
        self._journal = create_journal(self.settings)
        self._started = True

        _logger.info(
            "facade started",
            extra={
                "vector_db": self.settings.vector_db.provider,
                "embeddings": self.settings.embeddings.provider,
                "llm": self.settings.llm.provider,
            },
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release every resource, whether or not the body raised.

        Each close is attempted independently: one backend refusing to shut down cleanly
        must not leave the others open, which is what a single sequential chain would do.
        """
        self._started = False

        if self._generation is not None:
            await self._close_quietly(self._generation.close(), "generation")
            self._generation = None
        if self._cache is not None:
            await self._close_quietly(self._cache.close(), "cache")
            self._cache = None
        if self._embeddings is not None:
            await self._close_quietly(self._embeddings.close(), "embeddings")
            self._embeddings = None
        if self._vector_db is not None:
            await self._close_quietly(self._vector_db.close(), "vector_db")
            self._vector_db = None

        _logger.info("facade stopped")

    @staticmethod
    async def _close_quietly(closing: Any, name: str) -> None:
        """Await one close, logging a failure rather than aborting the rest of shutdown.

        # CRITICAL: this catches broadly on purpose, and it is the one place in the codebase
        # that does. Shutdown runs while an exception may already be propagating; letting a
        # second one escape from here would replace the original failure with a teardown
        # error and lose the cause. Every failure is logged with the resource that raised it.
        """
        try:
            await closing
        except Exception as exc:
            _logger.warning(
                "resource did not close cleanly",
                extra={"resource": name, "error": str(exc)},
            )

    def _require_started(self) -> None:
        """Fail with a usable message when the facade was never entered.

        Raises:
            FasterRagError: If used outside ``async with``. Reported here rather than left
                to surface as an ``AttributeError`` on a ``None`` adapter three frames away.
        """
        if not self._started:
            raise FasterRagError(
                "this FasterRag instance has not been started; use it as an async context "
                "manager: 'async with FasterRag.from_config(...) as rag:'",
                code=ErrorCode.CONFIG_INVALID,
            )

    @property
    def vector_db(self) -> VectorDBAdapter:
        """Return the live vector database adapter."""
        self._require_started()
        assert self._vector_db is not None
        return self._vector_db

    @property
    def embeddings(self) -> TieringRouter:
        """Return the process-scoped embedding router."""
        self._require_started()
        assert self._embeddings is not None
        return self._embeddings

    @property
    def cache(self) -> SemanticCache:
        """Return the process-scoped semantic cache."""
        self._require_started()
        assert self._cache is not None
        return self._cache

    @property
    def journal(self) -> Journal:
        """Return the ingestion journal."""
        self._require_started()
        assert self._journal is not None
        return self._journal

    def _retrieval(self) -> RetrievalService:
        """Build the retrieval service exactly as the API's dependency does."""
        from fasterrag.api.dependencies import build_retrieval

        return build_retrieval(self.settings, self.vector_db, self.embeddings)

    def _generation_service(self) -> GenerationService:
        """Return the generation service, built once and reused.

        Reused rather than rebuilt because it owns the LLM adapter's client, and a new one
        per query opens a connection pool per query.
        """
        from fasterrag.api.dependencies import build_generation

        if self._generation is None:
            self._generation = build_generation(
                self.settings, self.vector_db, self.embeddings, cache=self.cache
            )
        return self._generation

    async def ingest(
        self,
        sources: Sequence[str],
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Ingest sources, returning the settled job record.

        Identical semantics to ``POST /v1/ingest``: journaled, deduplicated, checkpointed,
        and dead-lettering what it cannot process. Unlike the REST endpoint this *awaits*
        completion rather than returning a job id, because a library caller already has the
        one thing an HTTP client lacks — somewhere to wait.

        Returns:
            The settled job, carrying its status and per-stage counts.
        """
        from fasterrag.api.dependencies import build_ingestion

        self._require_started()
        service = build_ingestion(
            self.settings, self.vector_db, self.journal, self.cache, router=self.embeddings
        )
        return await service.ingest(
            sources,
            collection=collection,
            metadata=metadata,
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    async def query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Answer:
        """Answer a question through the full pipeline.

        Returns:
            The answer with its citations, timings, and cache status. When
            ``generation.grounded_or_refuse`` is on and the answer scores below the
            threshold, ``answer`` is ``None`` and ``best_candidates`` carries the retrieved
            evidence instead — a refusal is a result, not an error (D5).
        """
        self._require_started()
        return await self._generation_service().answer(
            text, collection=collection, top_k=top_k, filters=filters
        )

    async def query_stream(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[QueryEvent]:
        """Answer a question as a stream of typed events.

        Yields ``meta``, then ``token`` events, then ``citations``, ``usage``, and ``done``,
        mirroring the SSE contract. A missing ``done`` means the answer is incomplete and
        must be treated as such — the absence carries meaning and is never papered over.
        """
        self._require_started()
        async for event in self._generation_service().stream(
            text, collection=collection, top_k=top_k, filters=filters
        ):
            yield event

    async def retrieve(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve without generating, for applications bringing their own LLM step.

        Returns:
            Chunks ordered best first, each carrying the rank and score every retrieval leg
            gave it.
        """
        self._require_started()
        return await self._retrieval().retrieve(
            text, collection=collection, top_k=top_k, filters=filters
        )

    def estimate(self, sources: Sequence[str], *, all_providers: bool = False) -> Estimate:
        """Report what ingesting ``sources`` would cost, before embedding any of it (D9).

        Synchronous and safe to call before starting: it parses and chunks for real but
        never contacts a backend, which is the whole point of a preflight estimate.
        """
        return estimate_sources(sources, self.settings, all_providers=all_providers)

    def index_lock(self, collection: str | None = None) -> IndexLock | None:
        """Return a collection's index lockfile, or ``None`` if there is none (D1).

        Returns:
            The lock recording what built the index — embedding model, dimensions, chunker
            settings, corpus hashes — or ``None`` when ``index.lockfile`` is off or nothing
            has been indexed into the collection yet.
        """
        store = create_lock_store(self.settings)
        if store is None or not store.enabled:
            return None
        return store.read(collection or self.settings.vector_db.collection.default_name)
