"""The ``FasterRag`` facade: fasterRag as a library rather than a service.

The third control surface. REST, CLI, and this all call the same service layer, so behavior,
configuration, and errors are identical whichever one an application uses
(``docs/python-api.md``).

This is a thin composition layer and nothing else. It owns no retrieval, ingestion, or
generation logic — it builds its services with the same ``services/composition.py`` builders
the REST API uses, from the same validated ``Settings``, and hands them the same arguments.
It imports nothing from ``fasterrag.api``, so a library install never loads the web
framework. Any behavior that lived here would be behavior the REST API does not have, which
is exactly the divergence the shared service layer exists to prevent.

**Resource-heavy objects are process-scoped, not per call.** The embedding router and the
semantic cache are built once when the facade starts and reused for every query. Building
either per call is not a small inefficiency: a local embedding model takes seconds to load,
and a cache constructed per call starts empty every time, so it can never hit while still
costing a query embedding. Both were real, measured regressions in the REST path.

**The LLM adapter is shared; the generation service is not** (TASK-0461). The service carries
the caller's tenant and the candidates of the query in flight, so one cached instance would
answer one tenant with another's state. Each query therefore builds its own service, as a
REST request does, and borrows the one provider client this facade owns and closes.

The facade is an async context manager because those resources need closing. Using it without
``async with`` is a leak, so every entry point checks and says so rather than failing later
somewhere unrelated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.adapters.vectordb.base import CollectionInfo, VectorDBAdapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.config.loader import DEFAULT_CONFIG_PATH, load_settings
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.composition import (
    build_documents,
    build_generation,
    build_ingestion,
    build_retrieval,
    create_embedding_router,
    create_llm_adapter,
    create_semantic_cache,
)
from fasterrag.services.doctor import DoctorReport
from fasterrag.services.documents import DEFAULT_PAGE_SIZE, DocumentDeletion, DocumentListing
from fasterrag.services.estimation import Estimate, estimate_sources, require_estimator
from fasterrag.services.generation import Answer, GenerationService, QueryEvent
from fasterrag.services.journal import JobRecord, Journal, create_journal
from fasterrag.services.lockfile import IndexLock, create_lock_store
from fasterrag.services.querying import RetrievalService
from fasterrag.services.replay import ReplayResult

__all__ = ["FasterRag"]

_logger = get_logger(__name__)


class FasterRag:
    """An embedded fasterRag instance."""

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
        self._llm: LLMAdapter | None = None

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Self:
        """Load and validate ``config.yaml``, then build a facade over it."""
        return cls(load_settings(path, export_env=True))

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
        self._cache = create_semantic_cache(self.settings)
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

        if self._llm is not None:
            await self._close_quietly(self._llm.close(), "llm")
            self._llm = None
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
        """Await one close, logging a failure rather than aborting the rest of shutdown."""
        try:
            await closing
        except Exception as exc:
            _logger.warning(
                "resource did not close cleanly",
                extra={"resource": name, "error": str(exc)},
            )

    def _require_started(self) -> None:
        """Fail with a usable message when the facade was never entered."""
        if not self._started:
            raise FasterRagError(
                "this FasterRag instance has not been started; use it as an async context "
                "manager: 'async with FasterRag.from_config(...) as rag:'",
                code=ErrorCode.USAGE_ERROR,
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

    def _retrieval(self, tenant: str | None = None) -> RetrievalService:
        """Build the retrieval service exactly as the API does, on behalf of ``tenant``."""
        return build_retrieval(self.settings, self.vector_db, self.embeddings, tenant)

    def _shared_llm(self) -> LLMAdapter:
        """Return this instance's LLM adapter, built on first use and closed on exit.

        Shared rather than built per query, because a new client per query opens a
        connection pool per query (TASK-0466).
        """
        self._require_started()
        if self._llm is None:
            self._llm = create_llm_adapter(self.settings)
        return self._llm

    def _generation_service(self, tenant: str | None = None) -> GenerationService:
        """Build one query's generation service for ``tenant``, lending it the shared LLM.

        Per call, never cached: the service holds its tenant and its query's candidates.
        """
        return build_generation(
            self.settings,
            self.vector_db,
            self.embeddings,
            cache=self.cache,
            tenant=tenant,
            llm=self._shared_llm(),
        )

    async def ingest(
        self,
        sources: Sequence[str],
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
        recursive: bool = False,
    ) -> JobRecord:
        """Ingest sources, returning the settled job record.

        A directory is replaced by the files inside it, as ``fasterrag ingest`` does;
        ``recursive`` descends into subdirectories.
        """
        from fasterrag.cli.sources import expand_sources

        self._require_started()
        expanded = expand_sources(sources, recursive=recursive)
        if not expanded:
            raise FasterRagError(
                "there is nothing to ingest: every source was a directory holding no "
                "visible files. Pass recursive=True to descend into subdirectories",
                code=ErrorCode.VALIDATION_FAILED,
            )
        service = build_ingestion(
            self.settings, self.vector_db, self.journal, self.cache, router=self.embeddings
        )
        return await service.ingest(
            expanded,
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
        tenant: str | None = None,
    ) -> Answer:
        """Answer a question through the full pipeline, on behalf of ``tenant`` if given."""
        self._require_started()
        service = self._generation_service(tenant)
        try:
            return await service.answer(text, collection=collection, top_k=top_k, filters=filters)
        finally:
            await service.close()

    async def query_stream(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        tenant: str | None = None,
    ) -> AsyncIterator[QueryEvent]:
        """Answer a question as a stream of typed events, on behalf of ``tenant`` if given."""
        self._require_started()
        service = self._generation_service(tenant)
        try:
            async for event in service.stream(
                text, collection=collection, top_k=top_k, filters=filters
            ):
                yield event
        finally:
            await service.close()

    async def retrieve(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        tenant: str | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve without generating, for applications bringing their own LLM step."""
        self._require_started()
        return await self._retrieval(tenant).retrieve(
            text, collection=collection, top_k=top_k, filters=filters
        )

    async def doctor(self) -> DoctorReport:
        """Run the D10 preflight checks against this instance's configuration."""
        from fasterrag.services.doctor import run_doctor

        return await run_doctor(self.settings)

    async def collections(self) -> list[CollectionInfo]:
        """List the collections the vector database holds."""
        self._require_started()
        return await self.vector_db.list_collections()

    async def create_collection(
        self,
        name: str,
        *,
        distance: str | None = None,
        shard_number: int = 1,
        replication_factor: int = 1,
    ) -> None:
        """Create a collection sized for the configured embedding model."""
        from fasterrag.adapters.vectordb.base import CollectionSpec

        self._require_started()
        dimensions = self.embeddings.default.dimensions or self.settings.embeddings.dimensions
        if not dimensions:
            raise FasterRagError(
                "the embedding model did not report a dimension and embeddings.dimensions is "
                "unset, so a collection cannot be sized; set embeddings.dimensions",
                code=ErrorCode.CONFIG_INVALID,
            )

        await self.vector_db.create_collection(
            CollectionSpec(
                name=name,
                dimensions=dimensions,
                distance=cast("Any", distance or self.settings.vector_db.collection.distance),
                shard_number=shard_number,
                replication_factor=replication_factor,
                sparse=self.settings.retrieval.hybrid,
            )
        )

    async def drop_collection(self, name: str) -> bool:
        """Drop a collection, returning whether it existed."""
        self._require_started()
        return await self.vector_db.drop_collection(name)

    async def delete_document(
        self,
        document_id: str,
        *,
        collection: str | None = None,
        tenant: str | None = None,
    ) -> DocumentDeletion:
        """Delete one document's chunks and every record that it was indexed.

        Idempotent: ``known`` on the result says whether the journal held the document.
        With ``tenant``, only chunks tagged for that tenant are removed.
        """
        self._require_started()
        service = build_documents(self.settings, self.vector_db, self.journal, self.cache)
        return await service.delete(
            collection or self.settings.vector_db.collection.default_name,
            document_id,
            tenant=tenant,
        )

    async def documents(
        self,
        collection: str | None = None,
        *,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> DocumentListing:
        """Return one page of the documents the journal records in a collection.

        Ordered by ``document_id``; a deleted document is not listed (TASK-0469).
        """
        self._require_started()
        service = build_documents(self.settings, self.vector_db, self.journal, self.cache)
        return await service.list_documents(
            collection or self.settings.vector_db.collection.default_name,
            offset=offset,
            limit=limit,
        )

    async def replay(self, trace_id: str, candidate: Settings | None = None) -> ReplayResult:
        """Re-execute a stored query under a candidate configuration and diff it (D8)."""
        from fasterrag.services.replay import replay_trace
        from fasterrag.services.traces import create_trace_store

        self._require_started()
        trace = create_trace_store(self.settings).load(trace_id)
        if trace is None:
            raise FasterRagError(
                f"no stored trace with id {trace_id!r}; list recent ones with "
                "'fasterrag traces list'",
                code=ErrorCode.NOT_FOUND,
            )

        settings = candidate or self.settings
        service = build_generation(settings, self.vector_db, self.embeddings)
        try:
            return await replay_trace(trace, settings, service)
        finally:
            await service.close()

    def estimate(self, sources: Sequence[str], *, all_providers: bool = False) -> Estimate:
        """Report what ingesting ``sources`` would cost, before embedding any of it (D9)."""
        require_estimator(self.settings)
        return estimate_sources(sources, self.settings, all_providers=all_providers)

    def index_lock(self, collection: str | None = None) -> IndexLock | None:
        """Return a collection's index lockfile, or ``None`` if there is none (D1)."""
        store = create_lock_store(self.settings)
        if store is None or not store.enabled:
            return None
        return store.read(collection or self.settings.vector_db.collection.default_name)
