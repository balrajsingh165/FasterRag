"""Shared request-scoped construction for the routers.

Services are built per request from ``app.state``, not stored on it, with one exception:
the vector database adapter is long-lived because it holds a connection pool, and rebuilding
it per request would open and close sockets on every call.

Everything else here exists so a router never constructs a service inline. Routers hold zero
business logic (``docs/structure.md``), which in practice means they receive a ready service
and return whatever it produced.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from fasterrag.adapters.embeddings.tiering import TieringRouter, create_embedding_router
from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.cache import create_semantic_store
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.rerank import CrossEncoderReranker
from fasterrag.services.generation import GenerationService
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal, create_journal
from fasterrag.services.querying import RetrievalService
from fasterrag.services.traces import TraceStore

__all__ = [
    "CurrentCache",
    "CurrentEmbeddings",
    "CurrentSettings",
    "CurrentVectorDB",
    "JournalDep",
    "get_journal",
    "get_settings",
    "get_vector_db",
    "shared_cache",
    "shared_embedding_router",
]


def get_settings(request: Request) -> Settings:
    """Return the validated settings the application was built with."""
    settings: Settings = request.app.state.settings
    return settings


def get_vector_db(request: Request) -> VectorDBAdapter:
    """Return the application's long-lived vector database adapter."""
    adapter: VectorDBAdapter = request.app.state.vector_db
    return adapter


def get_journal(request: Request) -> Journal:
    """Return the ingestion journal, built once per application."""
    journal: Journal | None = getattr(request.app.state, "journal", None)
    if journal is None:
        journal = create_journal(request.app.state.settings)
        request.app.state.journal = journal
    return journal


def build_embedding_router(settings: Settings) -> TieringRouter:
    """Build a fresh embedding router.

    Prefer :func:`shared_embedding_router` inside a request. This exists for the background
    ingest path, whose work outlives the request that started it.
    """
    return create_embedding_router(settings)


def shared_embedding_router(request: Request) -> TieringRouter:
    """Return the application's long-lived embedding router.

    # CRITICAL: built once per process, never per request. A local embedding model takes
    # seconds to load, and building a router per request reloads it every time — measured at
    # roughly five seconds added to every query, against forty milliseconds of retrieval.
    # The router is closed by the application lifespan, never by a request handler.
    """
    router: TieringRouter | None = getattr(request.app.state, "embeddings", None)
    if router is None:
        router = create_embedding_router(request.app.state.settings)
        request.app.state.embeddings = router
    return router


def shared_cache(request: Request) -> SemanticCache:
    """Return the application's long-lived semantic response cache.

    # CRITICAL: process-scoped, like the router. A cache built per request starts empty on
    # every request, so it can never hit — it would cost a query embedding per call and
    # return nothing, which is strictly worse than having no cache at all.
    """
    cache: SemanticCache | None = getattr(request.app.state, "cache", None)
    if cache is None:
        settings = request.app.state.settings
        cache = SemanticCache(settings, create_semantic_store(settings))
        request.app.state.cache = cache
    return cache


CurrentSettings = Annotated[Settings, Depends(get_settings)]
CurrentVectorDB = Annotated[VectorDBAdapter, Depends(get_vector_db)]
CurrentEmbeddings = Annotated[TieringRouter, Depends(shared_embedding_router)]
CurrentCache = Annotated[SemanticCache, Depends(shared_cache)]
JournalDep = Annotated[Journal, Depends(get_journal)]


def build_retrieval(
    settings: Settings, adapter: VectorDBAdapter, router: TieringRouter
) -> RetrievalService:
    """Assemble hybrid retrieval, with reranking only when configuration asks for it."""
    reranker = CrossEncoderReranker(settings) if settings.retrieval.rerank else None
    return RetrievalService(settings, adapter, router, reranker)


def build_generation(
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    cache: SemanticCache | None = None,
    traces: TraceStore | None = None,
) -> GenerationService:
    """Assemble the query path the CLI also assembles, so the two cannot diverge.

    ``cache`` and ``traces`` are used exactly as given, including ``None``. Replay depends
    on that: it must run with neither, and a helper that quietly substituted a default would
    have it populating the cache and writing traces while investigating one.
    """
    return GenerationService(
        settings,
        build_retrieval(settings, adapter, router),
        create_llm_adapter(settings),
        cache=cache,
        traces=traces,
        embedder=router.default,
    )


def build_ingestion(
    settings: Settings,
    adapter: VectorDBAdapter,
    journal: Journal,
    cache: SemanticCache | None,
    *,
    router: TieringRouter | None = None,
) -> IngestionService:
    """Assemble ingestion against the application's adapter.

    The adapter is passed in rather than built, so a background job shares the connection
    pool the request that started it was using instead of opening a second one.
    """
    return IngestionService(settings, journal=journal, adapter=adapter, router=router, cache=cache)
