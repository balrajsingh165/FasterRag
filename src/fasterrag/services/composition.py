"""Composition root: where an entry point gets the pieces it runs on.

The API, the CLI, the facade and the evaluation harness all assemble the same services, and
they must all assemble them here, or a behaviour added to one assembly reaches some entry
points and not others. That is what happened to the embedding cache (TASK-0442), and it is
why the builders left ``api/dependencies.py``: while they lived in the REST layer, the
facade and ``services/evaluation.py`` reached up into it through function-level imports, so
a library user who never starts a server still needed the web framework installed
(TASK-0456). This module imports nothing from ``fasterrag.api``; ``api/dependencies.py``
keeps only the FastAPI wrappers that read ``app.state``.

What is built here per call is cheap: a service object over process-scoped parts. The parts
that hold connections or models — the vector database adapter, the embedding router, the
semantic cache and the LLM adapter — are built once by the entry point and passed in.
"""

from __future__ import annotations

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.cache import create_semantic_store
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.rerank import CrossEncoderReranker
from fasterrag.services.documents import DocumentService
from fasterrag.services.embedding_router import create_embedding_router
from fasterrag.services.generation import GenerationService
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from fasterrag.services.lockfile import create_lock_store
from fasterrag.services.querying import RetrievalService
from fasterrag.services.traces import TraceStore

__all__ = [
    "build_documents",
    "build_generation",
    "build_ingestion",
    "build_retrieval",
    "create_embedding_router",
    "create_llm_adapter",
    "create_semantic_cache",
]


def create_semantic_cache(settings: Settings) -> SemanticCache:
    """Build the semantic response cache over the configured store."""
    return SemanticCache(settings, create_semantic_store(settings))


def build_retrieval(
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    tenant: str | None = None,
) -> RetrievalService:
    """Assemble hybrid retrieval, with reranking only when configuration asks for it.

    The tenant is passed in rather than read here, so the one middleware that decided who
    the caller is stays the only thing that decides it (``security.tenant_scoped_retrieval``).
    """
    reranker = CrossEncoderReranker(settings) if settings.retrieval.rerank else None
    return RetrievalService(settings, adapter, router, reranker, tenant=tenant)


def build_generation(
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    cache: SemanticCache | None = None,
    traces: TraceStore | None = None,
    tenant: str | None = None,
) -> GenerationService:
    """Assemble the query path every entry point uses, so no two of them can diverge."""
    return GenerationService(
        settings,
        build_retrieval(settings, adapter, router, tenant),
        create_llm_adapter(settings),
        cache=cache,
        traces=traces,
        tenant=tenant,
        embedder=router.default,
    )


def build_documents(
    settings: Settings,
    adapter: VectorDBAdapter,
    journal: Journal,
    cache: SemanticCache | None,
) -> DocumentService:
    """Assemble document deletion over the caller's adapter, journal and cache.

    The cache is the caller's, not a new one: invalidating a cache built for this call
    would clear nothing a query will ever read.
    """
    return DocumentService(adapter, journal, cache=cache, locks=create_lock_store(settings))


def build_ingestion(
    settings: Settings,
    adapter: VectorDBAdapter,
    journal: Journal,
    cache: SemanticCache | None,
    *,
    router: TieringRouter | None = None,
) -> IngestionService:
    """Assemble ingestion against the caller's adapter.

    The adapter is passed in rather than built, so a background job shares the connection
    pool the request that started it was using instead of opening a second one.
    """
    return IngestionService(
        settings,
        journal=journal,
        adapter=adapter,
        router=router,
        cache=cache,
        locks=create_lock_store(settings),
    )
