"""FastAPI dependencies: what a router reads from ``app.state`` and the request.

Only wrappers live here. The services themselves are assembled by
:mod:`fasterrag.services.composition`, which the CLI and the facade call too; this module
hands that code the process-scoped parts the application holds — the vector database
adapter, the journal, the embedding router, the semantic cache, the LLM adapter — and the
tenant the auth middleware resolved. Routers hold zero business logic
(``docs/structure.md``), which in practice means they receive a ready service and return
whatever it produced.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.services import composition
from fasterrag.services.composition import create_embedding_router, create_semantic_cache
from fasterrag.services.jobs import JobRegistry
from fasterrag.services.journal import Journal, create_journal

__all__ = [
    "AcceptingJobs",
    "CurrentCache",
    "CurrentEmbeddings",
    "CurrentJobs",
    "CurrentLLM",
    "CurrentSettings",
    "CurrentTenant",
    "CurrentVectorDB",
    "JournalDep",
    "accepting_jobs",
    "current_tenant",
    "get_jobs",
    "get_journal",
    "get_settings",
    "get_vector_db",
    "shared_cache",
    "shared_embedding_router",
    "shared_llm",
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


def get_jobs(request: Request) -> JobRegistry:
    """Return the registry holding every job this process is running."""
    jobs: JobRegistry = request.app.state.jobs
    return jobs


def accepting_jobs(request: Request) -> JobRegistry:
    """Return the job registry, refusing with 503 while the process is draining."""
    jobs = get_jobs(request)
    jobs.require_accepting()
    return jobs


def shared_embedding_router(request: Request) -> TieringRouter:
    """Return the application's long-lived embedding router."""
    router: TieringRouter | None = getattr(request.app.state, "embeddings", None)
    if router is None:
        router = create_embedding_router(request.app.state.settings)
        request.app.state.embeddings = router
    return router


def shared_cache(request: Request) -> SemanticCache:
    """Return the application's long-lived semantic response cache."""
    cache: SemanticCache | None = getattr(request.app.state, "cache", None)
    if cache is None:
        cache = create_semantic_cache(request.app.state.settings)
        request.app.state.cache = cache
    return cache


def shared_llm(request: Request) -> LLMAdapter:
    """Return the application's long-lived LLM adapter, closed only by the lifespan."""
    llm: LLMAdapter | None = getattr(request.app.state, "llm", None)
    if llm is None:
        llm = composition.create_llm_adapter(request.app.state.settings)
        request.app.state.llm = llm
    return llm


def current_tenant(request: Request) -> str | None:
    """Return the tenant the auth middleware resolved for this request.

    Read from the request scope rather than re-parsed from the header: exactly one place
    decides who the caller is, and a second reader could disagree with the first.
    """
    tenant = request.scope.get("state", {}).get("tenant")
    return str(tenant) if tenant else None


CurrentTenant = Annotated[str | None, Depends(current_tenant)]
CurrentSettings = Annotated[Settings, Depends(get_settings)]
CurrentVectorDB = Annotated[VectorDBAdapter, Depends(get_vector_db)]
CurrentEmbeddings = Annotated[TieringRouter, Depends(shared_embedding_router)]
CurrentCache = Annotated[SemanticCache, Depends(shared_cache)]
CurrentLLM = Annotated[LLMAdapter, Depends(shared_llm)]
JournalDep = Annotated[Journal, Depends(get_journal)]
CurrentJobs = Annotated[JobRegistry, Depends(get_jobs)]
AcceptingJobs = Annotated[JobRegistry, Depends(accepting_jobs)]
