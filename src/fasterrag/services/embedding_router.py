"""The embedding router every entry point uses, behind the embedding cache when configured.

Entry points reach this through :mod:`fasterrag.services.composition`, which re-exports it;
it is a module of its own because the ingestion service builds a router when it is handed
none, and the composition root builds the ingestion service, so the two cannot share a
module without importing each other.

The API, the CLI, the facade and the ingestion service all need the same embedding router,
and they must all get it from one place, or a behaviour that is added to the router reaches
some entry points and not others. That is what happened to the embedding cache: it was
implemented, tested and constructed by nothing, so ``embeddings.cache.enabled`` validated,
defaulted to ``true`` and changed no behaviour anywhere (TASK-0442).

The cache is applied here rather than inside ``adapters.embeddings.tiering`` because of the
direction the imports run. The cache lives in ``core/`` and wraps an adapter; for the
adapter factory to apply it, ``adapters/`` would have to import ``core/``, adding an edge to
a cycle that TASK-0456 exists to remove. This module is one layer above both, where
importing both is what a service is for, so the wrapping costs no new edge.

So there are two functions named ``create_embedding_router``, on purpose. The one in
``adapters.embeddings.tiering`` builds the bare router and is this module's raw material.
The one here, re-exported by the composition root, is the one to call. They share a name so
that an entry point cannot hold both and so that swapping the import is the whole migration;
a caller outside ``adapters/`` that imports the bare one gets a router the cache switch
cannot reach, and ``tests/unit/services/test_composition.py`` fails the build on it.
"""

from __future__ import annotations

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.embeddings.tiering import create_embedding_router as create_bare_router
from fasterrag.config.schema import Settings
from fasterrag.core.cache import cache_embeddings, create_embedding_store
from fasterrag.observability.logging import get_logger

__all__ = ["create_embedding_router"]

_logger = get_logger(__name__)


def create_embedding_router(settings: Settings) -> TieringRouter:
    """Build the embedding router, behind the embedding cache when it is enabled.

    Every tier shares one store, owned by the default adapter's wrapper.
    """
    cache = settings.embeddings.cache
    if not cache.enabled:
        return create_bare_router(settings)

    store = create_embedding_store(settings)
    bare = create_bare_router(settings)

    wrapped: dict[int, EmbeddingAdapter] = {
        id(bare.default): cache_embeddings(bare.default, store, owns_store=True)
    }
    for _, adapter in bare.tiers:
        if id(adapter) not in wrapped:
            wrapped[id(adapter)] = cache_embeddings(adapter, store, owns_store=False)

    _logger.info(
        "embedding cache enabled",
        extra={"backend": cache.backend, "max_entries": cache.max_entries, "tiers": len(wrapped)},
    )
    return TieringRouter(
        wrapped[id(bare.default)],
        [(filters, wrapped[id(adapter)]) for filters, adapter in bare.tiers],
    )
