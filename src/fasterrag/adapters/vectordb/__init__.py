"""Vector database adapters and the factory that selects one from configuration."""

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.adapters.vectordb.factory import (
    available_providers,
    create_vector_db_adapter,
)

__all__ = [
    "CollectionSpec",
    "HealthStatus",
    "Point",
    "PointSelector",
    "PointUpdate",
    "ScoredPoint",
    "SearchQuery",
    "UpsertResult",
    "VectorDBAdapter",
    "available_providers",
    "create_vector_db_adapter",
]
