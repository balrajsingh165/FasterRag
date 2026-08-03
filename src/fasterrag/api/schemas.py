"""Request and response models for the REST surface.

Pydantic models rather than raw dicts, because the bounds in ``docs/api-reference.md`` — a
query of 1 to 8192 characters, a ``top_k`` matching ``retrieval.top_k``'s range — have to be
enforced somewhere, and enforcing them at the edge turns a bad request into a ``422`` with a
field path instead of a failure three layers down with no context.

Response models are deliberately absent for the query and job bodies. Those shapes are
produced by the service layer (``Answer.as_dict``, ``JobRecord``) and are already the
documented contract; re-declaring them here would create a second definition to keep in step,
and the first divergence would silently drop a field from the response.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CollectionRequest",
    "EstimateRequest",
    "IngestRequest",
    "QueryRequest",
    "ReplayRequest",
    "Source",
]

MAXIMUM_QUERY_LENGTH = 8192
MAXIMUM_TOP_K = 100


class Strict(BaseModel):
    """Base model that refuses unknown fields.

    A misspelled field is a silent no-op under the permissive default: a caller who sends
    ``top-k`` instead of ``top_k`` would get the configured default and never learn their
    override was ignored.
    """

    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    """One ingestion source."""

    type: Literal["path", "url", "inline"]
    value: str = Field(min_length=1)


class IngestRequest(Strict):
    """Body of ``POST /v1/ingest``."""

    sources: Annotated[list[Source], Field(min_length=1)]
    collection: str | None = None
    metadata: dict[str, Any] | None = None
    priority_class: str | None = None
    idempotency_key: str | None = None


class QueryRequest(Strict):
    """Body of ``POST /v1/query``."""

    query: Annotated[str, Field(min_length=1, max_length=MAXIMUM_QUERY_LENGTH)]
    collection: str | None = None
    top_k: Annotated[int, Field(ge=1, le=MAXIMUM_TOP_K)] | None = None
    filters: dict[str, Any] | None = None
    stream: bool | None = None
    include_chunks: bool = False


class CollectionRequest(Strict):
    """Body of ``POST /v1/collections``."""

    name: Annotated[str, Field(min_length=1, max_length=255)]
    distance: Literal["cosine", "dot", "euclid"] | None = None
    shard_number: Annotated[int, Field(ge=1)] | None = None
    replication_factor: Annotated[int, Field(ge=1)] | None = None


class EstimateRequest(Strict):
    """Body of ``POST /v1/estimate``."""

    sources: Annotated[list[str], Field(min_length=1)]
    all_providers: bool = False


class ExportRequest(Strict):
    """Body of ``POST /v1/admin/export``."""

    out: Annotated[str, Field(min_length=1)]
    collection: str | None = None
    include_vectors: bool = False


class ImportRequest(Strict):
    """Body of ``POST /v1/admin/import``."""

    archive: Annotated[str, Field(min_length=1)]
    target_collection: str | None = None
    reembed: bool = False


class ReplayRequest(Strict):
    """Body of ``POST /v1/replay``."""

    trace_id: Annotated[str, Field(min_length=1)]
    config_overrides: dict[str, Any] | None = None
    diff_only: bool = False
