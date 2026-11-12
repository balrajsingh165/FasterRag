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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
MAXIMUM_DESCRIPTOR_LENGTH = 255

Descriptor = Annotated[str, Field(min_length=1, max_length=MAXIMUM_DESCRIPTOR_LENGTH)]


class Strict(BaseModel):
    """Base model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    """One ingestion source; ``media_type`` and ``filename`` describe an inline payload only."""

    type: Literal["path", "url", "inline"]
    value: str = Field(min_length=1)
    media_type: Descriptor | None = None
    filename: Descriptor | None = None

    @model_validator(mode="after")
    def _descriptors_describe_inline_payloads(self) -> Source:
        """Refuse a descriptor on a path or URL, whose type comes from the file or the server."""
        if self.type != "inline" and (self.media_type is not None or self.filename is not None):
            raise ValueError("media_type and filename apply to inline sources only")
        return self


class IngestRequest(Strict):
    """Body of ``POST /v1/ingest``."""

    sources: Annotated[list[Source], Field(min_length=1)]
    collection: str | None = None
    metadata: dict[str, Any] | None = None
    priority_class: str | None = None
    # CRITICAL: marked deprecated through the schema rather than through Field(deprecated=),
    # which warns on every *read* of the attribute — including the read that decides whether
    # the caller sent it. That turned one deprecated request into a DeprecationWarning on
    # every ingest, whether or not the field was used. The published OpenAPI flag is the
    # same either way; the response header is what reaches a client that ignores schemas.
    idempotency_key: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Deprecated: send the Idempotency-Key header instead. Removed in 1.0 (TASK-0318)."
            ),
            json_schema_extra={"deprecated": True},
        ),
    ] = None


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
