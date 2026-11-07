r"""Milvus adapter — the contract proved against a distributed, schema-first store.

Qdrant is the reference implementation (``docs/adr/ADR-0001``), pgvector proved the
contract holds over SQL, and Chroma proved it holds with no server at all. Milvus is the
fourth paradigm: a horizontally scaled cluster with a *declared schema*, where a
collection's fields and indexes are fixed at creation and the data is only queryable once
the collection is loaded into memory.

**Layout.** Everything is a collection in one Milvus database:

* One Milvus collection per fasterRag collection, named ``frag_<slug>_<digest>``. The
  derivation exists because Milvus identifiers match ``[A-Za-z_][A-Za-z0-9_]{0,254}`` while
  a fasterRag collection name may contain ``-`` and may start with a digit. The logical name
  is recorded in the collection's own ``description``, which is what ``list_collections``
  reads back.
* One Milvus collection per snapshot, named ``fragsnap_<slug>_<digest>`` and tagged in its
  description with the collection it was taken from.
* Aliases are **native Milvus aliases**, under derived names ``fragalias_<slug>_<digest>``
  for the same identifier reason.

**Every field is declared; nothing rides on a dynamic field.** The payload lives in one
``JSON`` column rather than in Milvus's dynamic ``$meta``, which costs nothing in
expressiveness — ``fr_payload["department"] == "finance"`` pushes down exactly as a dynamic
key would — and buys two things. A payload key can then be *anything*, including one that
collides with a declared field name, because it is inside the blob rather than beside it;
and the schema a collection was created with is the schema it keeps, so a reader can tell
what a collection holds without inspecting a row. The honest cost: Milvus indexes JSON paths
only when told which paths to index, and the adapter cannot know a deployment's payload keys
in advance, so a filtered search scans the JSON column rather than an index.

**The BM25 leg is a sparse vector, and Milvus supplies IDF through counting, not scoring.**
Milvus has no IDF modifier for a caller-supplied ``SPARSE_FLOAT_VECTOR``: the only metric
such a field accepts is ``IP``, a plain dot product of the query's weights against the
stored ones. Its BM25 metric belongs to the full-text ``Function``, which tokenizes text
server-side and would therefore hit different terms than every other backend
(``docs/adr/ADR-0007`` requires one tokenizer for all of them), and which never sees the
encoded vector this adapter is handed anyway.

So the ranking is pushed down, exactly, by moving IDF into the *query* vector. Each
document's term indices are stored a second time as an ``ARRAY`` of ``INT64`` under an
inverted index, which makes ``df(term)`` one server-side ``count(*)`` per query term; the
adapter turns those counts into IDF weights and sends them as the sparse query, so Milvus's
dot product computes ``sum(tf * idf)`` — the BM25 score — and does the top-k itself. IDF is
still a statistic of the live corpus read from the store that holds it, which is the split
the ADR mandates; what the adapter adds is arithmetic over ``len(terms) + 1`` counts, not a
scan of the postings. That is the difference from Chroma, which has to read every matching
posting into the process.

**Those stored weights are what ``iterate_points`` returns** (TASK-0291). Milvus hands a
sparse field back as an ``{index: weight}`` mapping; the adapter sorts it ascending by term
and hands over exactly the weights ``encode_document`` wrote, float32-rounded by the
storage format as the dense vectors already are.

**Snapshots are copies into a second collection, like pgvector's and Chroma's.** pymilvus
3.0 does expose ``create_snapshot``/``restore_snapshot``, but they are documented as
private preview against Zilliz Cloud and are not confirmed to exist on a self-hosted
open-source server, so building the contract on them would make ``snapshot()`` work on one
deployment and fail on another. A collection copy works everywhere. It is honestly less
than a backup: it lives in the same cluster as the data it protects, so it survives a bad
delete but not the loss of that cluster, and it cannot be shipped elsewhere as a file.

**Reads are ``Strong``, deliberately.** Milvus defaults to ``Bounded`` consistency, under
which a point just written is not necessarily visible to the next search. Every read this
adapter issues asks for ``Strong`` instead, because an ingest pipeline that cannot see what
it just indexed is indistinguishable from one that failed to index it.

**The client is synchronous, so every call runs off the event loop** through
``asyncio.to_thread``; no gRPC call is made on the loop thread. ``AsyncMilvusClient`` exists
but has no ``query_iterator``, and a hand-rolled keyset walk over an unordered ``query`` is
precisely the thing that silently skips rows.

**Remaining limits of this first cut**, none of them silent: ``shard_number`` and
``replication_factor`` are not yet mapped onto Milvus's own sharding and are ignored rather
than reinterpreted; ``update()`` is a read-modify-write because Milvus has no partial
update, so a concurrent writer can lose a merge; and a document carrying more than
``MAXIMUM_TERMS`` distinct terms is refused rather than truncated, because Milvus caps an
``ARRAY`` field's capacity and a truncated posting list would score wrong forever.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, TypeVar

from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    Distance,
    Filter,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    SparseVector,
    UpsertResult,
    VectorDBAdapter,
    validate_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError, ErrorCode, FasterRagError, ProviderError
from fasterrag.observability.logging import get_logger

try:
    from pymilvus import DataType, MilvusClient
    from pymilvus import exceptions as milvus_errors

    _VENDOR_ERRORS: tuple[type[BaseException], ...] = (
        milvus_errors.MilvusException,
        OSError,
        ValueError,
    )
    _DRIVER_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    _DRIVER_IMPORT_ERROR = exc

__all__ = [
    "DENSE_FIELD",
    "INSTALL_HINT",
    "MAXIMUM_TERMS",
    "PAYLOAD_FIELD",
    "PK_FIELD",
    "SPARSE_FIELD",
    "TERMS_FIELD",
    "MilvusAdapter",
    "alias_name_for",
    "collection_name_for",
    "snapshot_name_for",
    "to_milvus_filter",
]

INSTALL_HINT: Final = 'pip install "fasterrag[milvus]"'

# CRITICAL: these five field names and the three collection-name prefixes are the on-server
# contract of a fasterRag Milvus database. A Milvus schema is fixed at creation, so renaming
# one of these does not migrate anything — it makes every collection already written
# unreadable by the adapter that wrote it, with no error until the first query returns
# nothing.
PK_FIELD: Final = "fr_id"
DENSE_FIELD: Final = "fr_vector"
SPARSE_FIELD: Final = "fr_sparse"
TERMS_FIELD: Final = "fr_terms"
PAYLOAD_FIELD: Final = "fr_payload"

_COLLECTION_PREFIX: Final = "frag"
_SNAPSHOT_PREFIX: Final = "fragsnap"
_ALIAS_PREFIX: Final = "fragalias"

MAXIMUM_TERMS: Final = 4096

_LAYOUT_VERSION: Final = 1
_KIND_KEY: Final = "k"
_NAME_KEY: Final = "n"
_SOURCE_KEY: Final = "s"
_DISTANCE_KEY: Final = "d"
_VERSION_KEY: Final = "v"

_COLLECTION_KIND: Final = "c"
_SNAPSHOT_KIND: Final = "s"

_PK_MAX_LENGTH: Final = 512
_COPY_BATCH: Final = 500
_COUNT_EXPRESSION: Final = "count(*)"
_STRONG: Final = "Strong"
_MATCH_EVERY_POINT: Final = f'{PK_FIELD} != ""'

_SLUG_LIMIT: Final = 32
_DIGEST_LIMIT: Final = 12
_UNSAFE_NAME_CHARS: Final = re.compile(r"[^a-z0-9]+")

_METRICS: Final[dict[Distance, str]] = {"cosine": "COSINE", "dot": "IP", "euclid": "L2"}
_DISTANCES: Final[dict[str, Distance]] = {"cosine": "cosine", "dot": "dot", "euclid": "euclid"}
_SPARSE_METRIC: Final = "IP"

_COMPARISONS: Final[dict[str, str]] = {
    "$eq": "==",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
_SET_OPERATORS: Final[dict[str, str]] = {"$in": "in", "$nin": "not in"}

_NOT_FOUND_PHRASES: Final[tuple[str, ...]] = ("not found", "not exist", "doesn't exist")
_CONFLICT_PHRASES: Final[tuple[str, ...]] = ("already exist", "duplicate")
_AUTH_PHRASES: Final[tuple[str, ...]] = (
    "unauthenticated",
    "permission denied",
    "auth check failure",
    "user not authenticated",
)

_RATE_LIMIT_CODE: Final = 8
_FORCE_DENY_CODE: Final = 9
_COLLECTION_NOT_FOUND_CODE: Final = 100

_logger = get_logger(__name__)

_T = TypeVar("_T")


def _slug(value: str) -> str:
    """Return the readable half of a derived Milvus collection name."""
    cleaned = _UNSAFE_NAME_CHARS.sub("_", value.lower()).strip("_")
    return cleaned[:_SLUG_LIMIT].strip("_") or "collection"


def _digest(*parts: str) -> str:
    """Return the short stable hash that makes a derived name unique."""
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:_DIGEST_LIMIT]


def collection_name_for(collection: str) -> str:
    """Return the Milvus collection a fasterRag collection's points are stored in."""
    return f"{_COLLECTION_PREFIX}_{_slug(collection)}_{_digest(collection)}"


def snapshot_name_for(collection: str, snapshot: str) -> str:
    """Return the Milvus collection a snapshot's records are copied into."""
    return f"{_SNAPSHOT_PREFIX}_{_slug(snapshot)}_{_digest(collection, snapshot)}"


def alias_name_for(alias: str) -> str:
    """Return the Milvus alias a fasterRag alias name maps to."""
    return f"{_ALIAS_PREFIX}_{_slug(alias)}_{_digest(alias)}"


def _literal(key: str, value: Any) -> str:
    """Render one filter comparand as a Milvus expression literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise FasterRagError(
        f"filter on {key!r} compares against {type(value).__name__}, which has no Milvus "
        "expression literal; compare against a string, a number, or a boolean",
        code=ErrorCode.VALIDATION_FAILED,
    )


def _path(key: str) -> str:
    """Return the Milvus expression addressing one payload key inside the JSON column."""
    return f"{PAYLOAD_FIELD}[{json.dumps(key)}]"


def _clause(key: str, operator: str, value: Any) -> str:
    """Translate one supported filter operator into a Milvus boolean expression."""
    if operator in _SET_OPERATORS:
        if not isinstance(value, list):
            raise FasterRagError(
                f"filter on {key!r} with {operator} requires a list of values",
                code=ErrorCode.VALIDATION_FAILED,
            )
        rendered = ", ".join(_literal(key, member) for member in value)
        return f"{_path(key)} {_SET_OPERATORS[operator]} [{rendered}]"
    return f"{_path(key)} {_COMPARISONS[operator]} {_literal(key, value)}"


def to_milvus_filter(filters: Filter | None) -> str:
    """Translate a vendor-neutral filter expression into a Milvus boolean expression."""
    validate_filter(filters)
    if not filters:
        return ""

    clauses: list[str] = []
    for key, condition in filters.items():
        if isinstance(condition, Mapping):
            operator = next(iter(condition))
            clauses.append(_clause(key, operator, condition[operator]))
        else:
            clauses.append(_clause(key, "$eq", condition))
    return " and ".join(f"({clause})" for clause in clauses)


def _term_clause(term: int) -> str:
    """Return the expression matching every document whose posting list holds ``term``."""
    return f"array_contains({TERMS_FIELD}, {term})"


def _idf(total: int, documents: int) -> float:
    """Return the BM25 inverse document frequency for a term, as the other backends do."""
    return math.log(1 + (total - documents + 0.5) / (documents + 0.5))


def _score_of(distance: Distance, measured: float) -> float:
    """Turn a Milvus distance into the higher-is-better score the contract requires."""
    if distance == "euclid":
        return -measured
    return measured


def _distance_of(stored: str) -> Distance:
    """Translate a recorded distance value back into the neutral literal."""
    known = _DISTANCES.get(stored)
    if known is None:
        raise FasterRagError(
            f"the fasterRag description records distance {stored!r}, which this version "
            f"does not implement; known values are: {', '.join(sorted(_DISTANCES))}",
            code=ErrorCode.INTERNAL,
        )
    return known


def _floats(stored: object) -> list[float]:
    """Convert a returned dense vector into plain floats, so no vendor type escapes."""
    if stored is None:
        return []
    return [float(value) for value in _as_iterable(stored)]


def _as_iterable(stored: object) -> Iterable[Any]:
    """Return a returned vector as a plain iterable, refusing anything that is not one."""
    if isinstance(stored, Sequence) and not isinstance(stored, str | bytes):
        return stored
    if hasattr(stored, "tolist"):
        listed = stored.tolist()
        if isinstance(listed, list):
            return listed
    raise FasterRagError(
        f"milvus returned a dense vector as {type(stored).__name__}, which this adapter "
        "cannot read as a sequence of floats",
        code=ErrorCode.INTERNAL,
    )


def _to_sparse(stored: object) -> SparseVector | None:
    """Convert a returned sparse field into the neutral type, terms ascending.

    Milvus hands a sparse vector back as an ``{index: weight}`` mapping; anything else,
    including an absent or empty field, reads as no keyword leg.
    """
    if not isinstance(stored, Mapping) or not stored:
        return None
    postings = sorted((int(index), float(weight)) for index, weight in stored.items())
    return SparseVector(
        indices=[index for index, _ in postings], values=[weight for _, weight in postings]
    )


def _description_of(spec: CollectionSpec, **extra: str) -> str:
    """Build the JSON description carrying what a Milvus schema cannot record itself."""
    recorded: dict[str, Any] = {
        _VERSION_KEY: _LAYOUT_VERSION,
        _NAME_KEY: spec.name,
        _DISTANCE_KEY: spec.distance,
        **extra,
    }
    return json.dumps(recorded, separators=(",", ":"))


def _recorded(description: str) -> dict[str, Any]:
    """Read back a fasterRag collection description, tolerating a foreign collection."""
    if not description.startswith("{"):
        return {}
    try:
        decoded = json.loads(description)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


@dataclass(frozen=True, slots=True)
class _Collection:
    """One resolved collection: its logical name, its Milvus name, and its shape."""

    name: str
    physical: str
    dimensions: int
    distance: Distance
    sparse: bool


class MilvusAdapter(VectorDBAdapter):
    """Milvus implementation of the vector database contract."""

    def __init__(self, settings: Settings) -> None:
        """Build the adapter. The gRPC channel is opened on first use."""
        if _DRIVER_IMPORT_ERROR is not None:
            raise ConfigError(
                "vector_db.provider is 'milvus' but its driver is not installed "
                f"({_DRIVER_IMPORT_ERROR}); install it with: {INSTALL_HINT}"
            )

        vector_db = settings.vector_db
        scheme = "https" if vector_db.https else "http"
        self._uri = f"{scheme}://{vector_db.host}:{vector_db.port}"
        self._token_env = vector_db.api_key_env
        self._timeout_seconds = max(1.0, settings.reliability.timeouts.vector_db_ms / 1000)
        self._client: Any = None
        self._described: dict[str, _Collection] = {}
        self._loaded: set[str] = set()
        self._lock = asyncio.Lock()

    def _auth_error(self, operation: str) -> ProviderError:
        """Build the non-retryable error for a rejected credential."""
        named = self._token_env or "vector_db.api_key_env"
        return ProviderError(
            f"milvus rejected the credentials during {operation}; check the token in the "
            f"{named} environment variable and the server's authorization configuration",
            code=ErrorCode.VECTOR_DB_AUTH_FAILED,
            retryable=False,
        )

    def _translate(self, exc: BaseException, operation: str) -> FasterRagError:
        """Map a Milvus failure onto the typed error taxonomy."""
        if isinstance(exc, OSError):
            return ProviderError(
                f"milvus was unreachable during {operation}: {type(exc).__name__}",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        if not isinstance(exc, milvus_errors.MilvusException):
            return FasterRagError(
                f"milvus rejected the request during {operation}: {exc}",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return self._translate_server(exc, operation)

    def _translate_server(self, exc: Any, operation: str) -> FasterRagError:
        """Map one ``MilvusException`` onto the taxonomy, preferring its own codes."""
        message = str(getattr(exc, "message", "") or exc).lower()
        code = getattr(exc, "code", None)

        if code == _FORCE_DENY_CODE or any(phrase in message for phrase in _AUTH_PHRASES):
            return self._auth_error(operation)
        if code == _RATE_LIMIT_CODE:
            return ProviderError(
                f"milvus refused {operation} for exceeding a rate limit",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        if code == _COLLECTION_NOT_FOUND_CODE or any(
            phrase in message for phrase in _NOT_FOUND_PHRASES
        ):
            return FasterRagError(
                f"milvus reported a missing resource during {operation}",
                code=ErrorCode.NOT_FOUND,
            )
        if any(phrase in message for phrase in _CONFLICT_PHRASES):
            return FasterRagError(
                f"milvus reported a conflicting state during {operation}",
                code=ErrorCode.CONFLICT,
            )
        if bool(getattr(exc, "is_input_error", False)):
            return FasterRagError(
                f"milvus rejected the request during {operation}: {getattr(exc, 'message', exc)}",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return ProviderError(
            f"milvus failed during {operation}: {type(exc).__name__}",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=bool(getattr(exc, "retriable", True)),
        )

    async def _call(self, operation: str, work: Callable[[], _T]) -> _T:
        """Run one blocking Milvus call off the loop, letting no vendor type escape."""
        try:
            return await asyncio.to_thread(work)
        except _VENDOR_ERRORS as exc:
            raise self._translate(exc, operation) from exc

    def _build_client(self) -> Any:
        """Open the gRPC channel, naming the credential by its environment variable."""
        token = os.environ.get(self._token_env, "") if self._token_env else ""
        return MilvusClient(uri=self._uri, token=token, timeout=self._timeout_seconds)

    async def _ensure_client(self) -> Any:
        """Return the client, opening the channel on first use."""
        existing = self._client
        if existing is not None:
            return existing

        async with self._lock:
            if self._client is None:
                self._client = await self._call("connect", self._build_client)
            return self._client

    async def _has(self, physical: str) -> bool:
        """Report whether Milvus holds a collection under this derived name."""
        client = await self._ensure_client()
        return bool(await self._call("has_collection", lambda: client.has_collection(physical)))

    async def _ensure_loaded(self, physical: str) -> None:
        """Load a collection into memory, which Milvus requires before any read."""
        if physical in self._loaded:
            return
        client = await self._ensure_client()
        await self._call("load_collection", lambda: client.load_collection(physical))
        self._loaded.add(physical)

    async def _describe(self, physical: str) -> _Collection | None:
        """Read one Milvus collection's declared schema and recorded fasterRag shape."""
        cached = self._described.get(physical)
        if cached is not None:
            return cached

        if not await self._has(physical):
            return None

        client = await self._ensure_client()
        raw = await self._call("describe_collection", lambda: client.describe_collection(physical))
        described = self._shape_of(physical, raw)
        if described is not None:
            self._described[physical] = described
        return described

    @staticmethod
    def _shape_of(physical: str, raw: Mapping[str, Any]) -> _Collection | None:
        """Turn one ``describe_collection`` response into the adapter's view of it."""
        recorded = _recorded(str(raw.get("description") or ""))
        if recorded.get(_KIND_KEY) not in (_COLLECTION_KIND, _SNAPSHOT_KIND):
            return None

        fields = {str(field.get("name")): field for field in raw.get("fields") or []}
        dense = fields.get(DENSE_FIELD) or {}
        params = dense.get("params") or {}
        return _Collection(
            name=str(recorded.get(_NAME_KEY, "")),
            physical=physical,
            dimensions=int(params.get("dim", 0)),
            distance=_distance_of(str(recorded.get(_DISTANCE_KEY, "cosine"))),
            sparse=SPARSE_FIELD in fields,
        )

    async def _resolve(self, name: str) -> _Collection | None:
        """Resolve a collection name or an alias, a real name winning over an alias."""
        direct = await self._describe(collection_name_for(name))
        if direct is not None:
            return direct

        physical = await self._alias_collection(name)
        return None if physical is None else await self._describe(physical)

    async def _require(self, name: str) -> _Collection:
        """Resolve a collection, raising rather than letting a typo read nothing."""
        found = await self._resolve(name)
        if found is None:
            raise FasterRagError(
                f"collection {name!r} does not exist on {self._uri}, and no alias points at one",
                code=ErrorCode.NOT_FOUND,
            )
        await self._ensure_loaded(found.physical)
        return found

    def _schema(self, client: Any, spec: CollectionSpec, description: str) -> Any:
        """Declare the fields one fasterRag collection needs, sparse leg included."""
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(PK_FIELD, DataType.VARCHAR, is_primary=True, max_length=_PK_MAX_LENGTH)
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=spec.dimensions)
        schema.add_field(PAYLOAD_FIELD, DataType.JSON)
        if spec.sparse:
            schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(
                TERMS_FIELD,
                DataType.ARRAY,
                element_type=DataType.INT64,
                max_capacity=MAXIMUM_TERMS,
            )
        schema.description = description
        return schema

    @staticmethod
    def _indexes(client: Any, spec: CollectionSpec) -> Any:
        """Declare the indexes that make each leg searchable and ``df`` countable."""
        params = client.prepare_index_params()
        params.add_index(
            field_name=DENSE_FIELD, index_type="AUTOINDEX", metric_type=_METRICS[spec.distance]
        )
        if spec.sparse:
            params.add_index(
                field_name=SPARSE_FIELD,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type=_SPARSE_METRIC,
            )
            params.add_index(field_name=TERMS_FIELD, index_type="INVERTED")
        return params

    async def _create(self, physical: str, spec: CollectionSpec, description: str) -> None:
        """Create one Milvus collection with its schema, its indexes, and a strong read."""
        client = await self._ensure_client()
        schema = self._schema(client, spec, description)
        indexes = self._indexes(client, spec)
        await self._call(
            "create_collection",
            lambda: client.create_collection(
                collection_name=physical,
                schema=schema,
                index_params=indexes,
                consistency_level=_STRONG,
            ),
        )
        self._loaded.add(physical)
        self._described.pop(physical, None)

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create the Milvus collection, or verify an existing one is compatible."""
        existing = await self._resolve(spec.name)
        if existing is not None:
            self._require_compatible(spec, existing)
            return

        await self._create(
            collection_name_for(spec.name),
            spec,
            _description_of(spec, **{_KIND_KEY: _COLLECTION_KIND}),
        )

    @staticmethod
    def _require_compatible(spec: CollectionSpec, existing: _Collection) -> None:
        """Reject an existing collection that cannot hold ``spec``."""
        if existing.dimensions != spec.dimensions:
            raise FasterRagError(
                f"collection {spec.name!r} already exists with {existing.dimensions} "
                f"dimensions, but the configuration expects {spec.dimensions}; re-embed "
                "through a blue/green reindex rather than writing mixed vectors",
                code=ErrorCode.CONFLICT,
            )
        if existing.distance != spec.distance:
            raise FasterRagError(
                f"collection {spec.name!r} already exists with distance "
                f"{existing.distance!r}, but the configuration expects {spec.distance!r}",
                code=ErrorCode.CONFLICT,
            )
        if spec.sparse and not existing.sparse:
            raise FasterRagError(
                f"collection {spec.name!r} already exists without a sparse index; a Milvus "
                "schema is fixed at creation, so a keyword leg cannot be bolted onto it — "
                "rebuild it through a blue/green reindex",
                code=ErrorCode.CONFLICT,
            )

    async def _owned(self, prefix: str) -> list[_Collection]:
        """Return every Milvus collection this adapter owns under one name prefix."""
        client = await self._ensure_client()
        listed = await self._call("list_collections", client.list_collections)
        found: list[_Collection] = []
        for physical in sorted(str(name) for name in listed):
            if not physical.startswith(f"{prefix}_"):
                continue
            described = await self._describe(physical)
            if described is not None:
                found.append(described)
        return found

    async def _count(self, physical: str, expression: str = "") -> int:
        """Return an exact row count, which only a strong read can give."""
        client = await self._ensure_client()
        await self._ensure_loaded(physical)
        rows = await self._call(
            "count",
            lambda: client.query(
                collection_name=physical,
                filter=expression,
                output_fields=[_COUNT_EXPRESSION],
                consistency_level=_STRONG,
            ),
        )
        listed = list(rows)
        return int(listed[0][_COUNT_EXPRESSION]) if listed else 0

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection the database holds, with an exact row count."""
        return [
            CollectionInfo(
                name=described.name,
                vectors=await self._count(described.physical),
                dimensions=described.dimensions,
                distance=described.distance,
                sparse=described.sparse,
            )
            for described in await self._owned(_COLLECTION_PREFIX)
        ]

    async def _drop(self, physical: str) -> None:
        """Drop one Milvus collection and forget everything cached about it."""
        client = await self._ensure_client()
        await self._call("drop_collection", lambda: client.drop_collection(physical))
        self._described.pop(physical, None)
        self._loaded.discard(physical)

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, its snapshots, and any alias pointing at it."""
        physical = collection_name_for(name)
        described = await self._describe(physical)
        if described is None:
            return False

        for snapshot in await self.list_snapshots(name):
            await self._drop(snapshot_name_for(described.name, snapshot))

        client = await self._ensure_client()
        pointing = await self._call(
            "drop_collection", lambda: client.list_aliases(collection_name=physical)
        )
        for alias in _alias_names(pointing):
            await self._call("drop_collection", _bound(client.drop_alias, alias))

        await self._drop(physical)
        return True

    async def snapshot(self, collection: str) -> str:
        """Copy a collection's rows into a second collection and name the copy."""
        target = await self._require(collection)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        name = f"{target.name}-{stamp}-{uuid.uuid4().hex[:8]}"
        physical = snapshot_name_for(target.name, name)

        spec = CollectionSpec(
            name=name,
            dimensions=target.dimensions,
            distance=target.distance,
            sparse=target.sparse,
        )
        await self._create(
            physical,
            spec,
            _description_of(spec, **{_KIND_KEY: _SNAPSHOT_KIND, _SOURCE_KEY: target.name}),
        )
        await self._copy(target, physical)
        return name

    async def _copy(self, source: _Collection, destination: str) -> None:
        """Copy every row of one collection into another, one page at a time."""
        client = await self._ensure_client()
        page: list[dict[str, Any]] = []
        async for row in self._walk(source, with_vectors=True, batch_size=_COPY_BATCH):
            page.append(row)
            if len(page) >= _COPY_BATCH:
                await self._call("snapshot", _writer(client, destination, page))
                page = []
        if page:
            await self._call("snapshot", _writer(client, destination, page))

    async def list_snapshots(self, collection: str) -> list[str]:
        """Return the snapshots held for a collection, newest last."""
        target = await self._require(collection)
        client = await self._ensure_client()
        listed = await self._call("list_snapshots", client.list_collections)

        held: list[str] = []
        for physical in sorted(str(name) for name in listed):
            if not physical.startswith(f"{_SNAPSHOT_PREFIX}_"):
                continue
            raw: Mapping[str, Any] = await self._call(
                "list_snapshots", _bound(client.describe_collection, physical)
            )
            recorded = _recorded(str(raw.get("description") or ""))
            if recorded.get(_SOURCE_KEY) == target.name:
                held.append(str(recorded.get(_NAME_KEY, "")))
        return sorted(held)

    async def _snapshot_of(self, collection: str, snapshot: str) -> _Collection | None:
        """Resolve one snapshot of a collection to the Milvus collection holding it."""
        found = await self._resolve(collection)
        owner = found.name if found is not None else collection
        return await self._describe(snapshot_name_for(owner, snapshot))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Delete one snapshot, tolerating one a previous retention run already pruned."""
        held = await self._snapshot_of(collection, snapshot)
        if held is None:
            return False
        await self._drop(held.physical)
        return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Replace a collection's contents with those a snapshot holds."""
        held = await self._snapshot_of(collection, snapshot)
        if held is None:
            raise FasterRagError(
                f"no snapshot named {snapshot!r} is recorded for collection {collection!r}",
                code=ErrorCode.NOT_FOUND,
            )

        target = await self._require(collection)
        self._require_compatible(
            CollectionSpec(
                name=collection,
                dimensions=held.dimensions,
                distance=held.distance,
                sparse=held.sparse,
            ),
            target,
        )

        client = await self._ensure_client()
        await self._ensure_loaded(held.physical)
        await self._call(
            "restore_snapshot",
            lambda: client.delete(collection_name=target.physical, filter=_MATCH_EVERY_POINT),
        )
        await self._copy(held, target.physical)

    async def _alias_collection(self, alias: str) -> str | None:
        """Return the Milvus collection a fasterRag alias resolves to, if any."""
        client = await self._ensure_client()
        physical_alias = alias_name_for(alias)
        listed = await self._call("alias_target", lambda: client.list_aliases())
        if physical_alias not in _alias_names(listed):
            return None

        described = await self._call("alias_target", lambda: client.describe_alias(physical_alias))
        target = described.get("collection_name")
        return None if target is None else str(target)

    async def set_alias(self, alias: str, collection: str) -> None:
        """Point an alias at a collection through Milvus's own alias primitive.

        ``alter_alias`` replaces the previous target in one server-side operation, so
        there is no instant in which the alias resolves to nothing.
        """
        target = await self._require(collection)
        if await self._describe(collection_name_for(alias)) is not None:
            raise FasterRagError(
                f"{alias!r} is already a collection, so it cannot also be an alias; "
                "every query would resolve to the collection instead",
                code=ErrorCode.CONFLICT,
            )

        client = await self._ensure_client()
        physical_alias = alias_name_for(alias)
        existing = await self._alias_collection(alias)
        if existing is None:
            await self._call(
                "set_alias",
                lambda: client.create_alias(collection_name=target.physical, alias=physical_alias),
            )
            return
        await self._call(
            "set_alias",
            lambda: client.alter_alias(collection_name=target.physical, alias=physical_alias),
        )

    async def alias_target(self, alias: str) -> str | None:
        """Return the collection an alias resolves to."""
        physical = await self._alias_collection(alias)
        if physical is None:
            return None
        described = await self._describe(physical)
        return None if described is None else described.name

    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, leaving the collection it pointed at in place."""
        if await self._alias_collection(alias) is None:
            return False
        client = await self._ensure_client()
        await self._call("delete_alias", lambda: client.drop_alias(alias_name_for(alias)))
        return True

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points and their postings, overwriting any that already exist."""
        if not points:
            return UpsertResult(upserted=0)

        grouped: dict[str, list[Point]] = {}
        for entry in points:
            grouped.setdefault(entry.collection, []).append(entry)

        client = await self._ensure_client()
        for name, group in grouped.items():
            target = await self._require(name)
            self._require_matching_dimensions(target, group)
            self._require_sparse_index(target, group)
            for start in range(0, len(group), _COPY_BATCH):
                rows = [_row_for(entry, target) for entry in group[start : start + _COPY_BATCH]]
                await self._call("upsert", _writer(client, target.physical, rows))

        return UpsertResult(upserted=len(points))

    @staticmethod
    def _require_matching_dimensions(target: _Collection, batch: list[Point]) -> None:
        """Reject a batch whose vectors do not match the collection's vector size."""
        for entry in batch:
            actual = len(entry.vector)
            if actual != target.dimensions:
                raise EmbedError(
                    f"point {entry.point_id!r} has {actual} dimensions but collection "
                    f"{target.name!r} stores {target.dimensions}; the configured embedding "
                    "model does not match the one this collection was built with",
                    retryable=False,
                )

    @staticmethod
    def _require_sparse_index(target: _Collection, batch: list[Point]) -> None:
        """Reject postings a collection cannot hold, by schema or by array capacity."""
        for entry in batch:
            if entry.sparse is None or entry.sparse.empty:
                continue
            if not target.sparse:
                raise EmbedError(
                    f"point {entry.point_id!r} carries a sparse vector but collection "
                    f"{target.name!r} has no sparse index; recreate it with sparse enabled",
                    retryable=False,
                )
            if len(entry.sparse) > MAXIMUM_TERMS:
                raise EmbedError(
                    f"point {entry.point_id!r} carries {len(entry.sparse)} distinct terms "
                    f"but a Milvus array field holds at most {MAXIMUM_TERMS}; chunk the "
                    "document smaller rather than scoring it against a truncated posting "
                    "list",
                    retryable=False,
                )

    def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection through Milvus's query iterator."""
        return self._iterate(collection, with_vectors=with_vectors, batch_size=batch_size)

    async def _iterate(
        self, collection: str, *, with_vectors: bool, batch_size: int
    ) -> AsyncIterator[Point]:
        """Walk a resolved collection, converting each row into the neutral type."""
        target = await self._require(collection)
        async for row in self._walk(target, with_vectors=with_vectors, batch_size=batch_size):
            yield Point(
                point_id=str(row.get(PK_FIELD, "")),
                collection=collection,
                vector=_floats(row.get(DENSE_FIELD)) if with_vectors else [],
                payload=dict(row.get(PAYLOAD_FIELD) or {}),
                sparse=_to_sparse(row.get(SPARSE_FIELD)) if with_vectors else None,
            )

    async def _walk(
        self, target: _Collection, *, with_vectors: bool, batch_size: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Page every row of a collection off the event loop, cursor closed either way."""
        client = await self._ensure_client()
        fields = [PK_FIELD, PAYLOAD_FIELD]
        if with_vectors:
            fields.append(DENSE_FIELD)
            if target.sparse:
                fields.extend((SPARSE_FIELD, TERMS_FIELD))

        cursor = await self._call(
            "iterate_points",
            lambda: client.query_iterator(
                collection_name=target.physical,
                batch_size=batch_size,
                output_fields=fields,
                consistency_level=_STRONG,
            ),
        )
        try:
            while True:
                rows = await self._call("iterate_points", cursor.next)
                if not rows:
                    return
                for row in rows:
                    yield dict(row)
        finally:
            await self._call("iterate_points", cursor.close)

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points on whichever leg the query carries."""
        if query.limit <= 0:
            return []

        target = await self._require(query.collection)
        if query.sparse is not None:
            return await self._search_sparse(target, query)
        return await self._search_dense(target, query)

    @staticmethod
    def _output_fields(query: SearchQuery) -> list[str]:
        """Name the fields a search must bring back for the result it was asked for."""
        fields = [PK_FIELD]
        if query.with_payload:
            fields.append(PAYLOAD_FIELD)
        if query.with_vectors:
            fields.append(DENSE_FIELD)
        return fields

    async def _run_search(
        self,
        target: _Collection,
        query: SearchQuery,
        *,
        data: list[Any],
        field: str,
        metric: str,
        expression: str,
    ) -> list[list[dict[str, Any]]]:
        """Issue one Milvus vector search against a named index."""
        client = await self._ensure_client()
        fields = self._output_fields(query)
        found = await self._call(
            "search",
            lambda: client.search(
                collection_name=target.physical,
                data=data,
                anns_field=field,
                limit=query.limit,
                filter=expression,
                output_fields=fields,
                search_params={"metric_type": metric},
                consistency_level=_STRONG,
            ),
        )
        return [list(group) for group in found]

    @staticmethod
    def _hits_of(
        groups: list[list[dict[str, Any]]], query: SearchQuery, *, distance: Distance
    ) -> list[ScoredPoint]:
        """Convert one Milvus result group into vendor-neutral scored points."""
        if not groups:
            return []
        return [
            ScoredPoint(
                point_id=str(hit.get(PK_FIELD, "")),
                score=_score_of(distance, float(hit.get("distance", 0.0))),
                payload=dict((hit.get("entity") or {}).get(PAYLOAD_FIELD) or {})
                if query.with_payload
                else {},
                vector=_floats((hit.get("entity") or {}).get(DENSE_FIELD))
                if query.with_vectors
                else None,
            )
            for hit in groups[0]
        ]

    async def _search_dense(self, target: _Collection, query: SearchQuery) -> list[ScoredPoint]:
        """Run the dense leg, pushing any metadata filter into Milvus."""
        groups = await self._run_search(
            target,
            query,
            data=[[float(value) for value in query.vector or []]],
            field=DENSE_FIELD,
            metric=_METRICS[target.distance],
            expression=to_milvus_filter(query.filters),
        )
        return self._hits_of(groups, query, distance=target.distance)

    async def _search_sparse(self, target: _Collection, query: SearchQuery) -> list[ScoredPoint]:
        """Run the keyword leg, with IDF carried into the query vector.

        A sparse field is scored by inner product whatever the dense leg's distance is,
        so these hits are converted on that scale rather than the collection's.
        """
        if not target.sparse:
            raise FasterRagError(
                f"collection {target.name!r} has no sparse index, so it cannot serve a "
                "keyword search; recreate it with sparse enabled",
                code=ErrorCode.VALIDATION_FAILED,
            )
        sparse = query.sparse
        if sparse is None or sparse.empty:
            return []

        weighted = await self._weighted_query(target, sparse)
        if not weighted:
            return []

        groups = await self._run_search(
            target,
            query,
            data=[weighted],
            field=SPARSE_FIELD,
            metric=_SPARSE_METRIC,
            expression=to_milvus_filter(query.filters),
        )
        hits = self._hits_of(groups, query, distance="dot")
        return [hit for hit in hits if hit.score > 0.0]

    async def _weighted_query(self, target: _Collection, sparse: SparseVector) -> dict[int, float]:
        """Turn a flat query vector into an IDF-weighted one, counting on the server.

        ``df`` is one ``count(*)`` per query term, so the corpus statistic comes from the
        store that holds the corpus without any posting ever reaching this process.
        """
        total = await self._count(target.physical)
        if total == 0:
            return {}

        weighted: dict[int, float] = {}
        for index, weight in zip(sparse.indices, sparse.values, strict=True):
            documents = await self._count(target.physical, _term_clause(int(index)))
            if documents:
                weighted[int(index)] = float(weight) * _idf(total, documents)
        return weighted

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors.

        Milvus has no partial update, so each point is read back whole, merged, and
        rewritten; a concurrent writer to the same point can lose one of the two merges.
        """
        if not updates:
            return

        grouped: dict[str, list[PointUpdate]] = {}
        for entry in updates:
            grouped.setdefault(entry.collection, []).append(entry)

        client = await self._ensure_client()
        for name, group in grouped.items():
            target = await self._require(name)
            for start in range(0, len(group), _COPY_BATCH):
                page = group[start : start + _COPY_BATCH]
                rows = await self._merged(target, page)
                if rows:
                    await self._call("update", _writer(client, target.physical, rows))

    async def _merged(self, target: _Collection, page: list[PointUpdate]) -> list[dict[str, Any]]:
        """Read the named points back whole and return them with their payloads merged."""
        client = await self._ensure_client()
        keys = [entry.point_id for entry in page]
        fields = [PK_FIELD, DENSE_FIELD, PAYLOAD_FIELD]
        if target.sparse:
            fields.extend((SPARSE_FIELD, TERMS_FIELD))

        stored = await self._call(
            "update",
            lambda: client.get(
                collection_name=target.physical,
                ids=keys,
                output_fields=fields,
                consistency_level=_STRONG,
            ),
        )
        held = {str(row.get(PK_FIELD, "")): dict(row) for row in stored}

        merged: list[dict[str, Any]] = []
        for entry in page:
            row = held.get(entry.point_id)
            if row is None:
                raise FasterRagError(
                    f"point {entry.point_id!r} is not in collection {target.name!r}, so "
                    "there is nothing to merge metadata into",
                    code=ErrorCode.NOT_FOUND,
                )
            row[PAYLOAD_FIELD] = {**(row.get(PAYLOAD_FIELD) or {}), **entry.payload}
            merged.append(row)
        return merged

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points, and their postings with them."""
        target = await self._require(selector.collection)
        client = await self._ensure_client()

        if selector.point_ids is not None:
            keys = list(selector.point_ids)
            await self._call(
                "delete", lambda: client.delete(collection_name=target.physical, ids=keys)
            )
            return

        expression = to_milvus_filter(selector.filters)
        if not expression:
            raise FasterRagError(
                "a delete filter must select something; refusing to delete a whole "
                "collection through an empty filter",
                code=ErrorCode.VALIDATION_FAILED,
            )
        await self._call(
            "delete",
            lambda: client.delete(collection_name=target.physical, filter=expression),
        )

    async def health(self) -> HealthStatus:
        """Report reachability without raising, so probes can render the failure."""
        started = time.perf_counter()
        try:
            client = await self._ensure_client()
            await self._call("health", client.get_server_version)
        except FasterRagError as exc:
            _logger.warning(
                "vector database health check failed",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return HealthStatus(healthy=False, detail=exc.detail)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return HealthStatus(healthy=True, latency_ms=round(elapsed_ms, 3))

    async def close(self) -> None:
        """Release the gRPC channel."""
        client = self._client
        self._client = None
        self._described.clear()
        self._loaded.clear()
        if client is not None:
            await self._call("close", client.close)


def _bound(work: Callable[..., Any], *arguments: Any) -> Callable[[], Any]:
    """Bind one client call to its arguments, so a loop variable cannot be captured late."""
    return lambda: work(*arguments)


def _alias_names(reported: object) -> list[str]:
    """Read alias names out of a listing.

    ``MilvusClient.list_aliases`` is annotated as returning a list but hands back the
    handler's ``{"aliases": [...]}`` dict, so both shapes are accepted.
    """
    if isinstance(reported, Mapping):
        return [str(name) for name in reported.get("aliases") or []]
    if isinstance(reported, Sequence) and not isinstance(reported, str | bytes):
        return [str(name) for name in reported]
    return []


def _row_for(entry: Point, target: _Collection) -> dict[str, Any]:
    """Build the Milvus row one point is written as, postings included."""
    row: dict[str, Any] = {
        PK_FIELD: entry.point_id,
        DENSE_FIELD: [float(value) for value in entry.vector],
        PAYLOAD_FIELD: dict(entry.payload),
    }
    if target.sparse:
        sparse = entry.sparse
        postings = (
            {}
            if sparse is None
            else {
                int(index): float(weight)
                for index, weight in zip(sparse.indices, sparse.values, strict=True)
            }
        )
        row[SPARSE_FIELD] = postings
        row[TERMS_FIELD] = sorted(postings)
    return row


def _writer(client: Any, physical: str, rows: list[dict[str, Any]]) -> Callable[[], None]:
    """Bind one page write to its arguments, so a worker thread can run it unchanged."""
    page = list(rows)
    return lambda: client.upsert(collection_name=physical, data=page)
