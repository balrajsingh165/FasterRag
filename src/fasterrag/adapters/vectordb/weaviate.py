r"""Weaviate adapter — the contract proved against a store with its own opinion of search.

Qdrant is the reference implementation (``docs/adr/ADR-0001``), pgvector proved the
contract holds over SQL, Chroma proved it holds with no server at all, and Milvus proved
it holds over a declared schema. Weaviate is the fifth paradigm, and the first one that
already has a *competing* answer to the question the contract asks: it ships its own BM25,
its own hybrid fusion, and its own tokenizer, none of which this adapter uses. Why not is
the most important thing in this file, so it is the first thing said.

**The keyword leg stores fasterRag's own saturated weights, and Weaviate's native BM25 is
deliberately unused.** Three independent reasons, any one of which is sufficient:

1. *The adapter never sees the text.* ``search()`` is handed a ``SparseVector`` — term
   indices from ``core.retrieval.bm25.term_index`` paired with weights — not a query
   string. ``query.bm25(query=...)`` wants a string. The only string that could be
   reconstructed is the list of 32-bit term hashes, which is not the user's query.
2. *The tokenizer would differ.* ``docs/adr/ADR-0007`` makes tokenization and saturation
   fasterRag's half of the split precisely so one query hits the same terms on every
   backend. Weaviate's ``word`` tokenizer and its own stopword list would hit different
   ones, so the same corpus would answer the same keyword query differently here than on
   Qdrant — the drift the ADR exists to prevent.
3. *It would be structurally unexportable.* Weaviate's inverted index is internal: there is
   no read path that hands back the per-document term weights it scored with. A collection
   whose keyword leg cannot be read cannot be carried, which is exactly the defect TASK-0291
   fixed everywhere else. ``iterate_points`` must return the leg, so the leg must live
   somewhere this adapter can read.

A tempting middle road was considered and rejected on evidence rather than taste: store our
stemmed tokens in a ``TEXT`` property, let Weaviate's ``word`` tokenizer split them, and set
``bm25_k1``/``bm25_b`` on the collection so the server's arithmetic matches ours. It cannot
work. ``encode_document`` stores *saturated* term frequencies, and saturation is not
invertible — the raw count a native BM25 index needs cannot be recovered from the float we
hold. Weaviate would also normalise by the real average document length while
``encode_document`` normalises by a fixed constant, so even a reconstructed count would
score differently. The round trip is lossy in principle, not merely in practice.

So postings are two ordinary array properties, ``fr_terms`` (``INT_ARRAY``, filterable) and
``fr_weights`` (``NUMBER_ARRAY``), written ascending by term. ``iterate_points`` hands back
exactly the weights ``encode_document`` wrote, in the one shape all backends agree on, and
IDF is computed here over the live corpus — the Chroma split, for the same reason Chroma
needed it. **The honest limit is Chroma's too**: a keyword search reads every document
carrying any query term into this process. Weaviate pushes the candidate filter and the
corpus count down, but not the scoring, so this does not scale the way Qdrant's server-side
sparse index does. Paging is unbounded on purpose — truncating the candidate set would
silently return a wrong top-k, so a corpus large enough to exceed Weaviate's
``QUERY_MAXIMUM_RESULTS`` fails loudly instead.

**Layout.** Everything is a Weaviate collection:

* One per fasterRag collection, named ``Frag_<slug>_<digest>``. The derivation exists
  because a Weaviate collection name must match ``[A-Z][_0-9A-Za-z]*`` while a fasterRag
  name may contain ``-`` and start with a digit; the client also silently capitalises the
  first letter, so a name that is already capitalised is the only one that round-trips.
  The logical name, the distance, and the **declared dimension count** live in the
  collection's ``description``.
* One per snapshot, ``Fragsnap_<slug>_<digest>``, tagged with the collection it came from.
* One, ``Fasterrag_aliases``, holding one record per alias.

**Dimensions are recorded because Weaviate does not declare them.** With a self-provided
vector, HNSW infers its width from the first object written, so an 8-dimension vector
entering a 768-dimension collection is a server-side error at best and a second index at
worst. The width is written into the description at creation and checked here on every
upsert, which is what makes ``test_dimension_mismatch_is_rejected`` a typed ``EmbedError``
rather than whatever the server says.

**Point ids become UUIDv5.** A Weaviate object id is a UUID and a fasterRag chunk id is not,
so the id is derived deterministically — the same chunk id always yields the same object id,
which is what keeps upserts idempotent — and the original is stored in ``fr_id``, which is
the only id that ever crosses back out of this module.

**Aliases are records, not Weaviate's native aliases.** Weaviate grew a real alias primitive
in 1.32, and using it would have been the prettier answer. It is not used because it would
make this adapter work on one server version and fail on another with no way to tell at
configuration time, while a record in a collection depends on nothing but object CRUD,
which every other method here already exercises. A single-record upsert replaces the target
in place, so there is no instant in which the alias resolves to nothing. The cost is one id
lookup per operation that resolves a name.

**Snapshots are copies into a second collection**, like pgvector's, Chroma's and Milvus's.
Weaviate's own backup API is not used because it requires a backup module to be enabled in
the *server's* environment (``ENABLE_MODULES=backup-filesystem`` and a configured path);
an adapter that silently fails ``snapshot()`` on every deployment that did not set that is
worse than one that always works. The honest cost is the same as the other three: a copy
living in the same cluster as the data it protects survives a bad delete, not the loss of
that cluster, and cannot be shipped elsewhere as a file.

**Payloads are flattened, with a lossless remainder.** A payload key that is a legal
Weaviate property name, does not collide with the reserved ``fr_`` namespace, and holds a
scalar becomes a top-level property — which is what makes metadata filters push down.
Everything else — nested objects, lists, nulls, reserved or illegal names — is JSON-encoded
into ``fr_extra`` and decoded back on read, so the payload round-trips exactly. A filter on
a key that could not be flattened is **refused** with a typed error naming it, rather than
pushed down against a property that does not exist and silently matching nothing. Two costs,
neither hidden: flattened properties rely on the server's ``AUTOSCHEMA_ENABLED`` (the
default, and a loud failure when an operator turns it off), and a string payload key gets
Weaviate's default searchable inverted index, including the chunk text this adapter never
queries.

**The client is synchronous, so every call runs off the event loop** through
``asyncio.to_thread``. ``WeaviateAsyncClient`` exists, but the sync client is what the
``iterator`` paging helper is built on, and a hand-rolled cursor walk is precisely the thing
that silently skips rows.

**Remaining limits of this first cut**, none of them silent: ``update()`` is a
read-modify-write because a payload merge has to rewrite ``fr_extra`` whole, so a concurrent
writer to the same point can lose a merge; and consistency is Weaviate's default (quorum on
a cluster, trivially strong on a single node), because the contract's read-after-write
expectation is met by a single node and a cluster deployment has never been exercised here.
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
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances
    from weaviate.classes.data import DataObject
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout
    from weaviate.classes.query import Filter as WeaviateFilter
    from weaviate.classes.query import MetadataQuery
    from weaviate.exceptions import WeaviateBaseError

    _VENDOR_ERRORS: tuple[type[BaseException], ...] = (WeaviateBaseError, OSError, ValueError)
    _DRIVER_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    _DRIVER_IMPORT_ERROR = exc

__all__ = [
    "ALIAS_COLLECTION",
    "ALIAS_PROPERTY",
    "EXTRA_PROPERTY",
    "INSTALL_HINT",
    "PK_PROPERTY",
    "TARGET_PROPERTY",
    "TERMS_PROPERTY",
    "WEIGHTS_PROPERTY",
    "WeaviateAdapter",
    "alias_uuid_for",
    "collection_name_for",
    "point_uuid_for",
    "snapshot_name_for",
    "to_weaviate_filter",
]

INSTALL_HINT: Final = 'pip install "fasterrag[weaviate]"'

# CRITICAL: these six property names, the reserved prefix, the three collection-name
# prefixes, and the two UUID namespaces are the on-server contract of a fasterRag Weaviate
# database. Changing a property name does not migrate anything — it makes every object
# already written unreadable, with no error until a walk returns empty payloads. Changing a
# namespace changes every derived object id, so re-ingesting an unchanged corpus writes a
# second copy of it beside the first instead of overwriting it.
PK_PROPERTY: Final = "fr_id"
TERMS_PROPERTY: Final = "fr_terms"
WEIGHTS_PROPERTY: Final = "fr_weights"
EXTRA_PROPERTY: Final = "fr_extra"
ALIAS_PROPERTY: Final = "fr_alias"
TARGET_PROPERTY: Final = "fr_target"
RESERVED_PREFIX: Final = "fr_"
_NIL_UUID: Final = "00000000-0000-0000-0000-000000000000"

_COLLECTION_PREFIX: Final = "Frag"
_SNAPSHOT_PREFIX: Final = "Fragsnap"
ALIAS_COLLECTION: Final = "Fasterrag_aliases"

_POINT_NAMESPACE: Final = uuid.UUID("5a0d7f6e-9c21-5f3a-8f44-2b1d6c0e9a77")
_ALIAS_NAMESPACE: Final = uuid.UUID("c3f1b2a4-7d58-5e60-9a13-8e4f0b7c2d91")

_LAYOUT_VERSION: Final = 1
_KIND_KEY: Final = "k"
_NAME_KEY: Final = "n"
_SOURCE_KEY: Final = "s"
_DISTANCE_KEY: Final = "d"
_DIMENSIONS_KEY: Final = "m"
_VERSION_KEY: Final = "v"

_COLLECTION_KIND: Final = "c"
_SNAPSHOT_KIND: Final = "s"

_WRITE_BATCH: Final = 200
_PAGE_SIZE: Final = 500
_ALIAS_VECTOR: Final[tuple[float, ...]] = (0.0,)

_SLUG_LIMIT: Final = 32
_DIGEST_LIMIT: Final = 12
_UNSAFE_NAME_CHARS: Final = re.compile(r"[^a-z0-9]+")
_LEGAL_PROPERTY: Final = re.compile(r"^[A-Za-z][_0-9A-Za-z]*$")

_DISTANCES: Final[dict[str, Distance]] = {"cosine": "cosine", "dot": "dot", "euclid": "euclid"}

_UNREACHABLE_ERRORS: Final[frozenset[str]] = frozenset(
    {"WeaviateConnectionError", "WeaviateGRPCUnavailableError", "WeaviateTimeoutError"}
)
_NOT_FOUND_PHRASES: Final[tuple[str, ...]] = ("not found", "could not find", "does not exist")
_CONFLICT_PHRASES: Final[tuple[str, ...]] = ("already exists", "already exist")
_AUTH_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})
_AUTH_PHRASES: Final[tuple[str, ...]] = (
    "anonymous access",
    "forbidden",
    "insufficient permissions",
    "invalid api key",
    "unauthenticated",
    "unauthorized",
)

_logger = get_logger(__name__)

_T = TypeVar("_T")


def _slug(value: str) -> str:
    """Return the readable half of a derived Weaviate collection name."""
    cleaned = _UNSAFE_NAME_CHARS.sub("_", value.lower()).strip("_")
    return cleaned[:_SLUG_LIMIT].strip("_") or "collection"


def _digest(*parts: str) -> str:
    """Return the short stable hash that makes a derived name unique."""
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:_DIGEST_LIMIT]


def collection_name_for(collection: str) -> str:
    """Return the Weaviate collection a fasterRag collection's objects are stored in."""
    return f"{_COLLECTION_PREFIX}_{_slug(collection)}_{_digest(collection)}"


def snapshot_name_for(collection: str, snapshot: str) -> str:
    """Return the Weaviate collection a snapshot's objects are copied into."""
    return f"{_SNAPSHOT_PREFIX}_{_slug(snapshot)}_{_digest(collection, snapshot)}"


def point_uuid_for(point_id: str) -> str:
    """Return the deterministic object id one fasterRag point id maps to."""
    return str(uuid.uuid5(_POINT_NAMESPACE, point_id))


def alias_uuid_for(alias: str) -> str:
    """Return the deterministic object id one alias record is stored under."""
    return str(uuid.uuid5(_ALIAS_NAMESPACE, alias))


def _require_filterable(key: str) -> None:
    """Reject a filter on a key no object could have been stored under."""
    if key.startswith(RESERVED_PREFIX):
        raise FasterRagError(
            f"filter key {key!r} uses the reserved {RESERVED_PREFIX!r} namespace, which "
            "holds this adapter's own properties rather than payload",
            code=ErrorCode.VALIDATION_FAILED,
        )
    if not _LEGAL_PROPERTY.match(key):
        raise FasterRagError(
            f"filter key {key!r} is not a legal Weaviate property name, so it is stored "
            "inside the JSON remainder rather than as a filterable property; rename the "
            "payload key to match ^[A-Za-z][_0-9A-Za-z]*$ to filter on it",
            code=ErrorCode.VALIDATION_FAILED,
        )


def _comparison(key: str, operator: str, value: Any) -> Any:
    """Translate one supported filter operator into a Weaviate filter node.

    ``$in``/``$nin`` expand into ``equal`` disjunctions rather than ``contains_any``,
    which Weaviate defines over array properties rather than the scalars we store.
    """
    built = WeaviateFilter.by_property(key)
    if operator in ("$in", "$nin"):
        if not isinstance(value, list):
            raise FasterRagError(
                f"filter on {key!r} with {operator} requires a list of values",
                code=ErrorCode.VALIDATION_FAILED,
            )
        if operator == "$in":
            return WeaviateFilter.any_of(
                [WeaviateFilter.by_property(key).equal(member) for member in value]
            )
        return WeaviateFilter.all_of(
            [WeaviateFilter.by_property(key).not_equal(member) for member in value]
        )

    operators: dict[str, Callable[[Any], Any]] = {
        "$eq": built.equal,
        "$ne": built.not_equal,
        "$gt": built.greater_than,
        "$gte": built.greater_or_equal,
        "$lt": built.less_than,
        "$lte": built.less_or_equal,
    }
    return operators[operator](value)


def _everything() -> Any:
    """Return a filter matching every object in a collection, whatever it holds.

    Matching on the id keeps this independent of any property's tokenizer; the nil UUID
    can never collide because every object id here is a UUIDv5.
    """
    # CRITICAL: this must not be `by_property(PK_PROPERTY).not_equal("")`. A real Weaviate
    # tokenizes the empty string to nothing and refuses the whole batch delete with
    # "invalid search term, only stopwords provided", which broke every restore_snapshot
    # against a server (measured 2026-11-11) while the unit double matched it happily.
    return WeaviateFilter.by_id().not_equal(_NIL_UUID)


def to_weaviate_filter(filters: Filter | None) -> Any:
    """Translate a vendor-neutral filter expression into a Weaviate filter, or None."""
    validate_filter(filters)
    if not filters:
        return None

    clauses: list[Any] = []
    for key, condition in filters.items():
        _require_filterable(key)
        if isinstance(condition, Mapping):
            operator = next(iter(condition))
            clauses.append(_comparison(key, operator, condition[operator]))
        else:
            clauses.append(_comparison(key, "$eq", condition))

    return clauses[0] if len(clauses) == 1 else WeaviateFilter.all_of(clauses)


def _terms_filter(terms: Sequence[int]) -> Any:
    """Build the filter matching every object whose posting list holds one of ``terms``."""
    return WeaviateFilter.by_property(TERMS_PROPERTY).contains_any([int(term) for term in terms])


def _idf(total: int, documents: int) -> float:
    """Return the BM25 inverse document frequency for a term, as the other backends do."""
    return math.log(1 + (total - documents + 0.5) / (documents + 0.5))


def _score_of(distance: Distance, measured: float) -> float:
    """Turn a Weaviate distance into the higher-is-better score the contract requires.

    Weaviate reports cosine *distance*, a negated dot product, and squared L2, so only
    the cosine leg has a similarity to recover.
    """
    if distance == "cosine":
        return 1.0 - measured
    return -measured


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


def _metric_of(distance: Distance) -> Any:
    """Return the Weaviate distance metric one neutral distance maps to."""
    metrics: dict[Distance, Any] = {
        "cosine": VectorDistances.COSINE,
        "dot": VectorDistances.DOT,
        "euclid": VectorDistances.L2_SQUARED,
    }
    return metrics[distance]


def _flattenable(key: str, value: Any) -> bool:
    """Report whether one payload entry can live as a filterable Weaviate property."""
    if key.startswith(RESERVED_PREFIX) or not _LEGAL_PROPERTY.match(key):
        return False
    return isinstance(value, str | bool | int | float)


def _properties_for(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Split a payload into flattened properties plus a lossless JSON remainder."""
    flattened: dict[str, Any] = {}
    remainder: dict[str, Any] = {}
    for key, value in payload.items():
        if _flattenable(key, value):
            flattened[key] = value
        else:
            remainder[key] = value
    flattened[EXTRA_PROPERTY] = json.dumps(remainder, separators=(",", ":")) if remainder else ""
    return flattened


def _payload_of(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the original payload from an object's properties."""
    payload = {
        key: value for key, value in properties.items() if not key.startswith(RESERVED_PREFIX)
    }
    remainder = properties.get(EXTRA_PROPERTY)
    if isinstance(remainder, str) and remainder:
        decoded = json.loads(remainder)
        if not isinstance(decoded, dict):
            raise FasterRagError(
                f"the {EXTRA_PROPERTY} property of an object does not hold a JSON object, "
                "so its payload cannot be reconstructed",
                code=ErrorCode.INTERNAL,
            )
        payload.update(decoded)
    return payload


def _floats(stored: object) -> list[float]:
    """Convert a returned dense vector into plain floats, so no vendor type escapes."""
    if stored is None:
        return []
    if isinstance(stored, Mapping):
        chosen = stored.get("default")
        if chosen is None:
            values = list(stored.values())
            chosen = values[0] if len(values) == 1 else None
        return _floats(chosen)
    if isinstance(stored, Sequence) and not isinstance(stored, str | bytes):
        return [float(value) for value in stored]
    raise FasterRagError(
        f"weaviate returned a dense vector as {type(stored).__name__}, which this adapter "
        "cannot read as a sequence of floats",
        code=ErrorCode.INTERNAL,
    )


def _to_sparse(properties: Mapping[str, Any]) -> SparseVector | None:
    """Convert the two posting properties into the neutral type, terms ascending."""
    terms = properties.get(TERMS_PROPERTY)
    weights = properties.get(WEIGHTS_PROPERTY)
    if not isinstance(terms, Iterable) or not isinstance(weights, Iterable):
        return None

    listed = [int(term) for term in terms]
    valued = [float(weight) for weight in weights]
    if not listed:
        return None
    if len(listed) != len(valued):
        raise FasterRagError(
            f"an object carries {len(listed)} terms but {len(valued)} weights, so its "
            "keyword leg cannot be read back; the object was not written by this adapter "
            "or was modified outside it",
            code=ErrorCode.INTERNAL,
        )

    postings = sorted(zip(listed, valued, strict=True))
    return SparseVector(
        indices=[term for term, _ in postings], values=[weight for _, weight in postings]
    )


def _description_of(spec: CollectionSpec, **extra: str) -> str:
    """Build the JSON description carrying what a Weaviate schema cannot record itself."""
    recorded: dict[str, Any] = {
        _VERSION_KEY: _LAYOUT_VERSION,
        _NAME_KEY: spec.name,
        _DISTANCE_KEY: spec.distance,
        _DIMENSIONS_KEY: spec.dimensions,
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
    """One resolved collection: its logical name, its Weaviate name, and its shape."""

    name: str
    physical: str
    dimensions: int
    distance: Distance
    sparse: bool


class WeaviateAdapter(VectorDBAdapter):
    """Weaviate implementation of the vector database contract."""

    def __init__(self, settings: Settings) -> None:
        """Build the adapter. The HTTP and gRPC channels open on first use."""
        if _DRIVER_IMPORT_ERROR is not None:
            raise ConfigError(
                "vector_db.provider is 'weaviate' but its driver is not installed "
                f"({_DRIVER_IMPORT_ERROR}); install it with: {INSTALL_HINT}"
            )

        vector_db = settings.vector_db
        self._host = vector_db.host
        self._port = vector_db.port
        self._grpc_port = vector_db.grpc_port
        self._secure = vector_db.https
        self._token_env = vector_db.api_key_env
        self._timeout_seconds = max(1.0, settings.reliability.timeouts.vector_db_ms / 1000)
        self._endpoint = f"{'https' if self._secure else 'http'}://{self._host}:{self._port}"
        self._client: Any = None
        self._described: dict[str, _Collection] = {}
        self._aliases_ready = False
        self._lock = asyncio.Lock()

    def _auth_error(self, operation: str) -> ProviderError:
        """Build the non-retryable error for a rejected credential."""
        named = self._token_env or "vector_db.api_key_env"
        return ProviderError(
            f"weaviate rejected the credentials during {operation}; check the key in the "
            f"{named} environment variable and the server's authentication configuration",
            code=ErrorCode.VECTOR_DB_AUTH_FAILED,
            retryable=False,
        )

    def _translate(self, exc: BaseException, operation: str) -> FasterRagError:
        """Map a Weaviate failure onto the typed error taxonomy."""
        if isinstance(exc, OSError):
            return ProviderError(
                f"weaviate was unreachable during {operation}: {type(exc).__name__}",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        if not isinstance(exc, WeaviateBaseError):
            return FasterRagError(
                f"weaviate rejected the request during {operation}: {exc}",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return self._translate_server(exc, operation)

    def _translate_server(self, exc: BaseException, operation: str) -> FasterRagError:
        """Map one ``WeaviateBaseError`` onto the taxonomy by what the server said."""
        message = str(getattr(exc, "message", "") or exc).lower()
        named = type(exc).__name__

        # CRITICAL: the status code is checked before the prose because a real Weaviate
        # refusing a credential says neither "unauthorized" nor "forbidden". Against an
        # anonymous-access server it answers 401 with "oidc auth is not configured", which
        # matched no phrase below and came back out as a RETRYABLE provider error — a
        # caller retrying a rejected credential forever (measured 2026-11-11).
        if getattr(exc, "status_code", None) in _AUTH_STATUS_CODES:
            return self._auth_error(operation)
        if named in ("AuthenticationFailedError", "InsufficientPermissionsError") or any(
            phrase in message for phrase in _AUTH_PHRASES
        ):
            return self._auth_error(operation)
        if named in _UNREACHABLE_ERRORS:
            return ProviderError(
                f"weaviate was unreachable during {operation}: {named}",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        if named == "ObjectAlreadyExistsError" or any(
            phrase in message for phrase in _CONFLICT_PHRASES
        ):
            return FasterRagError(
                f"weaviate reported a conflicting state during {operation}",
                code=ErrorCode.CONFLICT,
            )
        if any(phrase in message for phrase in _NOT_FOUND_PHRASES):
            return FasterRagError(
                f"weaviate reported a missing resource during {operation}",
                code=ErrorCode.NOT_FOUND,
            )
        if named in ("WeaviateInvalidInputError", "SchemaValidationError"):
            return FasterRagError(
                f"weaviate rejected the request during {operation}: {exc}",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return ProviderError(
            f"weaviate failed during {operation}: {named}",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=True,
        )

    async def _call(self, operation: str, work: Callable[[], _T]) -> _T:
        """Run one blocking Weaviate call off the loop, letting no vendor type escape."""
        try:
            return await asyncio.to_thread(work)
        except _VENDOR_ERRORS as exc:
            raise self._translate(exc, operation) from exc

    def _build_client(self) -> Any:
        """Open the connection, naming the credential by its environment variable."""
        token = os.environ.get(self._token_env, "") if self._token_env else ""
        return weaviate.connect_to_custom(
            http_host=self._host,
            http_port=self._port,
            http_secure=self._secure,
            grpc_host=self._host,
            grpc_port=self._grpc_port,
            grpc_secure=self._secure,
            auth_credentials=Auth.api_key(token) if token else None,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    init=self._timeout_seconds,
                    query=self._timeout_seconds,
                    insert=self._timeout_seconds,
                )
            ),
        )

    async def _ensure_client(self) -> Any:
        """Return the client, connecting on first use."""
        existing = self._client
        if existing is not None:
            return existing

        async with self._lock:
            if self._client is None:
                self._client = await self._call("connect", self._build_client)
            return self._client

    async def _handle(self, physical: str) -> Any:
        """Return the client's handle on one Weaviate collection, without a round trip."""
        client = await self._ensure_client()
        return client.collections.get(physical)

    async def _has(self, physical: str) -> bool:
        """Report whether Weaviate holds a collection under this derived name."""
        client = await self._ensure_client()
        return bool(await self._call("exists", lambda: client.collections.exists(physical)))

    async def _describe(self, physical: str) -> _Collection | None:
        """Read one Weaviate collection's description and declared properties."""
        cached = self._described.get(physical)
        if cached is not None:
            return cached

        if not await self._has(physical):
            return None

        handle = await self._handle(physical)
        raw = await self._call("describe", lambda: handle.config.get(simple=True))
        described = self._shape_of(physical, raw)
        if described is not None:
            self._described[physical] = described
        return described

    @staticmethod
    def _shape_of(physical: str, raw: Any) -> _Collection | None:
        """Turn one collection configuration into the adapter's view of it."""
        recorded = _recorded(str(getattr(raw, "description", "") or ""))
        if recorded.get(_KIND_KEY) not in (_COLLECTION_KIND, _SNAPSHOT_KIND):
            return None

        declared = {str(getattr(entry, "name", "")) for entry in getattr(raw, "properties", [])}
        return _Collection(
            name=str(recorded.get(_NAME_KEY, "")),
            physical=physical,
            dimensions=int(recorded.get(_DIMENSIONS_KEY, 0)),
            distance=_distance_of(str(recorded.get(_DISTANCE_KEY, "cosine"))),
            sparse=TERMS_PROPERTY in declared,
        )

    async def _resolve(self, name: str) -> _Collection | None:
        """Resolve a collection name or an alias, a real name winning over an alias."""
        direct = await self._describe(collection_name_for(name))
        if direct is not None:
            return direct

        target = await self._alias_collection(name)
        return None if target is None else await self._describe(collection_name_for(target))

    async def _require(self, name: str) -> _Collection:
        """Resolve a collection, raising rather than letting a typo read nothing."""
        found = await self._resolve(name)
        if found is None:
            raise FasterRagError(
                f"collection {name!r} does not exist on {self._endpoint}, and no alias "
                "points at one",
                code=ErrorCode.NOT_FOUND,
            )
        return found

    @staticmethod
    def _schema(spec: CollectionSpec) -> list[Any]:
        """Declare the properties one fasterRag collection needs, postings included."""
        properties = [
            Property(name=PK_PROPERTY, data_type=DataType.TEXT, index_filterable=True),
            Property(name=EXTRA_PROPERTY, data_type=DataType.TEXT, index_searchable=False),
        ]
        if spec.sparse:
            properties.append(
                Property(name=TERMS_PROPERTY, data_type=DataType.INT_ARRAY, index_filterable=True)
            )
            properties.append(
                Property(
                    name=WEIGHTS_PROPERTY,
                    data_type=DataType.NUMBER_ARRAY,
                    index_filterable=False,
                )
            )
        return properties

    async def _create(self, physical: str, spec: CollectionSpec, description: str) -> None:
        """Create one Weaviate collection with its properties and its vector index."""
        client = await self._ensure_client()
        properties = self._schema(spec)
        vectors = Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=_metric_of(spec.distance)
            )
        )
        await self._call(
            "create_collection",
            lambda: client.collections.create(
                name=physical,
                description=description,
                properties=properties,
                vector_config=vectors,
                sharding_config=Configure.sharding(desired_count=spec.shard_number),
                replication_config=Configure.replication(factor=spec.replication_factor),
            ),
        )
        self._described.pop(physical, None)

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create the Weaviate collection, or verify an existing one is compatible."""
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
                f"collection {spec.name!r} already exists without the posting properties a "
                "keyword leg needs; rebuild it through a blue/green reindex rather than "
                "adding them to a collection whose objects do not carry them",
                code=ErrorCode.CONFLICT,
            )

    async def _owned(self, prefix: str) -> list[_Collection]:
        """Return every Weaviate collection this adapter owns under one name prefix."""
        client = await self._ensure_client()
        listed = await self._call(
            "list_collections", lambda: client.collections.list_all(simple=True)
        )
        found: list[_Collection] = []
        for physical in sorted(str(name) for name in listed):
            if not physical.startswith(f"{prefix}_"):
                continue
            described = self._shape_of(physical, listed[physical])
            if described is not None:
                self._described.setdefault(physical, described)
                found.append(described)
        return found

    async def _count(self, physical: str, where: Any = None) -> int:
        """Return how many objects a collection holds, counted on the server."""
        handle = await self._handle(physical)
        found = await self._call(
            "count", lambda: handle.aggregate.over_all(total_count=True, filters=where)
        )
        return int(getattr(found, "total_count", 0) or 0)

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection the database holds, with an object count."""
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
        """Drop one Weaviate collection and forget everything cached about it."""
        client = await self._ensure_client()
        await self._call("drop_collection", lambda: client.collections.delete(physical))
        self._described.pop(physical, None)

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, its snapshots, and any alias pointing at it."""
        physical = collection_name_for(name)
        described = await self._describe(physical)
        if described is None:
            return False

        for snapshot in await self.list_snapshots(name):
            await self._drop(snapshot_name_for(described.name, snapshot))

        for alias in await self._aliases_pointing_at(described.name):
            await self.delete_alias(alias)

        await self._drop(physical)
        return True

    async def snapshot(self, collection: str) -> str:
        """Copy a collection's objects into a second collection and name the copy."""
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
        """Copy every object of one collection into another, one page at a time."""
        page: list[Any] = []
        async for found in self._walk(source, with_vectors=True, batch_size=_WRITE_BATCH):
            page.append(
                DataObject(
                    properties=dict(found.properties),
                    uuid=str(found.uuid),
                    vector=_floats(found.vector),
                )
            )
            if len(page) >= _WRITE_BATCH:
                await self._write(destination, page)
                page = []
        if page:
            await self._write(destination, page)

    async def list_snapshots(self, collection: str) -> list[str]:
        """Return the snapshots held for a collection, newest last."""
        target = await self._require(collection)
        client = await self._ensure_client()
        listed = await self._call(
            "list_snapshots", lambda: client.collections.list_all(simple=True)
        )

        held: list[str] = []
        for physical in sorted(str(name) for name in listed):
            if not physical.startswith(f"{_SNAPSHOT_PREFIX}_"):
                continue
            recorded = _recorded(str(getattr(listed[physical], "description", "") or ""))
            if recorded.get(_SOURCE_KEY) == target.name:
                held.append(str(recorded.get(_NAME_KEY, "")))
        return sorted(held)

    async def _snapshot_of(self, collection: str, snapshot: str) -> _Collection | None:
        """Resolve one snapshot of a collection to the Weaviate collection holding it."""
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

        handle = await self._handle(target.physical)
        await self._call(
            "restore_snapshot",
            lambda: handle.data.delete_many(where=_everything()),
        )
        await self._copy(held, target.physical)

    async def _ensure_alias_collection(self) -> Any:
        """Return the alias collection, creating it on first use."""
        if not self._aliases_ready:
            async with self._lock:
                if not self._aliases_ready:
                    if not await self._has(ALIAS_COLLECTION):
                        client = await self._ensure_client()
                        await self._call(
                            "set_alias",
                            lambda: client.collections.create(
                                name=ALIAS_COLLECTION,
                                description="fasterRag alias records",
                                properties=[
                                    Property(name=ALIAS_PROPERTY, data_type=DataType.TEXT),
                                    Property(name=TARGET_PROPERTY, data_type=DataType.TEXT),
                                ],
                                vector_config=Configure.Vectors.self_provided(),
                            ),
                        )
                    self._aliases_ready = True
        return await self._handle(ALIAS_COLLECTION)

    async def _alias_collection(self, alias: str) -> str | None:
        """Return the collection name a fasterRag alias record points at, if any."""
        if not await self._has(ALIAS_COLLECTION):
            return None

        handle = await self._handle(ALIAS_COLLECTION)
        key = alias_uuid_for(alias)
        found = await self._call("alias_target", lambda: handle.query.fetch_object_by_id(key))
        if found is None:
            return None
        target = dict(found.properties).get(TARGET_PROPERTY)
        return None if target is None else str(target)

    async def _aliases_pointing_at(self, collection: str) -> list[str]:
        """Return every alias whose record names ``collection``."""
        if not await self._has(ALIAS_COLLECTION):
            return []

        handle = await self._handle(ALIAS_COLLECTION)
        found = await self._call(
            "drop_collection",
            lambda: handle.query.fetch_objects(
                filters=WeaviateFilter.by_property(TARGET_PROPERTY).equal(collection),
                limit=_PAGE_SIZE,
            ),
        )
        return [str(dict(entry.properties).get(ALIAS_PROPERTY, "")) for entry in found.objects]

    async def set_alias(self, alias: str, collection: str) -> None:
        """Point an alias at a collection by replacing its one record in place."""
        target = await self._require(collection)
        if await self._describe(collection_name_for(alias)) is not None:
            raise FasterRagError(
                f"{alias!r} is already a collection, so it cannot also be an alias; "
                "every query would resolve to the collection instead",
                code=ErrorCode.CONFLICT,
            )

        await self._ensure_alias_collection()
        await self._write(
            ALIAS_COLLECTION,
            [
                DataObject(
                    properties={ALIAS_PROPERTY: alias, TARGET_PROPERTY: target.name},
                    uuid=alias_uuid_for(alias),
                    vector=list(_ALIAS_VECTOR),
                )
            ],
        )

    async def alias_target(self, alias: str) -> str | None:
        """Return the collection an alias resolves to."""
        return await self._alias_collection(alias)

    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, leaving the collection it pointed at in place."""
        if await self._alias_collection(alias) is None:
            return False
        handle = await self._handle(ALIAS_COLLECTION)
        key = alias_uuid_for(alias)
        await self._call("delete_alias", lambda: handle.data.delete_by_id(key))
        return True

    async def _write(self, physical: str, objects: list[Any]) -> None:
        """Write one page, replacing object by object whatever the batch refused.

        The batch endpoint is expected to overwrite an object already holding the id;
        the fallback makes the upsert idempotent whether or not it does.
        """
        handle = await self._handle(physical)
        page = list(objects)
        written = await self._call("upsert", lambda: handle.data.insert_many(page))
        errors = dict(getattr(written, "errors", {}) or {})
        for position, failure in errors.items():
            entry = page[int(position)]
            await self._replace(handle, entry, str(getattr(failure, "message", failure)))

    async def _replace(self, handle: Any, entry: Any, reason: str) -> None:
        """Rewrite one object the batch would not insert, keeping the failure's reason."""
        _logger.debug("replacing an object the batch refused", extra={"reason": reason[:200]})
        await self._call(
            "upsert",
            lambda: handle.data.replace(
                uuid=entry.uuid, properties=entry.properties, vector=entry.vector
            ),
        )

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points and their postings, overwriting any that already exist."""
        if not points:
            return UpsertResult(upserted=0)

        grouped: dict[str, list[Point]] = {}
        for entry in points:
            grouped.setdefault(entry.collection, []).append(entry)

        for name, group in grouped.items():
            target = await self._require(name)
            self._require_matching_dimensions(target, group)
            self._require_sparse_properties(target, group)
            for start in range(0, len(group), _WRITE_BATCH):
                page = [_object_for(entry, target) for entry in group[start : start + _WRITE_BATCH]]
                await self._write(target.physical, page)

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
    def _require_sparse_properties(target: _Collection, batch: list[Point]) -> None:
        """Reject postings a collection has no declared properties to hold."""
        for entry in batch:
            if entry.sparse is None or entry.sparse.empty:
                continue
            if not target.sparse:
                raise EmbedError(
                    f"point {entry.point_id!r} carries a sparse vector but collection "
                    f"{target.name!r} has no posting properties; recreate it with sparse "
                    "enabled",
                    retryable=False,
                )

    def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection through Weaviate's object iterator."""
        return self._iterate(collection, with_vectors=with_vectors, batch_size=batch_size)

    async def _iterate(
        self, collection: str, *, with_vectors: bool, batch_size: int
    ) -> AsyncIterator[Point]:
        """Walk a resolved collection, converting each object into the neutral type."""
        target = await self._require(collection)
        async for found in self._walk(target, with_vectors=with_vectors, batch_size=batch_size):
            properties = dict(found.properties)
            yield Point(
                point_id=str(properties.get(PK_PROPERTY, "")),
                collection=collection,
                vector=_floats(found.vector) if with_vectors else [],
                payload=_payload_of(properties),
                sparse=_to_sparse(properties) if with_vectors else None,
            )

    async def _walk(
        self, target: _Collection, *, with_vectors: bool, batch_size: int
    ) -> AsyncIterator[Any]:
        """Page every object of a collection off the event loop."""
        handle = await self._handle(target.physical)
        cursor = iter(
            await self._call(
                "iterate_points",
                lambda: handle.iterator(include_vector=with_vectors, cache_size=batch_size),
            )
        )

        # CRITICAL: advance the cursor with `next`, never `islice` or any other helper that
        # calls `iter()` on it. Weaviate's `_ObjectIterator.__iter__` RESETS the walk to the
        # first page, so re-entering it per page restarted the collection every time: against
        # a real server this never terminated and re-yielded the same objects forever
        # (measured 2026-11-11, 400+ gRPC round trips for a one-object collection). The unit
        # double's iterator does not reset, which is why 95% coverage missed it.
        def page_of(taking: int) -> list[Any]:
            """Take up to ``taking`` objects without re-entering the cursor's ``__iter__``."""
            taken: list[Any] = []
            while len(taken) < taking:
                found = next(cursor, None)
                if found is None:
                    break
                taken.append(found)
            return taken

        while True:
            page = await self._call("iterate_points", lambda: page_of(batch_size))
            if not page:
                return
            for found in page:
                yield found

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points on whichever leg the query carries."""
        if query.limit <= 0:
            return []

        target = await self._require(query.collection)
        if query.sparse is not None:
            return await self._search_sparse(target, query)
        return await self._search_dense(target, query)

    @staticmethod
    def _returned_properties(query: SearchQuery) -> list[str] | None:
        """Name the properties a search must bring back, or None for all of them."""
        return None if query.with_payload else [PK_PROPERTY]

    async def _search_dense(self, target: _Collection, query: SearchQuery) -> list[ScoredPoint]:
        """Run the dense leg, pushing any metadata filter into Weaviate."""
        handle = await self._handle(target.physical)
        where = to_weaviate_filter(query.filters)
        asked = [float(value) for value in query.vector or []]
        fields = self._returned_properties(query)

        found = await self._call(
            "search",
            lambda: handle.query.near_vector(
                near_vector=asked,
                limit=query.limit,
                filters=where,
                include_vector=query.with_vectors,
                return_metadata=MetadataQuery(distance=True),
                return_properties=fields,
            ),
        )
        return [
            ScoredPoint(
                point_id=str(dict(entry.properties).get(PK_PROPERTY, "")),
                score=_score_of(target.distance, float(entry.metadata.distance or 0.0)),
                payload=_payload_of(dict(entry.properties)) if query.with_payload else {},
                vector=_floats(entry.vector) if query.with_vectors else None,
            )
            for entry in found.objects
        ]

    async def _search_sparse(self, target: _Collection, query: SearchQuery) -> list[ScoredPoint]:
        """Run the keyword leg, computing IDF over the live corpus.

        Candidates are every object carrying a query term, because that set is what both
        the document frequencies and the scores are computed from.
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

        terms = _terms_filter(sparse.indices)
        candidates = await self._page(target, terms, with_vectors=query.with_vectors)
        if not candidates:
            return []

        total = await self._count(target.physical)
        allowed = await self._permitted(target, terms, query.filters)
        ranked = _rank(sparse, candidates, total=total, allowed=allowed)[: query.limit]

        held = {str(dict(entry.properties).get(PK_PROPERTY, "")): entry for entry in candidates}
        return [
            ScoredPoint(
                point_id=point_id,
                score=score,
                payload=_payload_of(dict(held[point_id].properties)) if query.with_payload else {},
                vector=_floats(held[point_id].vector) if query.with_vectors else None,
            )
            for point_id, score in ranked
        ]

    async def _permitted(
        self, target: _Collection, terms: Any, filters: Filter | None
    ) -> set[str] | None:
        """Return the ids a pushed-down payload filter allows, or None when there is none."""
        payload = to_weaviate_filter(filters)
        if payload is None:
            return None

        combined = WeaviateFilter.all_of([terms, payload])
        found = await self._page(target, combined, with_vectors=False)
        return {str(dict(entry.properties).get(PK_PROPERTY, "")) for entry in found}

    async def _page(self, target: _Collection, where: Any, *, with_vectors: bool) -> list[Any]:
        """Read every object matching a filter, paging until the server runs out.

        Deliberately unbounded: a truncated candidate set returns a wrong top-k with no
        sign of it, so an oversized one must fail rather than mislead.
        """
        handle = await self._handle(target.physical)
        collected: list[Any] = []
        offset = 0
        while True:
            found = await self._call(
                "search", _fetcher(handle, where, offset, with_vectors=with_vectors)
            )
            page = list(found.objects)
            collected.extend(page)
            if len(page) < _PAGE_SIZE:
                return collected
            offset += _PAGE_SIZE

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors.

        A payload is split between flattened properties and one JSON remainder, so the
        remainder has to be rewritten whole; a concurrent writer can lose one merge.
        """
        if not updates:
            return

        grouped: dict[str, list[PointUpdate]] = {}
        for entry in updates:
            grouped.setdefault(entry.collection, []).append(entry)

        for name, group in grouped.items():
            target = await self._require(name)
            handle = await self._handle(target.physical)
            for entry in group:
                await self._merge(handle, target, entry)

    async def _merge(self, handle: Any, target: _Collection, entry: PointUpdate) -> None:
        """Read one point back, merge its payload, and write the merged properties."""
        key = point_uuid_for(entry.point_id)
        stored = await self._call("update", lambda: handle.query.fetch_object_by_id(key))
        if stored is None:
            raise FasterRagError(
                f"point {entry.point_id!r} is not in collection {target.name!r}, so there "
                "is nothing to merge metadata into",
                code=ErrorCode.NOT_FOUND,
            )

        merged = {**_payload_of(dict(stored.properties)), **entry.payload}
        properties = {**_properties_for(merged), PK_PROPERTY: entry.point_id}
        await self._call("update", lambda: handle.data.update(uuid=key, properties=properties))

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points, and their postings with them."""
        target = await self._require(selector.collection)
        handle = await self._handle(target.physical)

        if selector.point_ids is not None:
            keys = [point_uuid_for(point_id) for point_id in selector.point_ids]
            if not keys:
                return
            await self._call(
                "delete",
                lambda: handle.data.delete_many(where=WeaviateFilter.by_id().contains_any(keys)),
            )
            return

        where = to_weaviate_filter(selector.filters)
        if where is None:
            raise FasterRagError(
                "a delete filter must select something; refusing to delete a whole "
                "collection through an empty filter",
                code=ErrorCode.VALIDATION_FAILED,
            )
        await self._call("delete", lambda: handle.data.delete_many(where=where))

    async def health(self) -> HealthStatus:
        """Report reachability without raising, so probes can render the failure."""
        started = time.perf_counter()
        try:
            client = await self._ensure_client()
            await self._call("health", client.get_meta)
        except FasterRagError as exc:
            _logger.warning(
                "vector database health check failed",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return HealthStatus(healthy=False, detail=exc.detail)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return HealthStatus(healthy=True, latency_ms=round(elapsed_ms, 3))

    async def close(self) -> None:
        """Release the HTTP and gRPC connections."""
        client = self._client
        self._client = None
        self._described.clear()
        self._aliases_ready = False
        if client is not None:
            await self._call("close", client.close)


def _fetcher(handle: Any, where: Any, offset: int, *, with_vectors: bool) -> Callable[[], Any]:
    """Bind one filtered page read to its arguments, so no loop variable is captured late."""
    return lambda: handle.query.fetch_objects(
        filters=where, limit=_PAGE_SIZE, offset=offset, include_vector=with_vectors
    )


def _object_for(entry: Point, target: _Collection) -> Any:
    """Build the Weaviate object one point is written as, postings included."""
    properties = {**_properties_for(entry.payload), PK_PROPERTY: entry.point_id}
    if target.sparse:
        sparse = entry.sparse
        postings = (
            []
            if sparse is None
            else sorted(
                (int(index), float(weight))
                for index, weight in zip(sparse.indices, sparse.values, strict=True)
            )
        )
        properties[TERMS_PROPERTY] = [index for index, _ in postings]
        properties[WEIGHTS_PROPERTY] = [weight for _, weight in postings]
    return DataObject(
        properties=properties,
        uuid=point_uuid_for(entry.point_id),
        vector=[float(value) for value in entry.vector],
    )


def _rank(
    sparse: SparseVector,
    candidates: list[Any],
    *,
    total: int,
    allowed: set[str] | None,
) -> list[tuple[str, float]]:
    """Score every candidate with BM25, taking IDF from the live corpus."""
    postings: list[tuple[str, dict[int, float]]] = []
    for entry in candidates:
        properties = dict(entry.properties)
        carried = _to_sparse(properties)
        held = {} if carried is None else dict(zip(carried.indices, carried.values, strict=True))
        postings.append((str(properties.get(PK_PROPERTY, "")), held))

    frequencies = {
        int(index): sum(1 for _, held in postings if int(index) in held) for index in sparse.indices
    }
    idf = {index: _idf(total, documents) for index, documents in frequencies.items() if documents}

    scored: list[tuple[str, float]] = []
    for point_id, held in postings:
        if allowed is not None and point_id not in allowed:
            continue
        score = sum(
            held[int(index)] * float(weight) * idf[int(index)]
            for index, weight in zip(sparse.indices, sparse.values, strict=True)
            if int(index) in held and int(index) in idf
        )
        scored.append((point_id, score))

    scored.sort(key=lambda hit: (-hit[1], hit[0]))
    return scored
