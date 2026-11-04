r"""Chroma adapter — the contract proved against an in-process, file-backed store.

Qdrant is the reference implementation (``docs/adr/ADR-0001``) and pgvector proved the
contract holds over SQL. Chroma is the third paradigm: no server at all. It runs inside
the fasterRag process against a persistent directory, which makes it the backend a
laptop, a CI job, or a single-box deployment can use with nothing installed beside the
Python package.

**Layout.** Everything lives under ``vector_db.chroma.path``:

* One Chroma collection per fasterRag collection, named ``frag-<slug>-<digest>``. The
  derivation exists because Chroma validates collection names against
  ``[a-zA-Z0-9._-]{3,512}`` while fasterRag names allow ``_`` and a single character, so
  a legal fasterRag collection is not necessarily a legal Chroma one. The logical name is
  recorded in the collection's own metadata, which is what ``list_collections`` reads.
* One Chroma collection per snapshot, named ``fragsnap-<slug>-<digest>`` and tagged in
  its metadata with the collection it was taken from.
* One Chroma collection, ``fasterrag-aliases``, holding one record per alias.

**Aliases are records in a Chroma collection, not a rename.** Chroma has no alias
concept, and renaming the collection an alias points at is wrong for the same reason it
is wrong in SQL: after the swap the old collection no longer answers to its own name,
which breaks blue/green reindexing (D2), where both collections must stay individually
addressable while traffic moves. A single-record ``upsert`` into the alias collection is
one operation that replaces the previous target in place, so there is no instant in which
the alias resolves to nothing, and because the record lives in Chroma's own persistent
directory it survives a process restart exactly as the vectors do — the same files, the
same backup, nothing extra to keep in sync. The cost is one indexed id lookup per
operation that has to resolve a name.

**Payloads are flattened into Chroma metadata, losslessly.** Chroma metadata values are
scalars, so a nested payload such as ``span`` cannot be stored directly. Scalar values are
written as ordinary keys, which is what makes filters push down; everything else — nested
objects, lists, nulls, and any key colliding with the reserved ``fr.`` namespace — is
JSON-encoded under ``fr.j.<key>`` and decoded back on read. Reserved keys never reach a
caller's payload.

**The BM25 leg stores postings as metadata and computes IDF in the adapter.** Chroma has
no sparse-vector scoring primitive that weights by corpus rarity, so each document's
saturated term frequencies are written as ``fr.t.<term>`` metadata keys and a keyword
search is a metadata query for any of the query's terms followed by the BM25 sum here.
IDF still comes from the live corpus — document frequency is counted over every posting
the query's terms match, and the corpus size is the collection's own count — which is the
split ``docs/adr/ADR-0007`` mandates, just evaluated on this side of a process boundary
that does not exist for an embedded store. **The honest limit**: that read pulls every
document containing any query term into this process. It is correct, and it does not
scale the way Qdrant's server-side sparse index does.

**Those postings are still the collection's keyword leg, so ``iterate_points`` returns
them** (TASK-0291). They are stored in fasterRag's own representation rather than a vendor
one, which is the whole reason the question comes up — but the test that matters is whether
they are what the backend scores a keyword search with, and they are the only thing it
scores one with. Reading them back is exact: the weight under ``fr.t.<index>`` is the value
``encode_document`` produced, not a re-derivation, so a point walked out and written back
ranks identically. Refusing to return them would have made Chroma the one backend whose
archives silently lose the leg.

**Snapshots are copies into another Chroma collection in the same directory.** Chroma's
own ``fork`` is not implemented for a local client, so ``snapshot()`` creates a second
collection and copies the records into it. Like pgvector's table copy this is *not* a
physical backup: it lives in the same directory as the data it protects, so it survives a
bad delete but not the loss of the disk, and it cannot be shipped elsewhere as a file.

**Everything blocking runs off the event loop.** The Chroma client is synchronous, so
every call is dispatched through ``asyncio.to_thread``; nothing here touches sqlite or the
HNSW index on the loop thread.

**Remaining limits of this first cut**, none of them silent: ``shard_number`` and
``replication_factor`` have no embedded equivalent and are ignored rather than
reinterpreted, and ``close()`` clears Chroma's process-global client cache because an
embedded client exposes no per-instance shutdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar, cast

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
    import chromadb
    from chromadb import Collection as ChromaCollection
    from chromadb import errors as chroma_errors
    from chromadb.api import ClientAPI
    from chromadb.api.types import GetResult, Include, Metadata, Where

    _VENDOR_ERRORS: tuple[type[BaseException], ...] = (
        chroma_errors.ChromaError,
        ValueError,
        OSError,
    )
    _DRIVER_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    _DRIVER_IMPORT_ERROR = exc

__all__ = ["ALIAS_COLLECTION", "INSTALL_HINT", "ChromaAdapter", "collection_name_for"]

INSTALL_HINT: Final = 'pip install "fasterrag[chroma]"'

# CRITICAL: these two names and the `fr.` metadata namespace are the on-disk contract of a
# fasterRag Chroma directory. A physical collection name is derived but also recorded in
# metadata, so the derivation may change; these may not, or an upgrade loses every alias
# and reads every stored payload back with its reserved keys exposed.
ALIAS_COLLECTION: Final = "fasterrag-aliases"
RESERVED_PREFIX: Final = "fr."

_COLLECTION_PREFIX: Final = "frag"
_SNAPSHOT_PREFIX: Final = "fragsnap"
_TERM_PREFIX: Final = f"{RESERVED_PREFIX}t."
_JSON_PREFIX: Final = f"{RESERVED_PREFIX}j."
_VERSION_KEY: Final = f"{RESERVED_PREFIX}v"
_KIND_KEY: Final = f"{RESERVED_PREFIX}kind"
_NAME_KEY: Final = f"{RESERVED_PREFIX}name"
_DIMENSIONS_KEY: Final = f"{RESERVED_PREFIX}dims"
_DISTANCE_KEY: Final = f"{RESERVED_PREFIX}distance"
_SPARSE_KEY: Final = f"{RESERVED_PREFIX}sparse"
_SOURCE_KEY: Final = f"{RESERVED_PREFIX}source"
_SNAPSHOT_KEY: Final = f"{RESERVED_PREFIX}snapshot"
_TARGET_KEY: Final = f"{RESERVED_PREFIX}target"

_COLLECTION_KIND: Final = "collection"
_SNAPSHOT_KIND: Final = "snapshot"
_ALIAS_KIND: Final = "alias"

_LAYOUT_VERSION: Final = 1
_ALIAS_VECTOR: Final[tuple[float, ...]] = (0.0,)
_FALLBACK_BATCH: Final = 1000

_SLUG_LIMIT: Final = 32
_DIGEST_LIMIT: Final = 12
_UNSAFE_NAME_CHARS: Final = re.compile(r"[^a-z0-9]+")

_SPACES: Final[dict[Distance, str]] = {"cosine": "cosine", "dot": "ip", "euclid": "l2"}
_DISTANCES: Final[dict[str, Distance]] = {"cosine": "cosine", "dot": "dot", "euclid": "euclid"}

_READ_NOTHING: Final[Include] = []
_READ_METADATA: Final[Include] = ["metadatas"]
_READ_VECTORS: Final[Include] = ["embeddings"]
_READ_WITH_VECTORS: Final[Include] = ["metadatas", "embeddings"]

_logger = get_logger(__name__)

_T = TypeVar("_T")


def _slug(value: str) -> str:
    """Return the readable half of a derived Chroma collection name."""
    cleaned = _UNSAFE_NAME_CHARS.sub("-", value.lower()).strip("-")
    return cleaned[:_SLUG_LIMIT].strip("-") or "collection"


def _digest(*parts: str) -> str:
    """Return the short stable hash that makes a derived name unique."""
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:_DIGEST_LIMIT]


def collection_name_for(collection: str) -> str:
    """Return the Chroma collection a fasterRag collection's points are stored in."""
    return f"{_COLLECTION_PREFIX}-{_slug(collection)}-{_digest(collection)}"


def snapshot_name_for(collection: str, snapshot: str) -> str:
    """Return the Chroma collection a snapshot's records are copied into."""
    return f"{_SNAPSHOT_PREFIX}-{_slug(snapshot)}-{_digest(collection, snapshot)}"


def _distance_of(stored: str) -> Distance:
    """Translate a recorded distance value back into the neutral literal."""
    known = _DISTANCES.get(stored)
    if known is None:
        raise FasterRagError(
            f"the fasterRag metadata records distance {stored!r}, which this version does "
            f"not implement; known values are: {', '.join(sorted(_DISTANCES))}",
            code=ErrorCode.INTERNAL,
        )
    return known


def _score_of(distance: Distance, measured: float) -> float:
    """Turn a Chroma distance into the higher-is-better score the contract requires."""
    if distance == "euclid":
        return -measured
    return 1.0 - measured


def _term_clause(terms: Sequence[int]) -> Where:
    """Build the metadata filter matching any document carrying one of ``terms``."""
    clauses = [{f"{_TERM_PREFIX}{term}": {"$gte": 0.0}} for term in terms]
    if len(clauses) == 1:
        return cast("Where", clauses[0])
    return cast("Where", {"$or": clauses})


def to_chroma_where(filters: Filter | None) -> Where | None:
    """Translate a vendor-neutral filter expression into a Chroma metadata filter."""
    validate_filter(filters)
    if not filters:
        return None

    clauses: list[dict[str, Any]] = []
    for key, condition in filters.items():
        if isinstance(condition, Mapping):
            operator = next(iter(condition))
            clauses.append({key: {operator: condition[operator]}})
        else:
            clauses.append({key: {"$eq": condition}})

    if len(clauses) == 1:
        return cast("Where", clauses[0])
    return cast("Where", {"$and": clauses})


def _conjunction(first: Where, second: Where) -> Where:
    """Combine two Chroma filters, which Chroma only accepts as an explicit ``$and``."""
    return cast("Where", {"$and": [first, second]})


@dataclass(frozen=True, slots=True)
class _Collection:
    """One resolved collection: its logical name, its Chroma name, and its shape."""

    name: str
    physical: str
    dimensions: int
    distance: Distance
    sparse: bool


class ChromaAdapter(VectorDBAdapter):
    """Chroma implementation of the vector database contract."""

    def __init__(self, settings: Settings) -> None:
        """Build the adapter. The persistent directory is opened on first use."""
        if _DRIVER_IMPORT_ERROR is not None:
            raise ConfigError(
                "vector_db.provider is 'chroma' but its driver is not installed "
                f"({_DRIVER_IMPORT_ERROR}); install it with: {INSTALL_HINT}"
            )

        self._path = settings.vector_db.chroma.path
        self._batch_size = settings.vector_db.chroma.batch_size
        self._client: ClientAPI | None = None
        self._handles: dict[str, ChromaCollection] = {}
        self._lock = asyncio.Lock()

    def _translate(self, exc: BaseException, operation: str) -> FasterRagError:
        """Map a Chroma failure onto the typed error taxonomy."""
        if isinstance(exc, chroma_errors.NotFoundError):
            return FasterRagError(
                f"chroma reported a missing resource during {operation}",
                code=ErrorCode.NOT_FOUND,
            )
        if isinstance(
            exc,
            chroma_errors.UniqueConstraintError
            | chroma_errors.IDAlreadyExistsError
            | chroma_errors.DuplicateIDError
            | chroma_errors.VersionMismatchError,
        ):
            return FasterRagError(
                f"chroma reported a conflicting state during {operation}",
                code=ErrorCode.CONFLICT,
            )
        if isinstance(exc, chroma_errors.ChromaAuthError | chroma_errors.AuthorizationError):
            return ProviderError(
                f"chroma rejected the request during {operation} as unauthorized",
                code=ErrorCode.VECTOR_DB_AUTH_FAILED,
                retryable=False,
            )
        if isinstance(exc, chroma_errors.RateLimitError | chroma_errors.QuotaError):
            return ProviderError(
                f"chroma refused {operation} for exceeding a limit",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        if isinstance(
            exc,
            chroma_errors.InvalidArgumentError
            | chroma_errors.InvalidDimensionException
            | chroma_errors.InvalidUUIDError
            | chroma_errors.BatchSizeExceededError
            | ValueError,
        ):
            return FasterRagError(
                f"chroma rejected the request during {operation}: {exc}",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return ProviderError(
            f"chroma failed during {operation}: {type(exc).__name__}",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=True,
        )

    async def _call(self, operation: str, work: Callable[[], _T]) -> _T:
        """Run one blocking Chroma call off the loop, letting no vendor type escape."""
        try:
            return await asyncio.to_thread(work)
        except _VENDOR_ERRORS as exc:
            raise self._translate(exc, operation) from exc

    def _build_client(self) -> ClientAPI:
        """Open the persistent directory, refusing a path that is not a usable one."""
        root = Path(self._path).expanduser()
        if root.exists() and not root.is_dir():
            raise ConfigError(
                f"vector_db.chroma.path {self._path!r} exists but is not a directory; "
                "point it at a directory the process may create and write"
            )
        root.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(root))

    async def _ensure_client(self) -> ClientAPI:
        """Return the persistent client, opening the directory on first use."""
        existing = self._client
        if existing is not None:
            return existing

        async with self._lock:
            opened = self._client
            if opened is None:
                opened = await self._call("connect", self._build_client)
                self._client = opened
            return opened

    async def _write_batch(self) -> int:
        """Return how many records one write may carry, as Chroma reports it."""
        client = await self._ensure_client()
        reported = await self._call("max_batch_size", client.get_max_batch_size)
        return min(self._batch_size, reported) if reported > 0 else _FALLBACK_BATCH

    async def _open(self, physical: str) -> ChromaCollection | None:
        """Return the Chroma collection with this name, or None if there is none."""
        cached = self._handles.get(physical)
        if cached is not None:
            return cached

        client = await self._ensure_client()

        def work() -> ChromaCollection | None:
            try:
                return client.get_collection(physical, embedding_function=None)
            except chroma_errors.NotFoundError:
                return None

        found = await self._call("open_collection", work)
        if found is not None:
            self._handles[physical] = found
        return found

    async def _describe(self, physical: str) -> _Collection | None:
        """Read one Chroma collection's recorded fasterRag shape."""
        handle = await self._open(physical)
        if handle is None:
            return None

        metadata = handle.metadata or {}
        if metadata.get(_KIND_KEY) not in (_COLLECTION_KIND, _SNAPSHOT_KIND):
            return None
        return _Collection(
            name=str(metadata.get(_NAME_KEY, "")),
            physical=physical,
            dimensions=int(cast("int", metadata.get(_DIMENSIONS_KEY, 0))),
            distance=_distance_of(str(metadata.get(_DISTANCE_KEY, "cosine"))),
            sparse=bool(metadata.get(_SPARSE_KEY, False)),
        )

    async def _resolve(self, name: str) -> _Collection | None:
        """Resolve a collection name or an alias, a real name winning over an alias."""
        direct = await self._describe(collection_name_for(name))
        if direct is not None:
            return direct

        target = await self.alias_target(name)
        if target is None:
            return None
        return await self._describe(collection_name_for(target))

    async def _require(self, name: str) -> _Collection:
        """Resolve a collection, raising rather than letting a typo read nothing."""
        found = await self._resolve(name)
        if found is None:
            raise FasterRagError(
                f"collection {name!r} does not exist under {self._path!r}, and no alias "
                "points at one",
                code=ErrorCode.NOT_FOUND,
            )
        return found

    async def _aliases(self) -> ChromaCollection:
        """Return the collection holding the alias records, creating it if absent."""
        cached = self._handles.get(ALIAS_COLLECTION)
        if cached is not None:
            return cached

        client = await self._ensure_client()
        handle = await self._call(
            "aliases",
            lambda: client.get_or_create_collection(
                name=ALIAS_COLLECTION,
                metadata={_KIND_KEY: _ALIAS_KIND, _VERSION_KEY: _LAYOUT_VERSION},
                embedding_function=None,
            ),
        )
        self._handles[ALIAS_COLLECTION] = handle
        return handle

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create the Chroma collection, or verify an existing one is compatible."""
        existing = await self._resolve(spec.name)
        if existing is not None:
            self._require_compatible(spec, existing)
            return

        client = await self._ensure_client()
        physical = collection_name_for(spec.name)
        metadata: dict[str, str | int | float | bool] = {
            _KIND_KEY: _COLLECTION_KIND,
            _NAME_KEY: spec.name,
            _DIMENSIONS_KEY: spec.dimensions,
            _DISTANCE_KEY: spec.distance,
            _SPARSE_KEY: spec.sparse,
            _VERSION_KEY: _LAYOUT_VERSION,
        }
        created = await self._call(
            "create_collection",
            lambda: client.create_collection(
                name=physical,
                metadata=metadata,
                configuration=cast("Any", {"hnsw": {"space": _SPACES[spec.distance]}}),
                embedding_function=None,
                get_or_create=True,
            ),
        )
        self._handles[physical] = created

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
                f"collection {spec.name!r} already exists without a sparse index; a keyword "
                "leg cannot be bolted onto a populated collection, so rebuild it through a "
                "blue/green reindex",
                code=ErrorCode.CONFLICT,
            )

    async def _recorded(self, kind: str) -> list[tuple[str, Mapping[str, Any]]]:
        """Return every Chroma collection this adapter owns of one recorded kind."""
        client = await self._ensure_client()

        def work() -> list[tuple[str, dict[str, Any]]]:
            return [(entry.name, dict(entry.metadata or {})) for entry in client.list_collections()]

        listed = await self._call("list_collections", work)
        return [(name, metadata) for name, metadata in listed if metadata.get(_KIND_KEY) == kind]

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection the directory holds, with an exact record count."""
        described: list[CollectionInfo] = []
        for physical, metadata in sorted(await self._recorded(_COLLECTION_KIND)):
            handle = await self._open(physical)
            if handle is None:
                continue
            counted = await self._call("count", handle.count)
            described.append(
                CollectionInfo(
                    name=str(metadata.get(_NAME_KEY, physical)),
                    vectors=counted,
                    dimensions=int(cast("int", metadata.get(_DIMENSIONS_KEY, 0))),
                    distance=_distance_of(str(metadata.get(_DISTANCE_KEY, "cosine"))),
                    sparse=bool(metadata.get(_SPARSE_KEY, False)),
                )
            )
        return described

    async def _delete_collection(self, physical: str) -> None:
        """Delete one Chroma collection and forget any handle held for it."""
        client = await self._ensure_client()
        await self._call("delete_collection", lambda: client.delete_collection(physical))
        self._handles.pop(physical, None)

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, its snapshots, and any alias pointing at it."""
        physical = collection_name_for(name)
        if await self._describe(physical) is None:
            return False

        for snapshot in await self.list_snapshots(name):
            await self._delete_collection(snapshot_name_for(name, snapshot))

        aliases = await self._aliases()
        pointing = await self._call(
            "drop_collection",
            lambda: list(aliases.get(where=cast("Where", {_TARGET_KEY: name}), include=[])["ids"]),
        )
        if pointing:
            await self._call("drop_collection", lambda: aliases.delete(ids=pointing))

        await self._delete_collection(physical)
        return True

    async def snapshot(self, collection: str) -> str:
        """Copy a collection's records into a second collection and name the copy."""
        target = await self._require(collection)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        name = f"{target.name}-{stamp}-{uuid.uuid4().hex[:8]}"
        physical = snapshot_name_for(target.name, name)

        client = await self._ensure_client()
        metadata: dict[str, str | int | float | bool] = {
            _KIND_KEY: _SNAPSHOT_KIND,
            _NAME_KEY: name,
            _SOURCE_KEY: target.name,
            _SNAPSHOT_KEY: name,
            _DIMENSIONS_KEY: target.dimensions,
            _DISTANCE_KEY: target.distance,
            _SPARSE_KEY: target.sparse,
            _VERSION_KEY: _LAYOUT_VERSION,
        }
        created = await self._call(
            "snapshot",
            lambda: client.create_collection(
                name=physical,
                metadata=metadata,
                configuration=cast("Any", {"hnsw": {"space": _SPACES[target.distance]}}),
                embedding_function=None,
            ),
        )
        self._handles[physical] = created

        source = await self._open(target.physical)
        if source is not None:
            await self._copy(source, created)
        return name

    async def _copy(self, source: ChromaCollection, destination: ChromaCollection) -> None:
        """Copy every record of one Chroma collection into another, in batches."""
        ids = await self._call("copy", lambda: list(source.get(include=_READ_NOTHING)["ids"]))
        batch = await self._write_batch()
        for start in range(0, len(ids), batch):
            page = ids[start : start + batch]
            fetched = await self._call("copy", _reader(source, page, _READ_WITH_VECTORS))
            vectors = _vectors_of(fetched.get("embeddings"))
            metadatas = [
                cast("Metadata", dict(entry or {})) for entry in fetched["metadatas"] or []
            ]
            await self._call("copy", _writer(destination, list(fetched["ids"]), vectors, metadatas))

    async def list_snapshots(self, collection: str) -> list[str]:
        """Return the snapshots held for a collection, newest last."""
        target = await self._require(collection)
        return sorted(
            str(metadata.get(_SNAPSHOT_KEY, ""))
            for _, metadata in await self._recorded(_SNAPSHOT_KIND)
            if metadata.get(_SOURCE_KEY) == target.name
        )

    async def _snapshot_of(self, collection: str, snapshot: str) -> _Collection | None:
        """Resolve one snapshot of a collection to the Chroma collection holding it."""
        found = await self._resolve(collection)
        owner = found.name if found is not None else collection
        return await self._describe(snapshot_name_for(owner, snapshot))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Delete one snapshot, tolerating one a previous retention run already pruned."""
        held = await self._snapshot_of(collection, snapshot)
        if held is None:
            return False
        await self._delete_collection(held.physical)
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

        live = await self._open(target.physical)
        source = await self._open(held.physical)
        if live is None or source is None:
            raise FasterRagError(
                f"collection {collection!r} or its snapshot {snapshot!r} vanished mid-restore",
                code=ErrorCode.NOT_FOUND,
            )

        existing = await self._call("restore_snapshot", lambda: list(live.get(include=[])["ids"]))
        if existing:
            await self._call("restore_snapshot", lambda: live.delete(ids=existing))
        await self._copy(source, live)

    async def set_alias(self, alias: str, collection: str) -> None:
        """Point an alias at a collection in one record write.

        The write replaces the previous target in place, so there is no instant in which
        the alias resolves to nothing, and Chroma persists it with the vectors.
        """
        await self._require(collection)
        if await self._describe(collection_name_for(alias)) is not None:
            raise FasterRagError(
                f"{alias!r} is already a collection, so it cannot also be an alias; "
                "every query would resolve to the collection instead",
                code=ErrorCode.CONFLICT,
            )

        aliases = await self._aliases()
        record: Metadata = {_TARGET_KEY: collection, _VERSION_KEY: _LAYOUT_VERSION}
        await self._call("set_alias", _writer(aliases, [alias], [list(_ALIAS_VECTOR)], [record]))

    async def alias_target(self, alias: str) -> str | None:
        """Return the collection an alias resolves to."""
        aliases = await self._aliases()
        found = await self._call(
            "alias_target",
            lambda: aliases.get(ids=[alias], include=_READ_METADATA),
        )
        rows = found.get("metadatas") or []
        if not rows:
            return None
        target = (rows[0] or {}).get(_TARGET_KEY)
        return None if target is None else str(target)

    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, leaving the collection it pointed at in place."""
        if await self.alias_target(alias) is None:
            return False
        aliases = await self._aliases()
        await self._call("delete_alias", lambda: aliases.delete(ids=[alias]))
        return True

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points and their postings, overwriting any that already exist."""
        if not points:
            return UpsertResult(upserted=0)

        grouped: dict[str, list[Point]] = {}
        for point in points:
            grouped.setdefault(point.collection, []).append(point)

        batch = await self._write_batch()
        for name, group in grouped.items():
            target = await self._require(name)
            self._require_matching_dimensions(target, group)
            self._require_sparse_index(target, group)
            handle = await self._open(target.physical)
            if handle is None:
                raise FasterRagError(
                    f"collection {name!r} vanished between resolution and write",
                    code=ErrorCode.NOT_FOUND,
                )

            for start in range(0, len(group), batch):
                page = group[start : start + batch]
                await self._call(
                    "upsert",
                    _writer(
                        handle,
                        [entry.point_id for entry in page],
                        [[float(value) for value in entry.vector] for entry in page],
                        [_to_metadata(entry.payload, entry.sparse) for entry in page],
                    ),
                )

        return UpsertResult(upserted=len(points))

    @staticmethod
    def _require_matching_dimensions(target: _Collection, batch: list[Point]) -> None:
        """Reject a batch whose vectors do not match the collection's vector size."""
        for point in batch:
            actual = len(point.vector)
            if actual != target.dimensions:
                raise EmbedError(
                    f"point {point.point_id!r} has {actual} dimensions but collection "
                    f"{target.name!r} stores {target.dimensions}; the configured embedding "
                    "model does not match the one this collection was built with",
                    retryable=False,
                )

    @staticmethod
    def _require_sparse_index(target: _Collection, batch: list[Point]) -> None:
        """Reject postings written into a collection that has no keyword leg."""
        if target.sparse:
            return
        for point in batch:
            if point.sparse is not None and not point.sparse.empty:
                raise EmbedError(
                    f"point {point.point_id!r} carries a sparse vector but collection "
                    f"{target.name!r} has no sparse index; recreate it with sparse enabled",
                    retryable=False,
                )

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection, one page of ids at a time."""
        target = await self._require(collection)
        handle = await self._open(target.physical)
        if handle is None:
            return

        walked = await self._call(
            "iterate_points", lambda: list(handle.get(include=_READ_NOTHING)["ids"])
        )
        ids = sorted(walked)
        carry_sparse = with_vectors and target.sparse
        include = _READ_WITH_VECTORS if with_vectors else _READ_METADATA
        for start in range(0, len(ids), batch_size):
            page = ids[start : start + batch_size]
            fetched = await self._call("iterate_points", _reader(handle, page, include))
            vectors = _vectors_of(fetched.get("embeddings")) if with_vectors else []
            metadatas = fetched.get("metadatas") or []
            for index, point_id in enumerate(fetched["ids"]):
                stored = metadatas[index] if index < len(metadatas) else None
                yield Point(
                    point_id=point_id,
                    collection=collection,
                    vector=vectors[index] if index < len(vectors) else [],
                    payload=_from_metadata(stored),
                    sparse=_sparse_from_metadata(stored) if carry_sparse else None,
                )

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points on whichever leg the query carries."""
        if query.limit <= 0:
            return []

        target = await self._require(query.collection)
        if query.sparse is not None:
            return await self._search_sparse(target, query)
        return await self._search_dense(target, query)

    async def _search_dense(self, target: _Collection, query: SearchQuery) -> list[ScoredPoint]:
        """Run the dense leg, pushing any metadata filter into Chroma."""
        handle = await self._open(target.physical)
        if handle is None:
            raise FasterRagError(
                f"collection {target.name!r} vanished between resolution and search",
                code=ErrorCode.NOT_FOUND,
            )

        where = to_chroma_where(query.filters)
        include: Include = ["distances"]
        if query.with_payload:
            include.append("metadatas")
        if query.with_vectors:
            include.append("embeddings")

        asked: list[Sequence[float] | Sequence[int]] = [[float(v) for v in query.vector or []]]
        found = await self._call(
            "search",
            lambda: handle.query(
                query_embeddings=asked,
                n_results=query.limit,
                where=where,
                include=include,
            ),
        )

        distances = (found.get("distances") or [[]])[0]
        metadatas = (found.get("metadatas") or [[]])[0]
        vectors = _vectors_of((found.get("embeddings") or [[]])[0]) if query.with_vectors else []
        hits: list[ScoredPoint] = []
        for index, point_id in enumerate(found["ids"][0]):
            hits.append(
                ScoredPoint(
                    point_id=point_id,
                    score=_score_of(target.distance, float(distances[index])),
                    payload=(
                        _from_metadata(metadatas[index] if index < len(metadatas) else None)
                        if query.with_payload
                        else {}
                    ),
                    vector=vectors[index] if index < len(vectors) else None,
                )
            )
        return hits

    async def _search_sparse(self, target: _Collection, query: SearchQuery) -> list[ScoredPoint]:
        """Run the keyword leg, computing IDF over the live corpus."""
        if not target.sparse:
            raise FasterRagError(
                f"collection {target.name!r} has no sparse index, so it cannot serve a "
                "keyword search; recreate it with sparse enabled",
                code=ErrorCode.VALIDATION_FAILED,
            )
        sparse = query.sparse
        if sparse is None or sparse.empty:
            return []

        handle = await self._open(target.physical)
        if handle is None:
            raise FasterRagError(
                f"collection {target.name!r} vanished between resolution and search",
                code=ErrorCode.NOT_FOUND,
            )

        terms = _term_clause(sparse.indices)
        postings = await self._call(
            "search", lambda: handle.get(where=terms, include=_READ_METADATA)
        )
        candidates = list(postings["ids"])
        metadatas = [dict(entry or {}) for entry in postings.get("metadatas") or []]
        if not candidates:
            return []

        total = await self._call("search", handle.count)
        allowed = await self._permitted(handle, terms, query.filters)
        ranked = _rank(sparse, candidates, metadatas, total=total, allowed=allowed)[: query.limit]
        return await self._to_scored(handle, ranked, candidates, metadatas, query)

    async def _permitted(
        self, handle: ChromaCollection, terms: Where, filters: Filter | None
    ) -> set[str] | None:
        """Return the ids a pushed-down payload filter allows, or None when there is none."""
        payload = to_chroma_where(filters)
        if payload is None:
            return None

        combined = _conjunction(terms, payload)
        found = await self._call(
            "search", lambda: handle.get(where=combined, include=_READ_NOTHING)
        )
        return set(found["ids"])

    async def _to_scored(
        self,
        handle: ChromaCollection,
        ranked: list[tuple[str, float]],
        candidates: list[str],
        metadatas: list[dict[str, Any]],
        query: SearchQuery,
    ) -> list[ScoredPoint]:
        """Turn ranked keyword hits into results, fetching vectors only when asked."""
        positions = {point_id: index for index, point_id in enumerate(candidates)}
        vectors: dict[str, list[float]] = {}
        if query.with_vectors and ranked:
            keys = [point_id for point_id, _ in ranked]
            fetched = await self._call(
                "search", lambda: handle.get(ids=keys, include=_READ_VECTORS)
            )
            rows = _vectors_of(fetched.get("embeddings"))
            vectors = dict(zip(fetched["ids"], rows, strict=False))

        return [
            ScoredPoint(
                point_id=point_id,
                score=score,
                payload=(
                    _from_metadata(metadatas[positions[point_id]]) if query.with_payload else {}
                ),
                vector=vectors.get(point_id) if query.with_vectors else None,
            )
            for point_id, score in ranked
        ]

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors."""
        if not updates:
            return

        grouped: dict[str, list[PointUpdate]] = {}
        for entry in updates:
            grouped.setdefault(entry.collection, []).append(entry)

        batch = await self._write_batch()
        for name, group in grouped.items():
            target = await self._require(name)
            handle = await self._open(target.physical)
            if handle is None:
                raise FasterRagError(
                    f"collection {name!r} vanished between resolution and update",
                    code=ErrorCode.NOT_FOUND,
                )
            for start in range(0, len(group), batch):
                page = group[start : start + batch]
                await self._call("update", _merger(handle, page))

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points, and their postings with them."""
        target = await self._require(selector.collection)
        handle = await self._open(target.physical)
        if handle is None:
            raise FasterRagError(
                f"collection {selector.collection!r} vanished between resolution and delete",
                code=ErrorCode.NOT_FOUND,
            )

        if selector.point_ids is not None:
            keys = list(selector.point_ids)
            await self._call("delete", lambda: handle.delete(ids=keys))
            return

        where = to_chroma_where(selector.filters)
        if where is None:
            raise FasterRagError(
                "a delete filter must select something; refusing to delete a whole "
                "collection through an empty filter",
                code=ErrorCode.VALIDATION_FAILED,
            )
        await self._call("delete", lambda: handle.delete(where=where))

    async def health(self) -> HealthStatus:
        """Report reachability without raising, so probes can render the failure."""
        started = time.perf_counter()
        try:
            client = await self._ensure_client()
            await self._call("health", client.heartbeat)
        except FasterRagError as exc:
            _logger.warning(
                "vector database health check failed",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return HealthStatus(healthy=False, detail=exc.detail)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return HealthStatus(healthy=True, latency_ms=round(elapsed_ms, 3))

    async def close(self) -> None:
        """Release the persistent directory.

        Chroma caches its embedded system per path for the whole process and offers no
        per-instance shutdown, so clearing that cache is what actually frees the files.
        """
        client = self._client
        self._handles.clear()
        self._client = None
        if client is not None:
            await self._call("close", client.clear_system_cache)


def _reader(handle: ChromaCollection, ids: list[str], include: Include) -> Callable[[], GetResult]:
    """Bind one page read to its arguments, so a worker thread can run it unchanged."""
    return lambda: handle.get(ids=ids, include=include)


def _writer(
    handle: ChromaCollection,
    ids: list[str],
    vectors: list[list[float]],
    metadatas: list[Metadata],
) -> Callable[[], None]:
    """Bind one page write to its arguments, so a worker thread can run it unchanged."""
    rows: list[Sequence[float] | Sequence[int]] = list(vectors)
    return lambda: handle.upsert(ids=ids, embeddings=rows, metadatas=metadatas)


def _merger(handle: ChromaCollection, updates: list[PointUpdate]) -> Callable[[], None]:
    """Bind one metadata merge to its arguments, so a worker thread can run it unchanged."""
    ids = [entry.point_id for entry in updates]
    metadatas = [_to_metadata(entry.payload, None) for entry in updates]
    return lambda: handle.update(ids=ids, metadatas=metadatas)


def _vectors_of(stored: object) -> list[list[float]]:
    """Convert Chroma's embedding rows into plain floats, so no vendor type escapes."""
    if stored is None:
        return []
    rows: list[list[float]] = []
    for row in cast("Iterable[Iterable[float]]", stored):
        rows.append([float(value) for value in row])
    return rows


def _to_metadata(payload: Mapping[str, Any], sparse: SparseVector | None) -> Metadata:
    """Flatten a payload and its postings into Chroma's scalar-only metadata."""
    stored: dict[str, str | int | float | bool] = {_VERSION_KEY: _LAYOUT_VERSION}
    for key, value in payload.items():
        if isinstance(value, bool | int | float | str) and not key.startswith(RESERVED_PREFIX):
            stored[key] = value
        else:
            stored[f"{_JSON_PREFIX}{key}"] = json.dumps(value, default=str)

    if sparse is not None:
        for index, weight in zip(sparse.indices, sparse.values, strict=True):
            stored[f"{_TERM_PREFIX}{index}"] = float(weight)
    return stored


def _from_metadata(stored: Mapping[str, Any] | None) -> dict[str, Any]:
    """Rebuild the original payload, dropping every reserved key."""
    if not stored:
        return {}

    payload: dict[str, Any] = {
        key: value for key, value in stored.items() if not key.startswith(RESERVED_PREFIX)
    }
    for key, value in stored.items():
        if key.startswith(_JSON_PREFIX):
            payload[key.removeprefix(_JSON_PREFIX)] = _decode(key, value)
    return payload


def _sparse_from_metadata(stored: Mapping[str, Any] | None) -> SparseVector | None:
    """Rebuild the keyword leg from the postings held in reserved metadata keys.

    The representation is fasterRag's, but the data is the backend's: these are the exact
    weights ``_rank`` scores a keyword search with.
    """
    if not stored:
        return None

    postings = sorted(
        _posting(key, value) for key, value in stored.items() if key.startswith(_TERM_PREFIX)
    )
    if not postings:
        return None
    return SparseVector(
        indices=[term for term, _ in postings], values=[weight for _, weight in postings]
    )


def _posting(key: str, raw: object) -> tuple[int, float]:
    """Decode one stored posting, refusing to guess at corruption."""
    try:
        return int(key.removeprefix(_TERM_PREFIX)), float(str(raw))
    except ValueError as exc:
        raise FasterRagError(
            f"stored posting {key!r} does not hold a term index and a weight; the "
            "collection was written by something other than this adapter",
            code=ErrorCode.INTERNAL,
        ) from exc


def _decode(key: str, raw: object) -> Any:
    """Decode one JSON-encoded payload value, refusing to guess at corruption."""
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise FasterRagError(
            f"stored payload key {key!r} does not hold decodable JSON; the collection was "
            "written by something other than this adapter",
            code=ErrorCode.INTERNAL,
        ) from exc


def _rank(
    sparse: SparseVector,
    candidates: list[str],
    metadatas: list[dict[str, Any]],
    *,
    total: int,
    allowed: set[str] | None,
) -> list[tuple[str, float]]:
    """Score every candidate with BM25, taking IDF from the live corpus."""
    keys = [f"{_TERM_PREFIX}{index}" for index in sparse.indices]
    frequencies = {key: sum(1 for entry in metadatas if key in entry) for key in keys}
    idf = {
        key: math.log(1 + (total - documents + 0.5) / (documents + 0.5))
        for key, documents in frequencies.items()
        if documents
    }

    scored: list[tuple[str, float]] = []
    for position, point_id in enumerate(candidates):
        if allowed is not None and point_id not in allowed:
            continue
        entry = metadatas[position] if position < len(metadatas) else {}
        score = sum(
            float(entry[key]) * float(weight) * idf[key]
            for key, weight in zip(keys, sparse.values, strict=True)
            if key in entry and key in idf
        )
        scored.append((point_id, score))

    scored.sort(key=lambda hit: (-hit[1], hit[0]))
    return scored
