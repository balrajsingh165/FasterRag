r"""pgvector adapter — the vendor-neutral contract proved against a SQL backend.

Qdrant is the reference implementation (``docs/adr/ADR-0001``), and a contract only one
backend has ever satisfied is a contract shaped like that backend. PostgreSQL is the
opposite paradigm — tables, transactions, and SQL instead of collections, points, and a
REST/gRPC API — so an adapter that passes the same suite is what turns "any vector
database" into evidence rather than intent (``docs/testing-strategy.md`` §1.5).

**Layout.** Everything lives in one PostgreSQL schema (``vector_db.pgvector.db_schema``):

* Three catalog tables — ``fasterrag_collections``, ``fasterrag_aliases``,
  ``fasterrag_snapshots`` — hold the metadata a vector database keeps for itself and
  PostgreSQL has nowhere to put: a collection's declared dimensions and distance, what an
  alias points at, and which tables are snapshots of what.
* One table per collection, named deterministically from the collection name so a human
  reading ``\dt`` can find it, with the mapping also recorded in the catalog so the
  derivation can change without orphaning data.
* One term table per *sparse* collection, holding the BM25 posting list.

Point ids need no translation, unlike Qdrant's UUID mapping: a ``text`` primary key takes
the fasterRag chunk id verbatim, so re-upserting a chunk overwrites it by definition
rather than by construction.

**Aliases are catalog rows, not table renames.** A rename looks like the obvious SQL
analogue and is wrong twice over: ``ALTER TABLE`` takes an ``ACCESS EXCLUSIVE`` lock, so a
swap blocks every in-flight query on both tables, and after the swap the *old* collection
no longer answers to its own name — which breaks blue/green reindexing (D2), where both
collections must stay individually addressable while traffic moves. A single
``INSERT ... ON CONFLICT DO UPDATE`` on one catalog row is atomic, takes no lock a reader
can feel, and leaves no instant in which the alias resolves to nothing. The cost is one
indexed primary-key lookup per operation to resolve the name.

**Snapshots are logical copies inside the same database, and that is a real limit.**
``snapshot()`` runs ``CREATE TABLE ... AS TABLE ...`` in a transaction and records the copy
in the catalog. That is honestly less than a Qdrant snapshot: it is *not* a physical
backup; it lives in the same database and tablespace as the data it protects, so it
survives a bad ``DELETE`` but not the loss of the server, the disk, or the database; it
cannot be shipped to another machine; and it copies rows without indexes, which belong to
the live table and are simply reused on restore. It is deliberately not dressed up as more
than that — disaster recovery for a PostgreSQL deployment is ``pg_dump`` or PITR, which
fasterRag does not wrap (``docs/disaster-recovery.md``).

**The BM25 leg stores fasterRag's term frequencies; PostgreSQL supplies IDF.** Not
``tsvector``: full-text search would re-tokenize server-side with PostgreSQL's own stemmer,
so one query would hit different terms here than on Qdrant — and the adapter never sees the
query text anyway, only the encoded sparse vector (``docs/adr/ADR-0007``). Postings live in
a term table and IDF is computed per query over the live corpus, which is exactly the split
the ADR mandates. ``sparsevec`` cannot serve this: its dimension ceiling is far below the
32-bit term-hash space, and its operators offer no way to weight by corpus rarity.

**Connections are pooled, and every statement is time-bounded twice over.** Operations run
on a ``psycopg_pool.AsyncConnectionPool`` sized by ``vector_db.pgvector.pool_max_size``, so
a query that blocks on a lock no longer stalls unrelated work; the adapter previously held
one connection behind an ``asyncio.Lock``, which made it a global lock on the whole
vector-database path. Two statement budgets apply, because one cannot fit both jobs:

* Ordinary statements — searches, lookups, point writes, deletes, health — inherit
  ``reliability.timeouts.vector_db_ms`` as a session ``statement_timeout``, which is the
  bound ``docs/config-reference.md`` already promises for every vector-DB call.
* Schema changes, snapshot copies, and restores take
  ``vector_db.pgvector.maintenance_timeout_ms`` instead, applied with ``SET LOCAL`` inside
  the transaction that carries them. A ``CREATE INDEX`` building HNSW over a real corpus
  legitimately runs for minutes, so the query bound would abort exactly the operation the
  collection cannot be used without. ``SET LOCAL`` rather than ``SET`` because a failed
  maintenance statement must not hand a connection back to the pool still carrying the
  longer budget.

**Remaining limit of this first cut**, not silent: ``shard_number`` and
``replication_factor`` have no single-instance equivalent and are ignored rather than
reinterpreted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, LiteralString

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
    UpsertResult,
    VectorDBAdapter,
    validate_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError, ErrorCode, FasterRagError, ProviderError
from fasterrag.observability.logging import get_logger

try:
    import psycopg
    import psycopg_pool
    from pgvector import Vector
    from pgvector.psycopg import register_vector_async
    from psycopg import sql
    from psycopg.rows import DictRow, dict_row

    _DRIVER_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - reached only without the pgvector extra
    _DRIVER_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from psycopg import AsyncConnection

__all__ = [
    "ALIASES_TABLE",
    "COLLECTIONS_TABLE",
    "INSTALL_HINT",
    "SNAPSHOTS_TABLE",
    "PgvectorAdapter",
    "table_name_for",
]

INSTALL_HINT: Final = 'pip install "fasterrag[pgvector]"'

# CRITICAL: these three names are the operator-visible contract of a fasterRag schema.
# A per-collection table name is derived but also recorded, so that derivation may change;
# these may not, or an upgrade loses every collection's metadata and every alias.
COLLECTIONS_TABLE: Final = "fasterrag_collections"
ALIASES_TABLE: Final = "fasterrag_aliases"
SNAPSHOTS_TABLE: Final = "fasterrag_snapshots"

_TABLE_PREFIX: Final = "frag"
_SNAPSHOT_PREFIX: Final = "fragsnap"
_TERMS_SUFFIX: Final = "_t"
_SLUG_LIMIT: Final = 32
_DIGEST_LIMIT: Final = 12

_UNSAFE_IDENTIFIER_CHARS: Final = re.compile(r"[^a-z0-9_]+")

_DISTANCE_OPERATORS: Final[dict[Distance, LiteralString]] = {
    "cosine": "<=>",
    "dot": "<#>",
    "euclid": "<->",
}

_DISTANCE_OPCLASSES: Final[dict[Distance, LiteralString]] = {
    "cosine": "vector_cosine_ops",
    "dot": "vector_ip_ops",
    "euclid": "vector_l2_ops",
}

# CRITICAL: `<#>` returns the *negated* inner product and `<=>` a distance, while a
# ScoredPoint's score must rank higher-is-better for cosine and dot to match Qdrant.
# Euclid is the exception in both systems: the score is the distance and lower wins.
_SCORE_TEMPLATES: Final[dict[Distance, LiteralString]] = {
    "cosine": "1 - (embedding <=> {vector})",
    "dot": "-(embedding <#> {vector})",
    "euclid": "embedding <-> {vector}",
}

_DISTANCES: Final[dict[str, Distance]] = {"cosine": "cosine", "dot": "dot", "euclid": "euclid"}

_RANGE_OPERATORS: Final[dict[str, LiteralString]] = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
_SET_OPERATORS: Final[frozenset[str]] = frozenset({"$in", "$nin"})

_AUTH_SQLSTATES: Final[frozenset[str]] = frozenset({"28000", "28P01"})
_STATEMENT_TIMEOUT_SQLSTATE: Final = "57014"
_NOT_FOUND_SQLSTATES: Final[frozenset[str]] = frozenset({"42P01", "3F000"})
_CONFLICT_SQLSTATES: Final[frozenset[str]] = frozenset({"42P07", "23505", "42710"})
_RETRYABLE_SQLSTATE_CLASSES: Final[frozenset[str]] = frozenset({"08", "40", "53", "57", "58"})

# CRITICAL: libpq reports no SQLSTATE for a failure during connection setup — the server's
# FATAL text is the only signal that a credential was rejected rather than a socket lost.
# Matching it is the only way to keep an authentication failure non-retryable, so a wrong
# password fails once instead of being retried until the circuit breaker opens. A server
# running a non-English lc_messages degrades to the retryable transport classification.
_AUTH_MESSAGE_MARKERS: Final[tuple[str, ...]] = (
    "authentication failed",
    "no password supplied",
    "pg_hba.conf",
)

_logger = get_logger(__name__)


def _slug(value: str) -> str:
    """Return the readable half of a derived table name."""
    cleaned = _UNSAFE_IDENTIFIER_CHARS.sub("_", value.lower()).strip("_")
    return cleaned[:_SLUG_LIMIT].strip("_") or "collection"


def _digest(*parts: str) -> str:
    """Return the short stable hash that makes a derived table name unique."""
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:_DIGEST_LIMIT]


def table_name_for(collection: str) -> str:
    """Return the deterministic table name a collection's points are stored in.

    Collection names allow characters and lengths PostgreSQL identifiers do not, so the
    name is slugged for readability and suffixed with a hash for uniqueness. Exported
    because an operator inspecting a database has to map a table back to a collection, and
    guessing is how the wrong table gets dropped.
    """
    return f"{_TABLE_PREFIX}_{_slug(collection)}_{_digest(collection)}"


def _snapshot_table_name(collection: str, snapshot: str) -> str:
    """Return the table a snapshot's rows are copied into."""
    return f"{_SNAPSHOT_PREFIX}_{_slug(snapshot)}_{_digest(collection, snapshot)}"


def _distance_of(stored: str) -> Distance:
    """Translate a catalog distance value back into the neutral literal.

    Raises:
        FasterRagError: With ``INTERNAL`` if the catalog holds a distance this build does
            not know, which means the schema was written by a different version.
    """
    known = _DISTANCES.get(stored)
    if known is None:
        raise FasterRagError(
            f"the fasterRag catalog records distance {stored!r}, which this version does "
            f"not implement; known values are: {', '.join(sorted(_DISTANCES))}",
            code=ErrorCode.INTERNAL,
        )
    return known


@dataclass(frozen=True, slots=True)
class _Collection:
    """One row of the collection catalog, resolved from a collection name or an alias."""

    name: str
    table: str
    dimensions: int
    distance: Distance
    sparse: bool

    @property
    def terms_table(self) -> str:
        """Return the table holding this collection's BM25 postings."""
        return f"{self.table}{_TERMS_SUFFIX}"


class PgvectorAdapter(VectorDBAdapter):
    """PostgreSQL + pgvector implementation of the vector database contract."""

    def __init__(self, settings: Settings) -> None:
        """Build the adapter. No connection is opened until an operation runs.

        Args:
            settings: Validated configuration supplying the environment variable holding
                the DSN, the schema to own, and the shared vector-database timeout.

        Raises:
            ConfigError: If the ``pgvector`` extra is not installed. Failing here rather
                than at first query means a misinstalled deployment cannot start and then
                die under traffic.
        """
        if _DRIVER_IMPORT_ERROR is not None:
            raise ConfigError(
                "vector_db.provider is 'pgvector' but its driver is not installed "
                f"({_DRIVER_IMPORT_ERROR}); install it with: {INSTALL_HINT}"
            )

        self._dsn_env = settings.vector_db.pgvector.dsn_env
        self._schema = settings.vector_db.pgvector.db_schema
        self._statement_timeout_ms = settings.reliability.timeouts.vector_db_ms
        self._maintenance_timeout_ms = settings.vector_db.pgvector.maintenance_timeout_ms
        self._pool_max_size = settings.vector_db.pgvector.pool_max_size
        self._connect_timeout = max(1, round(settings.reliability.timeouts.vector_db_ms / 1000))
        self._pool: psycopg_pool.AsyncConnectionPool[AsyncConnection[DictRow]] | None = None
        self._lock = asyncio.Lock()

    def _qualified(self, table: str) -> sql.Identifier:
        """Return a schema-qualified identifier, so no search_path can redirect a write."""
        return sql.Identifier(self._schema, table)

    def _dsn(self) -> str:
        """Return the configured connection string.

        Raises:
            ConfigError: If no variable is named, or the named one is unset or empty.
        """
        if self._dsn_env is None:
            raise ConfigError(
                "vector_db.pgvector.dsn_env is unset, so there is no PostgreSQL connection "
                "string to read; name the environment variable holding it, for example "
                "dsn_env: PGVECTOR_DSN"
            )

        dsn = os.environ.get(self._dsn_env, "")
        if not dsn:
            raise ConfigError(
                f"vector_db.pgvector.dsn_env names {self._dsn_env!r} but that environment "
                "variable is unset; put the PostgreSQL DSN there — config.yaml never "
                "contains credentials"
            )
        return dsn

    @staticmethod
    def _require_compatible_event_loop() -> None:
        """Reject Windows' default event loop before psycopg reports it as a query failure.

        Raises:
            ProviderError: On Windows under a proactor loop, where psycopg's async driver
                refuses to run at all and would otherwise surface as an opaque interface
                error on the first statement.
        """
        if sys.platform != "win32":  # pragma: no cover - platform-specific guard
            return
        loop_name = type(asyncio.get_running_loop()).__name__
        if "Proactor" not in loop_name:
            return
        raise ProviderError(
            f"psycopg's async driver cannot run on {loop_name}, which is Windows' default "
            "event loop; run the loop with asyncio.SelectorEventLoop (Linux and macOS "
            "deployments are unaffected)",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=False,
        )

    async def _ensure_pool(self) -> psycopg_pool.AsyncConnectionPool[AsyncConnection[DictRow]]:
        """Return the pool, bootstrapping the schema and opening it on first use."""
        pool = self._pool
        if pool is not None and not pool.closed:
            return pool

        async with self._lock:
            existing = self._pool
            if existing is not None and not existing.closed:
                return existing

            self._require_compatible_event_loop()
            await self._bootstrap()
            opened: psycopg_pool.AsyncConnectionPool[AsyncConnection[DictRow]] = (
                psycopg_pool.AsyncConnectionPool(
                    self._dsn(),
                    min_size=1,
                    max_size=self._pool_max_size,
                    open=False,
                    timeout=self._connect_timeout,
                    kwargs={
                        "autocommit": True,
                        "row_factory": dict_row,
                        "connect_timeout": self._connect_timeout,
                    },
                    configure=self._configure,
                    name=f"fasterrag-pgvector-{self._schema}",
                )
            )
            await opened.open(wait=True, timeout=self._connect_timeout)
            self._pool = opened
            return opened

    async def _bootstrap(self) -> None:
        """Converge the schema on one standalone connection, before the pool opens.

        Deliberately not pool ``configure`` work, for two independent reasons. ``CREATE TABLE
        IF NOT EXISTS`` is not race-free — two sessions running it at the same instant still
        collide on the catalog's unique index — and a pool opens several connections at once,
        so putting the DDL there would manufacture the race a single connection never had.
        And a rejected credential has to surface as the authentication failure it is: a pool
        retries a failed connection in the background and reports only that it could not fill
        in time, which would turn a wrong password into a retryable timeout.
        """
        async with await psycopg.AsyncConnection.connect(
            self._dsn(),
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=self._connect_timeout,
        ) as connection:
            await self._prepare(connection)

    async def _configure(self, connection: AsyncConnection[DictRow]) -> None:
        """Prepare one pooled connection: vector types, then the ordinary statement budget.

        The budget is set per session rather than per statement so that every path is bounded
        by construction — a query added later cannot forget to ask for a timeout.
        """
        await register_vector_async(connection)
        await connection.execute(
            sql.SQL("SET statement_timeout = {budget}").format(
                budget=sql.Literal(self._statement_timeout_ms)
            )
        )

    @asynccontextmanager
    async def _maintenance(self, connection: AsyncConnection[DictRow]) -> AsyncIterator[None]:
        """Run a transaction under the longer maintenance budget.

        ``SET LOCAL`` rather than ``SET``: the setting dies with the transaction, so a
        maintenance statement that fails cannot return a connection to the pool still
        carrying a budget long enough to hide a hung query.
        """
        async with connection.transaction():
            await connection.execute(
                sql.SQL("SET LOCAL statement_timeout = {budget}").format(
                    budget=sql.Literal(self._maintenance_timeout_ms)
                )
            )
            yield

    async def _prepare(self, connection: AsyncConnection[DictRow]) -> None:
        """Converge the extension, the schema, and the catalog tables.

        Every statement is idempotent, so two processes starting against an empty database
        both succeed instead of one failing against a half-created schema.
        """
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                schema=sql.Identifier(self._schema)
            )
        )
        await connection.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "name text PRIMARY KEY, "
                "table_name text NOT NULL, "
                "dimensions integer NOT NULL, "
                "distance text NOT NULL, "
                "sparse boolean NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now())"
            ).format(table=self._qualified(COLLECTIONS_TABLE))
        )
        await connection.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "alias text PRIMARY KEY, "
                "collection text NOT NULL, "
                "updated_at timestamptz NOT NULL DEFAULT now())"
            ).format(table=self._qualified(ALIASES_TABLE))
        )
        await connection.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "collection text NOT NULL, "
                "name text NOT NULL, "
                "table_name text NOT NULL, "
                "dimensions integer NOT NULL, "
                "distance text NOT NULL, "
                "sparse boolean NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now(), "
                "PRIMARY KEY (collection, name))"
            ).format(table=self._qualified(SNAPSHOTS_TABLE))
        )

    def _auth_error(self, operation: str) -> ProviderError:
        """Build the non-retryable error for a rejected credential.

        Names the environment variable rather than any part of the DSN, so a password
        cannot reach a log line through an error message.
        """
        return ProviderError(
            f"postgresql rejected the credentials during {operation}; check the DSN in the "
            f"{self._dsn_env} environment variable and the server's pg_hba.conf",
            code=ErrorCode.VECTOR_DB_AUTH_FAILED,
            retryable=False,
        )

    def _translate(self, exc: BaseException, operation: str) -> FasterRagError:
        """Map a driver or server failure onto the typed error taxonomy.

        Keyed on SQLSTATE wherever there is one: the codes are a stable part of the
        PostgreSQL contract, while messages are localized and reworded between releases.
        """
        sqlstate = getattr(exc, "sqlstate", None)
        if not isinstance(sqlstate, str):
            return self._translate_without_sqlstate(exc, operation)

        if sqlstate in _AUTH_SQLSTATES:
            return self._auth_error(operation)
        if sqlstate == _STATEMENT_TIMEOUT_SQLSTATE:
            return ProviderError(
                f"postgresql cancelled the statement during {operation} because it exceeded "
                f"its budget; ordinary statements are bounded by "
                f"reliability.timeouts.vector_db_ms ({self._statement_timeout_ms} ms) and "
                f"schema changes, snapshots, and restores by "
                f"vector_db.pgvector.maintenance_timeout_ms "
                f"({self._maintenance_timeout_ms} ms)",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        if sqlstate in _NOT_FOUND_SQLSTATES:
            return FasterRagError(
                f"postgresql reported a missing relation during {operation} (SQLSTATE {sqlstate})",
                code=ErrorCode.NOT_FOUND,
            )
        if sqlstate in _CONFLICT_SQLSTATES:
            return FasterRagError(
                f"postgresql reported a conflicting object during {operation} "
                f"(SQLSTATE {sqlstate})",
                code=ErrorCode.CONFLICT,
            )
        return ProviderError(
            f"postgresql failed during {operation} (SQLSTATE {sqlstate})",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=sqlstate[:2] in _RETRYABLE_SQLSTATE_CLASSES,
        )

    def _translate_without_sqlstate(self, exc: BaseException, operation: str) -> FasterRagError:
        """Classify a pool or connection-setup failure, which carries no SQLSTATE to key on."""
        if isinstance(exc, psycopg_pool.PoolTimeout | psycopg_pool.TooManyRequests):
            return ProviderError(
                f"every one of the {self._pool_max_size} pooled postgresql connections was "
                f"still busy after {self._connect_timeout}s during {operation}; raise "
                "vector_db.pgvector.pool_max_size or shed load upstream",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=True,
            )
        message = str(exc).lower()
        if any(marker in message for marker in _AUTH_MESSAGE_MARKERS):
            return self._auth_error(operation)
        return ProviderError(
            f"postgresql was unreachable during {operation}: {type(exc).__name__}",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=True,
        )

    @asynccontextmanager
    async def _session(self, operation: str) -> AsyncIterator[AsyncConnection[DictRow]]:
        """Borrow one pooled connection for one operation, translating vendor failures.

        Nothing is serialized here. Each caller gets its own connection, so a statement
        waiting on a lock delays only itself; the adapter used to hold a single connection
        under an ``asyncio.Lock``, which made every operation queue behind the slowest one.
        Discarding a connection left unusable is the pool's job now, done when it is returned.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.connection() as connection:
                yield connection
        except (psycopg.Error, OSError) as exc:
            raise self._translate(exc, operation) from exc

    def _filter_sql(
        self, filters: Filter | None, *, payload: sql.Composable | None = None
    ) -> tuple[sql.Composable, list[Any]]:
        """Translate a vendor-neutral filter expression into a jsonb ``WHERE`` clause.

        Equality goes through ``@>`` rather than ``->>`` so it matches Qdrant's semantics —
        a scalar also matches an array payload containing it — and so the GIN index on the
        payload can serve it. Range comparisons compare jsonb values directly rather than
        casting to numeric: a cast raises on the first non-numeric value stored under that
        key, which turns one malformed document into a failed query.

        Args:
            filters: The validated expression, or ``None`` for no filtering.
            payload: The payload column reference, qualified by the caller when the query
                joins more than one table and a bare name would be ambiguous.

        Returns:
            The clause and its positional parameters; ``TRUE`` when nothing is filtered.

        Raises:
            FasterRagError: With ``VALIDATION_FAILED`` if the expression is unsupported.
        """
        validate_filter(filters)
        if not filters:
            return sql.SQL("TRUE"), []

        column = payload if payload is not None else sql.SQL("payload")
        clauses: list[sql.Composable] = []
        params: list[Any] = []
        for key, condition in filters.items():
            if isinstance(condition, Mapping):
                operator = next(iter(condition))
                clause, values = self._operator_sql(column, key, operator, condition[operator])
            else:
                clause, values = self._contains_sql(column, key, condition)
            clauses.append(clause)
            params.extend(values)

        return sql.SQL(" AND ").join(clauses), params

    @staticmethod
    def _contains_sql(
        payload: sql.Composable, key: str, value: Any
    ) -> tuple[sql.Composable, list[Any]]:
        """Build the jsonb containment test one equality comparison reduces to."""
        clause = sql.SQL("{payload} @> {value}::jsonb").format(
            payload=payload, value=sql.Placeholder()
        )
        return clause, [json.dumps({key: value})]

    def _operator_sql(
        self, payload: sql.Composable, key: str, operator: str, value: Any
    ) -> tuple[sql.Composable, list[Any]]:
        """Translate one validated filter operator into SQL and its parameters."""
        if operator in _RANGE_OPERATORS:
            clause: sql.Composable = sql.SQL(
                "{payload} -> {key}::text {comparison} {value}::jsonb"
            ).format(
                payload=payload,
                key=sql.Placeholder(),
                comparison=sql.SQL(_RANGE_OPERATORS[operator]),
                value=sql.Placeholder(),
            )
            return clause, [key, json.dumps(value)]

        if operator in _SET_OPERATORS:
            members = [self._contains_sql(payload, key, member) for member in value]
            if not members:
                return (sql.SQL("FALSE") if operator == "$in" else sql.SQL("TRUE")), []
            joined = sql.SQL("({})").format(sql.SQL(" OR ").join(clause for clause, _ in members))
            params = [value for _, values in members for value in values]
            if operator == "$nin":
                return sql.SQL("NOT {}").format(joined), params
            return joined, params

        clause, params = self._contains_sql(payload, key, value)
        if operator == "$ne":
            return sql.SQL("NOT ({})").format(clause), params
        return clause, params

    async def _lookup(self, connection: AsyncConnection[DictRow], name: str) -> _Collection | None:
        """Resolve a collection name or an alias to its catalog row, in one round trip.

        A direct name match wins over an alias of the same name — the safer precedence,
        because a real collection can then always be addressed by its own name.
        """
        cursor = await connection.execute(
            sql.SQL(
                "SELECT c.name, c.table_name, c.dimensions, c.distance, c.sparse "
                "FROM {collections} c "
                "WHERE c.name = %s "
                "OR c.name = (SELECT a.collection FROM {aliases} a WHERE a.alias = %s) "
                "ORDER BY (c.name = %s) DESC LIMIT 1"
            ).format(
                collections=self._qualified(COLLECTIONS_TABLE),
                aliases=self._qualified(ALIASES_TABLE),
            ),
            (name, name, name),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _Collection(
            name=str(row["name"]),
            table=str(row["table_name"]),
            dimensions=int(row["dimensions"]),
            distance=_distance_of(str(row["distance"])),
            sparse=bool(row["sparse"]),
        )

    async def _require(self, connection: AsyncConnection[DictRow], name: str) -> _Collection:
        """Resolve a collection, raising rather than letting a typo read an empty result.

        Raises:
            FasterRagError: With ``NOT_FOUND`` when neither a collection nor an alias
                answers to the name.
        """
        found = await self._lookup(connection, name)
        if found is None:
            raise FasterRagError(
                f"collection {name!r} does not exist in schema {self._schema!r}, and no "
                "alias points at one",
                code=ErrorCode.NOT_FOUND,
            )
        return found

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create the collection's tables and indexes, or verify an existing one fits."""
        async with self._session("create_collection") as connection:
            existing = await self._lookup(connection, spec.name)
            if existing is not None:
                self._require_compatible(spec, existing)
                return
            async with self._maintenance(connection):
                await self._create_tables(connection, spec)

    async def _create_tables(
        self, connection: AsyncConnection[DictRow], spec: CollectionSpec
    ) -> None:
        """Create one collection's tables, indexes, and catalog row.

        ``shard_number`` and ``replication_factor`` are accepted and ignored: a single
        PostgreSQL instance has no equivalent, and reinterpreting them as partitions or
        replicas would silently give an operator something other than what was asked for.
        """
        table = table_name_for(spec.name)
        await connection.execute(
            sql.SQL(
                "CREATE TABLE {table} ("
                "point_id text PRIMARY KEY, "
                "embedding vector({dimensions}) NOT NULL, "
                "payload jsonb NOT NULL DEFAULT '{{}}'::jsonb)"
            ).format(table=self._qualified(table), dimensions=sql.Literal(spec.dimensions)),
        )
        await connection.execute(
            sql.SQL("CREATE INDEX ON {table} USING hnsw (embedding {opclass})").format(
                table=self._qualified(table),
                opclass=sql.SQL(_DISTANCE_OPCLASSES[spec.distance]),
            )
        )
        await connection.execute(
            sql.SQL("CREATE INDEX ON {table} USING gin (payload jsonb_path_ops)").format(
                table=self._qualified(table)
            )
        )
        if spec.sparse:
            await self._create_terms_table(connection, table)
        await connection.execute(
            sql.SQL(
                "INSERT INTO {catalog} (name, table_name, dimensions, distance, sparse) "
                "VALUES (%s, %s, %s, %s, %s)"
            ).format(catalog=self._qualified(COLLECTIONS_TABLE)),
            (spec.name, table, spec.dimensions, spec.distance, spec.sparse),
        )

    async def _create_terms_table(self, connection: AsyncConnection[DictRow], table: str) -> None:
        """Create the BM25 posting list for a collection.

        ``ON DELETE CASCADE`` is the point: postings and points die by the same statement,
        so a filtered delete cannot leave terms behind to skew every later IDF.
        """
        terms = self._qualified(f"{table}{_TERMS_SUFFIX}")
        await connection.execute(
            sql.SQL(
                "CREATE TABLE {terms} ("
                "point_id text NOT NULL REFERENCES {table}(point_id) ON DELETE CASCADE, "
                "term bigint NOT NULL, "
                "weight double precision NOT NULL, "
                "PRIMARY KEY (point_id, term))"
            ).format(terms=terms, table=self._qualified(table))
        )
        await connection.execute(sql.SQL("CREATE INDEX ON {terms} (term)").format(terms=terms))

    @staticmethod
    def _require_compatible(spec: CollectionSpec, existing: _Collection) -> None:
        """Reject an existing collection that cannot hold ``spec``.

        Raises:
            FasterRagError: With ``CONFLICT`` on a dimension, distance, or sparse-leg
                mismatch.
        """
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

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection in the schema, with an exact row count.

        ``count(*)`` rather than ``reltuples``: the estimate reads as ``-1`` until
        ``ANALYZE`` has run, which is exactly the state a freshly ingested collection is in
        when someone asks how much landed.
        """
        async with self._session("list_collections") as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT name, table_name, dimensions, distance, sparse "
                    "FROM {catalog} ORDER BY name"
                ).format(catalog=self._qualified(COLLECTIONS_TABLE))
            )
            rows = await cursor.fetchall()

            described: list[CollectionInfo] = []
            for row in rows:
                counted = await connection.execute(
                    sql.SQL("SELECT count(*) AS total FROM {table}").format(
                        table=self._qualified(str(row["table_name"]))
                    )
                )
                total = await counted.fetchone()
                described.append(
                    CollectionInfo(
                        name=str(row["name"]),
                        vectors=int(total["total"]) if total is not None else 0,
                        dimensions=int(row["dimensions"]),
                        distance=_distance_of(str(row["distance"])),
                        sparse=bool(row["sparse"]),
                    )
                )
            return described

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, its postings, its snapshots, and any alias pointing at it.

        Snapshots go with it, matching Qdrant: a snapshot of a collection that no longer
        exists is a table nothing references and nothing will ever restore, and leaving it
        behind is how a database quietly fills up.
        """
        async with self._session("drop_collection") as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT table_name FROM {catalog} WHERE name = %s").format(
                    catalog=self._qualified(COLLECTIONS_TABLE)
                ),
                (name,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False

            snapshots = await connection.execute(
                sql.SQL("SELECT table_name FROM {catalog} WHERE collection = %s").format(
                    catalog=self._qualified(SNAPSHOTS_TABLE)
                ),
                (name,),
            )
            snapshot_tables = [str(entry["table_name"]) for entry in await snapshots.fetchall()]

            async with self._maintenance(connection):
                for snapshot_table in snapshot_tables:
                    await self._drop_tables(connection, snapshot_table)
                await self._drop_tables(connection, str(row["table_name"]))
                for catalog, key in (
                    (SNAPSHOTS_TABLE, "collection"),
                    (ALIASES_TABLE, "collection"),
                    (COLLECTIONS_TABLE, "name"),
                ):
                    await connection.execute(
                        sql.SQL("DELETE FROM {catalog} WHERE {key} = %s").format(
                            catalog=self._qualified(catalog), key=sql.Identifier(key)
                        ),
                        (name,),
                    )
            return True

    async def _drop_tables(self, connection: AsyncConnection[DictRow], table: str) -> None:
        """Drop a data table and its posting list together."""
        await connection.execute(
            sql.SQL("DROP TABLE IF EXISTS {terms}").format(
                terms=self._qualified(f"{table}{_TERMS_SUFFIX}")
            )
        )
        await connection.execute(
            sql.SQL("DROP TABLE IF EXISTS {table}").format(table=self._qualified(table))
        )

    async def snapshot(self, collection: str) -> str:
        """Copy a collection's rows into a snapshot table and record it in the catalog.

        See the module docstring for what this is and is not: a logical copy in the same
        database, which protects against a bad write and not against losing the database.
        """
        async with self._session("snapshot") as connection:
            found = await self._require(connection, collection)
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            name = f"{found.name}-{stamp}-{uuid.uuid4().hex[:8]}"
            table = _snapshot_table_name(found.name, name)

            async with self._maintenance(connection):
                await connection.execute(
                    sql.SQL("CREATE TABLE {snapshot} AS TABLE {source}").format(
                        snapshot=self._qualified(table), source=self._qualified(found.table)
                    )
                )
                if found.sparse:
                    await connection.execute(
                        sql.SQL("CREATE TABLE {snapshot} AS TABLE {source}").format(
                            snapshot=self._qualified(f"{table}{_TERMS_SUFFIX}"),
                            source=self._qualified(found.terms_table),
                        )
                    )
                await connection.execute(
                    sql.SQL(
                        "INSERT INTO {catalog} (collection, name, table_name, dimensions, "
                        "distance, sparse) VALUES (%s, %s, %s, %s, %s, %s)"
                    ).format(catalog=self._qualified(SNAPSHOTS_TABLE)),
                    (found.name, name, table, found.dimensions, found.distance, found.sparse),
                )
            return name

    async def list_snapshots(self, collection: str) -> list[str]:
        """Return the snapshots held for a collection, newest last."""
        async with self._session("list_snapshots") as connection:
            found = await self._require(connection, collection)
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT name FROM {catalog} WHERE collection = %s ORDER BY created_at, name"
                ).format(catalog=self._qualified(SNAPSHOTS_TABLE)),
                (found.name,),
            )
            return [str(row["name"]) for row in await cursor.fetchall()]

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Delete one snapshot, tolerating one a previous retention run already pruned."""
        async with self._session("delete_snapshot") as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT table_name FROM {catalog} WHERE collection = %s AND name = %s"
                ).format(catalog=self._qualified(SNAPSHOTS_TABLE)),
                (collection, snapshot),
            )
            row = await cursor.fetchone()
            if row is None:
                return False

            async with self._maintenance(connection):
                await self._drop_tables(connection, str(row["table_name"]))
                await connection.execute(
                    sql.SQL("DELETE FROM {catalog} WHERE collection = %s AND name = %s").format(
                        catalog=self._qualified(SNAPSHOTS_TABLE)
                    ),
                    (collection, snapshot),
                )
            return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Replace a collection's contents with those a snapshot holds.

        The whole restore is one transaction, so a failure part-way leaves the collection as
        it was rather than emptied — an incident is the worst possible time to discover that
        a restore truncated first and failed second.

        This undoes writes; it does not resurrect a dropped collection. Dropping takes the
        collection's snapshots with it, here and on Qdrant alike, so there is nothing left
        to restore from — recovering a dropped collection is a ``pg_dump`` restore, and
        pretending otherwise would be the more dangerous answer.

        Raises:
            FasterRagError: With ``NOT_FOUND`` if no such snapshot is recorded or the
                collection is gone, or ``CONFLICT`` if the live collection cannot hold what
                the snapshot carries.
        """
        async with self._session("restore_snapshot") as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT table_name, dimensions, distance, sparse FROM {catalog} "
                    "WHERE collection = %s AND name = %s"
                ).format(catalog=self._qualified(SNAPSHOTS_TABLE)),
                (collection, snapshot),
            )
            row = await cursor.fetchone()
            if row is None:
                raise FasterRagError(
                    f"no snapshot named {snapshot!r} is recorded for collection {collection!r}",
                    code=ErrorCode.NOT_FOUND,
                )

            spec = CollectionSpec(
                name=collection,
                dimensions=int(row["dimensions"]),
                distance=_distance_of(str(row["distance"])),
                sparse=bool(row["sparse"]),
            )
            source = str(row["table_name"])

            target = await self._require(connection, collection)
            self._require_compatible(spec, target)

            async with self._maintenance(connection):
                await connection.execute(
                    sql.SQL("DELETE FROM {table}").format(table=self._qualified(target.table))
                )
                await connection.execute(
                    sql.SQL("INSERT INTO {table} SELECT * FROM {snapshot}").format(
                        table=self._qualified(target.table), snapshot=self._qualified(source)
                    )
                )
                if spec.sparse:
                    await connection.execute(
                        sql.SQL("INSERT INTO {terms} SELECT * FROM {snapshot}").format(
                            terms=self._qualified(target.terms_table),
                            snapshot=self._qualified(f"{source}{_TERMS_SUFFIX}"),
                        )
                    )

    async def set_alias(self, alias: str, collection: str) -> None:
        """Point an alias at a collection in one atomic catalog write.

        Raises:
            FasterRagError: With ``NOT_FOUND`` if the collection does not exist, or
                ``CONFLICT`` if a real collection already answers to the alias name — which
                would make the alias unreachable and hide the swap it exists to perform.
        """
        async with self._session("set_alias") as connection:
            await self._require(connection, collection)
            clash = await connection.execute(
                sql.SQL("SELECT 1 AS found FROM {catalog} WHERE name = %s").format(
                    catalog=self._qualified(COLLECTIONS_TABLE)
                ),
                (alias,),
            )
            if await clash.fetchone() is not None:
                raise FasterRagError(
                    f"{alias!r} is already a collection, so it cannot also be an alias; "
                    "every query would resolve to the collection instead",
                    code=ErrorCode.CONFLICT,
                )
            await connection.execute(
                sql.SQL(
                    "INSERT INTO {catalog} (alias, collection) VALUES (%s, %s) "
                    "ON CONFLICT (alias) DO UPDATE SET collection = EXCLUDED.collection, "
                    "updated_at = now()"
                ).format(catalog=self._qualified(ALIASES_TABLE)),
                (alias, collection),
            )

    async def alias_target(self, alias: str) -> str | None:
        """Return the collection an alias resolves to."""
        async with self._session("alias_target") as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT collection FROM {catalog} WHERE alias = %s").format(
                    catalog=self._qualified(ALIASES_TABLE)
                ),
                (alias,),
            )
            row = await cursor.fetchone()
            return None if row is None else str(row["collection"])

    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, leaving the collection it pointed at in place."""
        async with self._session("delete_alias") as connection:
            cursor = await connection.execute(
                sql.SQL("DELETE FROM {catalog} WHERE alias = %s").format(
                    catalog=self._qualified(ALIASES_TABLE)
                ),
                (alias,),
            )
            return cursor.rowcount > 0

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points and their postings, overwriting any that already exist."""
        if not points:
            return UpsertResult(upserted=0)

        grouped: dict[str, list[Point]] = {}
        for point in points:
            grouped.setdefault(point.collection, []).append(point)

        async with self._session("upsert") as connection:
            for name, batch in grouped.items():
                target = await self._require(connection, name)
                self._require_matching_dimensions(target, batch)
                self._require_sparse_index(target, batch)
                async with connection.transaction():
                    await self._write_points(connection, target, batch)
                    if target.sparse:
                        await self._write_terms(connection, target, batch)

        return UpsertResult(upserted=len(points))

    async def _write_points(
        self, connection: AsyncConnection[DictRow], target: _Collection, batch: list[Point]
    ) -> None:
        """Insert or overwrite the dense half of a batch."""
        async with connection.cursor() as cursor:
            await cursor.executemany(
                sql.SQL(
                    "INSERT INTO {table} (point_id, embedding, payload) "
                    "VALUES (%s, %s, %s::jsonb) "
                    "ON CONFLICT (point_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                    "payload = EXCLUDED.payload"
                ).format(table=self._qualified(target.table)),
                [
                    (point.point_id, Vector(list(point.vector)), json.dumps(dict(point.payload)))
                    for point in batch
                ],
            )

    async def _write_terms(
        self, connection: AsyncConnection[DictRow], target: _Collection, batch: list[Point]
    ) -> None:
        """Replace the posting list for every point in a batch.

        Replaced rather than merged: a re-ingested chunk whose text lost a term must lose
        that posting too, or the keyword leg keeps matching wording the corpus no longer
        contains.
        """
        await connection.execute(
            sql.SQL("DELETE FROM {terms} WHERE point_id = ANY(%s)").format(
                terms=self._qualified(target.terms_table)
            ),
            ([point.point_id for point in batch],),
        )
        rows = list(self._posting_rows(batch))
        if not rows:
            return
        async with connection.cursor() as cursor:
            await cursor.executemany(
                sql.SQL("INSERT INTO {terms} (point_id, term, weight) VALUES (%s, %s, %s)").format(
                    terms=self._qualified(target.terms_table)
                ),
                rows,
            )

    @staticmethod
    def _posting_rows(points: Iterable[Point]) -> Iterable[tuple[str, int, float]]:
        """Flatten sparse vectors into posting-list rows."""
        for point in points:
            sparse = point.sparse
            if sparse is None or sparse.empty:
                continue
            for term, weight in zip(sparse.indices, sparse.values, strict=True):
                yield point.point_id, int(term), float(weight)

    @staticmethod
    def _require_sparse_index(target: _Collection, batch: list[Point]) -> None:
        """Reject a sparse vector aimed at a collection that has no keyword index.

        Raises:
            EmbedError: Naming the point, because the alternative is accepting the write
                and silently discarding the leg the caller asked for.
        """
        if target.sparse:
            return
        carried = next((point for point in batch if point.sparse is not None), None)
        if carried is not None:
            raise EmbedError(
                f"point {carried.point_id!r} carries a sparse vector but collection "
                f"{target.name!r} has no sparse index; recreate it with sparse enabled",
                retryable=False,
            )

    @staticmethod
    def _require_matching_dimensions(target: _Collection, batch: list[Point]) -> None:
        """Reject a batch whose vectors do not match the collection's declared width.

        Raises:
            EmbedError: Naming both widths, because the cause is a changed embedding model
                and the fix is a reindex, not a retry.
        """
        for point in batch:
            actual = len(point.vector)
            if actual != target.dimensions:
                raise EmbedError(
                    f"point {point.point_id!r} has {actual} dimensions but collection "
                    f"{target.name!r} stores {target.dimensions}; the configured embedding "
                    "model does not match the one this collection was built with",
                    retryable=False,
                )

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection, one keyset page at a time.

        Keyset pagination on the primary key rather than a held server-side cursor: a cursor
        pins a transaction open for the whole walk, so exporting a large collection would
        block autovacuum for as long as the export runs. The trade is that a concurrent
        insert may or may not be seen, which the contract already permits — it specifies
        neither an ordering nor a snapshot.
        """
        after = ""
        while True:
            async with self._session("iterate_points") as connection:
                target = await self._require(connection, collection)
                columns = [sql.SQL("point_id"), sql.SQL("payload")]
                if with_vectors:
                    columns.append(sql.SQL("embedding"))
                cursor = await connection.execute(
                    sql.SQL(
                        "SELECT {columns} FROM {table} WHERE point_id > %s "
                        "ORDER BY point_id LIMIT %s"
                    ).format(
                        columns=sql.SQL(", ").join(columns),
                        table=self._qualified(target.table),
                    ),
                    (after, batch_size),
                )
                rows = await cursor.fetchall()

            for row in rows:
                after = str(row["point_id"])
                yield Point(
                    point_id=after,
                    collection=collection,
                    vector=self._to_vector(row.get("embedding")),
                    payload=dict(row["payload"] or {}),
                )

            if len(rows) < batch_size:
                return

    @staticmethod
    def _to_vector(stored: Any) -> list[float]:
        """Convert a stored pgvector value into plain floats, so no vendor type escapes."""
        if isinstance(stored, Vector):
            return stored.to_list()
        if isinstance(stored, list):
            return [float(value) for value in stored]
        return []

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points on whichever leg the query carries."""
        async with self._session("search") as connection:
            target = await self._require(connection, query.collection)
            if query.sparse is not None:
                rows = await self._search_sparse(connection, target, query)
            else:
                rows = await self._search_dense(connection, target, query)

        return [
            ScoredPoint(
                point_id=str(row["point_id"]),
                score=float(row["score"]),
                payload=dict(row["payload"] or {}) if query.with_payload else {},
                vector=self._to_vector(row["embedding"]) if query.with_vectors else None,
            )
            for row in rows
        ]

    @staticmethod
    def _projection(query: SearchQuery, *, qualified: bool = False) -> sql.Composable:
        """Return the payload and vector columns a search should transfer.

        Nothing unrequested is selected: a payload or a full vector per hit is the largest
        thing a search moves, and transferring it to discard it client-side is the kind of
        waste that only shows up at corpus scale.
        """
        columns: list[sql.Composable] = []
        if query.with_payload:
            columns.append(sql.SQL(", points.payload") if qualified else sql.SQL(", payload"))
        if query.with_vectors:
            columns.append(sql.SQL(", points.embedding") if qualified else sql.SQL(", embedding"))
        return sql.SQL("").join(columns)

    async def _search_dense(
        self, connection: AsyncConnection[DictRow], target: _Collection, query: SearchQuery
    ) -> list[DictRow]:
        """Run the dense leg, ordering by the index's own distance operator."""
        clause, filters = self._filter_sql(query.filters)
        vector = Vector(list(query.vector or []))
        cursor = await connection.execute(
            sql.SQL(
                "SELECT point_id, {score} AS score{projection} FROM {table} "
                "WHERE {clause} ORDER BY embedding {operator} %s LIMIT %s"
            ).format(
                score=sql.SQL(_SCORE_TEMPLATES[target.distance]).format(vector=sql.Placeholder()),
                projection=self._projection(query),
                table=self._qualified(target.table),
                clause=clause,
                operator=sql.SQL(_DISTANCE_OPERATORS[target.distance]),
            ),
            [vector, *filters, vector, query.limit],
        )
        return await cursor.fetchall()

    async def _search_sparse(
        self, connection: AsyncConnection[DictRow], target: _Collection, query: SearchQuery
    ) -> list[DictRow]:
        """Run the keyword leg, computing IDF over the live corpus.

        The IDF term is the Robertson/Sparck-Jones form Qdrant's ``Modifier.IDF`` applies,
        so the same query produces the same ranking on both backends. Document frequency is
        counted over the whole collection rather than the filtered subset, because a filter
        decides which documents may be returned, not what the corpus is.

        Raises:
            FasterRagError: With ``VALIDATION_FAILED`` if the collection has no sparse
                index, rather than returning the empty result a missing posting table would
                otherwise produce.
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

        clause, filters = self._filter_sql(query.filters, payload=sql.SQL("points.payload"))
        cursor = await connection.execute(
            sql.SQL(
                "WITH corpus AS (SELECT count(*)::float8 AS total FROM {table}), "
                "query AS (SELECT * FROM unnest(%s::bigint[], %s::float8[]) AS q(term, weight)), "
                "frequency AS (SELECT postings.term, count(*)::float8 AS documents "
                "FROM {terms} postings JOIN query ON query.term = postings.term "
                "GROUP BY postings.term) "
                "SELECT points.point_id, sum(postings.weight * query.weight * "
                "ln(1 + (corpus.total - frequency.documents + 0.5) "
                "/ (frequency.documents + 0.5)))::float8 AS score{projection} "
                "FROM {terms} postings "
                "JOIN query ON query.term = postings.term "
                "JOIN frequency ON frequency.term = postings.term "
                "JOIN {table} points ON points.point_id = postings.point_id "
                "CROSS JOIN corpus WHERE {clause} "
                "GROUP BY points.point_id ORDER BY score DESC LIMIT %s"
            ).format(
                table=self._qualified(target.table),
                terms=self._qualified(target.terms_table),
                projection=self._projection(query, qualified=True),
                clause=clause,
            ),
            [
                [int(term) for term in sparse.indices],
                [float(weight) for weight in sparse.values],
                *filters,
                query.limit,
            ],
        )
        return await cursor.fetchall()

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors."""
        if not updates:
            return

        grouped: dict[str, list[PointUpdate]] = {}
        for entry in updates:
            grouped.setdefault(entry.collection, []).append(entry)

        async with self._session("update") as connection:
            for name, batch in grouped.items():
                target = await self._require(connection, name)
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        sql.SQL(
                            "UPDATE {table} SET payload = payload || %s::jsonb WHERE point_id = %s"
                        ).format(table=self._qualified(target.table)),
                        [(json.dumps(dict(entry.payload)), entry.point_id) for entry in batch],
                    )

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points, and their postings with them.

        Raises:
            FasterRagError: With ``VALIDATION_FAILED`` for an empty filter, which would
                otherwise read as "delete everything".
        """
        if selector.point_ids is not None:
            clause: sql.Composable = sql.SQL("point_id = ANY(%s)")
            params: list[Any] = [list(selector.point_ids)]
        elif not selector.filters:
            raise FasterRagError(
                "a delete filter must select something; refusing to delete a whole "
                "collection through an empty filter",
                code=ErrorCode.VALIDATION_FAILED,
            )
        else:
            clause, params = self._filter_sql(selector.filters)

        async with self._session("delete") as connection:
            target = await self._require(connection, selector.collection)
            await connection.execute(
                sql.SQL("DELETE FROM {table} WHERE {clause}").format(
                    table=self._qualified(target.table), clause=clause
                ),
                params,
            )

    async def health(self) -> HealthStatus:
        """Report reachability without raising, so probes can render the failure."""
        started = time.perf_counter()
        try:
            async with self._session("health") as connection:
                await connection.execute("SELECT 1")
        except FasterRagError as exc:
            _logger.warning(
                "vector database health check failed",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return HealthStatus(healthy=False, detail=exc.detail)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return HealthStatus(healthy=True, latency_ms=round(elapsed_ms, 3))

    async def close(self) -> None:
        """Close every connection the adapter's pool holds."""
        async with self._lock:
            pool, self._pool = self._pool, None
            if pool is None or pool.closed:
                return
            try:
                await pool.close()
            except (psycopg.Error, OSError) as exc:
                raise self._translate(exc, "close") from exc
