"""pgvector adapter unit tests.

The adapter shipped with seventeen integration cases and **no unit suite at all**, while the
other reference adapter carries thirty-eight against a fake client. The coverage gate sees
only the unit run, so 1400 lines of gated adapter code arrived invisible to it and took the
gated packages from 87% to 83% (TASK-0252).

Everything here runs without PostgreSQL. The integration suite still owns whether the SQL is
*correct* against a real server; these own the logic around it — identifier derivation, the
SQLSTATE-to-taxonomy mapping, filter translation, and the guards that refuse before a
statement is ever sent.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import psycopg_pool
import pytest

from fasterrag.adapters.vectordb.base import Point, SparseVector
from fasterrag.adapters.vectordb.pgvector import PgvectorAdapter, _Collection, table_name_for
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError, ErrorCode, FasterRagError

DSN_VAR = "FASTERRAG_TEST_PG_DSN"
MAXIMUM_IDENTIFIER_BYTES = 63


def settings(**pgvector: Any) -> Settings:
    """Return settings selecting pgvector, with the DSN variable named but never read."""
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "pgvector",
                "mode": "external",
                "pgvector": {"dsn_env": DSN_VAR, **pgvector},
            }
        }
    )


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> PgvectorAdapter:
    """An adapter that never opens a connection; the DSN is present but unused."""
    monkeypatch.setenv(DSN_VAR, "postgresql://user@localhost/db")
    return PgvectorAdapter(settings())


def driver_error(sqlstate: str | None) -> psycopg.Error:
    """Return a driver error carrying ``sqlstate``, as the server would set it."""
    built = psycopg.Error("postgresql said no")
    if sqlstate is not None:
        built.sqlstate = sqlstate
    return built


class TestDerivedNames:
    """Collection names allow characters and lengths PostgreSQL identifiers do not."""

    def test_a_name_is_slugged_and_suffixed(self) -> None:
        assert table_name_for("policies").startswith("frag_policies_")

    def test_names_differing_only_in_case_never_share_a_table(self) -> None:
        """The slug lowercases, so the digest is the only thing keeping these apart."""
        assert table_name_for("policies") != table_name_for("Policies")

    def test_the_name_is_stable_across_calls(self) -> None:
        """A moving table name orphans every row already written to the old one."""
        assert table_name_for("policies") == table_name_for("policies")

    def test_punctuation_and_case_are_slugged_away(self) -> None:
        derived = table_name_for("Legal / HR 2026")

        assert derived.islower()
        assert all(character.isalnum() or character == "_" for character in derived)

    def test_a_name_of_pure_punctuation_still_yields_a_legal_identifier(self) -> None:
        """The slug can empty out, and a bare digest would be a name starting with a digit."""
        assert table_name_for("///").startswith("frag_collection_")

    def test_a_very_long_name_is_bounded(self) -> None:
        """PostgreSQL truncates identifiers at 63 bytes, and truncation can collide."""
        assert len(table_name_for("x" * 500)) <= MAXIMUM_IDENTIFIER_BYTES


class TestErrorTranslation:
    """SQLSTATE is a stable part of the PostgreSQL contract; messages are not."""

    @pytest.mark.parametrize("sqlstate", ["28000", "28P01"])
    def test_an_authentication_failure_is_not_retryable(
        self, adapter: PgvectorAdapter, sqlstate: str
    ) -> None:
        """A rejected credential is permanent, not transient.

        It fails identically forever, so retrying only delays the one
        message that explains it.
        """
        assert adapter._translate(driver_error(sqlstate), "search").retryable is False

    def test_a_cancelled_statement_names_both_budgets(self, adapter: PgvectorAdapter) -> None:
        """There are two ceilings, and the operator has to know which one fired."""
        translated = adapter._translate(driver_error("57014"), "snapshot")

        assert translated.retryable is True
        assert "vector_db_ms" in str(translated)
        assert "maintenance_timeout_ms" in str(translated)

    @pytest.mark.parametrize("sqlstate", ["42P01", "3F000"])
    def test_a_missing_relation_is_not_found(self, adapter: PgvectorAdapter, sqlstate: str) -> None:
        assert adapter._translate(driver_error(sqlstate), "search").code is ErrorCode.NOT_FOUND

    @pytest.mark.parametrize("sqlstate", ["42P07", "23505", "42710"])
    def test_a_duplicate_object_is_a_conflict(
        self, adapter: PgvectorAdapter, sqlstate: str
    ) -> None:
        translated = adapter._translate(driver_error(sqlstate), "create_collection")

        assert translated.code is ErrorCode.CONFLICT

    @pytest.mark.parametrize("sqlstate", ["08006", "40001", "53300", "57P01", "58030"])
    def test_a_transient_class_is_retryable(self, adapter: PgvectorAdapter, sqlstate: str) -> None:
        """Transient SQLSTATE classes describe a server that may answer the next call.

        Connection, serialization, resource, operator-intervention and IO classes all
        describe a server that may answer the next call.
        """
        assert adapter._translate(driver_error(sqlstate), "search").retryable is True

    @pytest.mark.parametrize("sqlstate", ["22001", "42601"])
    def test_a_permanent_class_is_not_retryable(
        self, adapter: PgvectorAdapter, sqlstate: str
    ) -> None:
        """A malformed statement or an oversized value fails the same way every time."""
        assert adapter._translate(driver_error(sqlstate), "upsert").retryable is False

    def test_an_exhausted_pool_names_the_setting_that_sizes_it(
        self, adapter: PgvectorAdapter
    ) -> None:
        """An exhausted pool is a queueing problem, not an outage.

        It carries no SQLSTATE, and reporting it as unreachable would be the wrong
        story: the server is fine and the caller is queueing.
        """
        translated = adapter._translate(psycopg_pool.PoolTimeout("busy"), "search")

        assert translated.retryable is True
        assert "pool_max_size" in str(translated)

    def test_an_unreachable_server_is_retryable(self, adapter: PgvectorAdapter) -> None:
        translated = adapter._translate(OSError("connection refused"), "health")

        assert translated.retryable is True
        assert translated.code is ErrorCode.EMBED_PROVIDER_ERROR

    def test_an_auth_failure_without_a_sqlstate_is_still_recognised(
        self, adapter: PgvectorAdapter
    ) -> None:
        """An auth failure with no SQLSTATE is still permanent.

        A connection-time rejection carries none, so the message is the only
        thing left to key on — and calling it retryable would hammer a server that will
        never accept the credential.
        """
        translated = adapter._translate(
            OSError("FATAL: password authentication failed for user"), "connect"
        )

        assert translated.retryable is False


class TestConfiguration:
    """Refusals that happen before any statement is sent."""

    def test_a_missing_dsn_variable_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal names the variable to populate.

        config.yaml never holds credentials, so the operator has to be told which
        environment variable to populate.
        """
        monkeypatch.delenv(DSN_VAR, raising=False)

        with pytest.raises(ConfigError, match=DSN_VAR):
            PgvectorAdapter(settings())._dsn()

    def test_a_blank_dsn_variable_is_refused_like_an_absent_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty variable is the shape a half-written .env actually takes."""
        monkeypatch.setenv(DSN_VAR, "   ")

        with pytest.raises(ConfigError):
            PgvectorAdapter(settings())._dsn()

    def test_the_dsn_is_read_from_the_named_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DSN_VAR, "postgresql://user@localhost/db")

        assert PgvectorAdapter(settings())._dsn() == "postgresql://user@localhost/db"

    def test_the_budgets_come_from_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both statement budgets are read from configuration.

        They are separate settings for a reason (TASK-0240), so a
        single value reaching both would be invisible here.
        """
        monkeypatch.setenv(DSN_VAR, "postgresql://user@localhost/db")
        built = PgvectorAdapter(
            Settings.model_validate(
                {
                    "vector_db": {
                        "provider": "pgvector",
                        "mode": "external",
                        "pgvector": {
                            "dsn_env": DSN_VAR,
                            "pool_max_size": 17,
                            "maintenance_timeout_ms": 123_456,
                        },
                    },
                    "reliability": {"timeouts": {"vector_db_ms": 4321}},
                }
            )
        )

        assert built._pool_max_size == 17
        assert built._maintenance_timeout_ms == 123_456
        assert built._statement_timeout_ms == 4321


class TestFilterTranslation:
    """Filters are pushed into SQL rather than applied after the rows come back."""

    def test_no_filter_binds_nothing(self, adapter: PgvectorAdapter) -> None:
        _, params = adapter._filter_sql(None)

        assert params == []

    def test_an_equality_filter_binds_a_containment_document(
        self, adapter: PgvectorAdapter
    ) -> None:
        """Equality goes through jsonb containment rather than a scalar comparison.

        That is what makes a filter match an array payload holding the value as well as a
        scalar one, matching Qdrant's semantics, and what lets the GIN index serve it. The
        value is bound, never interpolated: a payload value is caller-supplied.
        """
        _, params = adapter._filter_sql({"tenant": "acme"})

        assert params == ['{"tenant": "acme"}']

    def test_a_range_filter_binds_the_key_and_the_bound(self, adapter: PgvectorAdapter) -> None:
        """A range binds the key and the bound, comparing jsonb directly.

        Casting to numeric instead would raise on the first non-numeric value: a cast
        raises on the first non-numeric value stored under that key, turning one malformed
        document into a failed query.
        """
        _, params = adapter._filter_sql({"year": {"$gte": 2020}})

        assert params == ["year", "2020"]


class FakeCursor:
    """The two cursor methods the adapter calls."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[dict[str, Any]]:
        return self._rows

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class FakeConnection:
    """Records every statement, answering each with a scripted result."""

    def __init__(self, results: list[list[dict[str, Any]]] | None = None) -> None:
        self.statements: list[str] = []
        self._results = results or []
        self.raises: Exception | None = None

    async def execute(self, statement: Any, params: Any = None) -> FakeCursor:
        if self.raises is not None:
            raise self.raises
        self.statements.append(str(statement))
        return FakeCursor(self._results.pop(0) if self._results else [])

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class FakePool:
    """Stands in for psycopg_pool.AsyncConnectionPool, handing out one connection."""

    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self.closed = False

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[FakeConnection]:
        yield self._connection

    async def close(self) -> None:
        self.closed = True


def pooled(adapter: PgvectorAdapter, connection: FakeConnection) -> PgvectorAdapter:
    """Inject a fake pool, which `_ensure_pool` returns as-is while it is open."""
    adapter._pool = FakePool(connection)  # type: ignore[assignment]
    return adapter


class TestPooledOperations:
    """Everything reachable once a connection is in hand, without PostgreSQL."""

    async def test_health_reports_reachable_with_a_latency(self, adapter: PgvectorAdapter) -> None:
        status = await pooled(adapter, FakeConnection()).health()

        assert status.healthy is True
        assert status.latency_ms is not None

    async def test_health_reports_a_failure_rather_than_raising(
        self, adapter: PgvectorAdapter
    ) -> None:
        """A probe that raises cannot render the reason, and the reason is the whole point."""
        connection = FakeConnection()
        connection.raises = driver_error("08006")

        status = await pooled(adapter, connection).health()

        assert status.healthy is False
        assert status.detail

    async def test_listing_collections_reads_the_catalog(self, adapter: PgvectorAdapter) -> None:
        rows = [
            {
                "name": "policies",
                "table_name": table_name_for("policies"),
                "dimensions": 384,
                "distance": "cosine",
                "sparse": True,
            }
        ]
        listed = await pooled(adapter, FakeConnection([rows])).list_collections()

        assert [info.name for info in listed] == ["policies"]
        assert listed[0].dimensions == 384

    async def test_an_empty_catalog_lists_nothing(self, adapter: PgvectorAdapter) -> None:
        assert await pooled(adapter, FakeConnection([[]])).list_collections() == []

    async def test_a_failed_statement_arrives_typed(self, adapter: PgvectorAdapter) -> None:
        """`_session` is the one place vendor failures become taxonomy errors."""
        connection = FakeConnection()
        connection.raises = driver_error("42P01")

        with pytest.raises(FasterRagError) as caught:
            await pooled(adapter, connection).list_collections()

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_closing_releases_the_pool(self, adapter: PgvectorAdapter) -> None:
        """A second close must not raise; shutdown paths run twice more often than once."""
        built = pooled(adapter, FakeConnection())

        await built.close()
        await built.close()

        assert built._pool is None

    async def test_an_alias_resolves_to_its_collection(self, adapter: PgvectorAdapter) -> None:
        target = await pooled(
            adapter, FakeConnection([[{"collection": "policies_v2"}]])
        ).alias_target("policies")

        assert target == "policies_v2"

    async def test_an_unknown_alias_resolves_to_nothing(self, adapter: PgvectorAdapter) -> None:
        """`None` rather than a raise: asking whether an alias exists is a normal question."""
        assert await pooled(adapter, FakeConnection([[]])).alias_target("missing") is None

    async def test_dropping_an_unknown_collection_reports_false(
        self, adapter: PgvectorAdapter
    ) -> None:
        """Idempotent by contract — dropping what is already gone is success, not an error."""
        assert await pooled(adapter, FakeConnection([[]])).drop_collection("missing") is False

    async def test_an_unknown_snapshot_reports_false(self, adapter: PgvectorAdapter) -> None:
        deleted = await pooled(adapter, FakeConnection([[]])).delete_snapshot("policies", "gone")

        assert deleted is False


def collection(*, dimensions: int = 3, sparse: bool = False) -> _Collection:
    """Return a resolved catalog row for the write-path guards to check against."""
    return _Collection(
        name="policies",
        table=table_name_for("policies"),
        dimensions=dimensions,
        distance="cosine",
        sparse=sparse,
    )


def point(
    point_id: str = "c_1", *, vector: list[float] | None = None, sparse: SparseVector | None = None
) -> Point:
    """Return one point aimed at the collection above."""
    return Point(
        point_id=point_id,
        collection="policies",
        vector=vector if vector is not None else [0.1, 0.2, 0.3],
        payload={"tenant": "acme"},
        sparse=sparse,
    )


class TestWriteGuards:
    """Refusals that run before a row is written, where accepting is the worse outcome."""

    async def test_an_empty_batch_writes_nothing_and_asks_for_no_connection(
        self, adapter: PgvectorAdapter
    ) -> None:
        """The early return matters: no pool is touched, so an empty flush cannot fail."""
        result = await adapter.upsert([])

        assert result.upserted == 0

    def test_a_dimension_mismatch_names_both_widths(self) -> None:
        """A dimension mismatch names both widths and refuses a retry.

        The cause is a changed embedding model and the fix is a reindex, so the message
        has to carry both numbers.
        """
        with pytest.raises(EmbedError) as caught:
            PgvectorAdapter._require_matching_dimensions(
                collection(dimensions=384), [point(vector=[0.1, 0.2, 0.3])]
            )

        assert caught.value.retryable is False
        assert "384" in str(caught.value)
        assert "3 dimensions" in str(caught.value)

    def test_a_matching_batch_passes_the_dimension_guard(self) -> None:
        PgvectorAdapter._require_matching_dimensions(collection(dimensions=3), [point()])

    def test_the_offending_point_is_named_not_just_the_batch(self) -> None:
        """A batch is thousands of points; "a point has the wrong width" is unactionable."""
        batch = [point("c_ok"), point("c_bad", vector=[0.1])]

        with pytest.raises(EmbedError, match="c_bad"):
            PgvectorAdapter._require_matching_dimensions(collection(dimensions=3), batch)

    def test_a_sparse_vector_is_refused_by_a_collection_without_a_sparse_index(self) -> None:
        """A sparse vector is refused where no sparse index exists.

        Accepting it would write the dense half and discard the keyword leg the caller
        asked for, leaving a collection that answers hybrid queries with half an answer.
        """
        carried = point(sparse=SparseVector(indices=[1, 2], values=[0.5, 0.5]))

        with pytest.raises(EmbedError, match="no sparse index"):
            PgvectorAdapter._require_sparse_index(collection(sparse=False), [carried])

    def test_a_sparse_vector_is_accepted_where_the_index_exists(self) -> None:
        carried = point(sparse=SparseVector(indices=[1], values=[1.0]))

        PgvectorAdapter._require_sparse_index(collection(sparse=True), [carried])

    def test_a_dense_only_batch_passes_the_sparse_guard(self) -> None:
        """A collection with no sparse index is the ordinary case, not an error."""
        PgvectorAdapter._require_sparse_index(collection(sparse=False), [point()])

    async def test_upserting_into_an_unknown_collection_is_not_found(
        self, adapter: PgvectorAdapter
    ) -> None:
        """A typo must not read as an empty result and silently write nowhere."""
        with pytest.raises(FasterRagError) as caught:
            await pooled(adapter, FakeConnection([[]])).upsert([point()])

        assert caught.value.code is ErrorCode.NOT_FOUND


class TestCollectionCatalogRow:
    """The resolved catalog row the write path works against."""

    def test_the_terms_table_derives_from_the_points_table(self) -> None:
        """The terms table derives from the points table.

        Both have to follow from the collection name alone, or an operator inspecting the
        database cannot pair them.
        """
        resolved = collection()

        assert resolved.terms_table.startswith(resolved.table)
        assert resolved.terms_table != resolved.table
