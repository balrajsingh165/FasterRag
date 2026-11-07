"""Milvus adapter unit tests, and an explicit statement of what they cannot prove.

There is no Milvus on this machine and no Docker daemon to start one (B7.2), so the
integration contract suite under ``tests/integration/milvus`` skips in full. These cases
run against a fake client and therefore prove exactly one class of thing: that the
adapter *issues the calls it believes it issues* — the schema it declares, the indexes it
asks for, the consistency level it reads at, the expressions it pushes down, the IDF
weights it builds into a sparse query, and the shape it converts results back into.

They cannot prove that a real Milvus agrees with the fake about what those calls do.
Whether ``array_contains`` counts what we think, whether a sparse ``IP`` search returns a
plain dot product, whether ``query_iterator`` hands back a ``SPARSE_FLOAT_VECTOR`` as an
``{index: weight}`` mapping, and whether ``Strong`` consistency makes a just-written point
visible are all assertions about the server, and no test here touches a server. Where a
pure function can be pinned without a client at all — name derivation, filter rendering,
error translation — it is, because those are the parts a fake cannot flatter.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pymilvus import exceptions as milvus_errors

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    PointUpdate,
    SearchQuery,
    SparseVector,
)
from fasterrag.adapters.vectordb.milvus import (
    DENSE_FIELD,
    MAXIMUM_TERMS,
    PAYLOAD_FIELD,
    PK_FIELD,
    SPARSE_FIELD,
    TERMS_FIELD,
    MilvusAdapter,
    _alias_names,
    _Collection,
    _idf,
    _row_for,
    _score_of,
    _to_sparse,
    alias_name_for,
    collection_name_for,
    snapshot_name_for,
    to_milvus_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError

DIMENSIONS = 4
MILVUS_NAME_LIMIT = 255


def settings(**vector_db: Any) -> Settings:
    """Return settings selecting Milvus, with no credential named."""
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "milvus",
                "mode": "external",
                "host": "localhost",
                "port": 19530,
                "api_key_env": None,
                **vector_db,
            }
        }
    )


def server_error(message: str, *, code: int = 1, **flags: bool) -> milvus_errors.MilvusException:
    """Return the exception pymilvus raises for one server status."""
    return milvus_errors.MilvusException(code=code, message=message, **flags)


class FakeIndexParams:
    """Records the indexes the adapter asks Milvus to build."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add_index(self, field_name: str = "", index_type: str = "", **kwargs: Any) -> None:
        self.entries.append({"field_name": field_name, "index_type": index_type, **kwargs})

    def fields(self) -> set[str]:
        return {str(entry["field_name"]) for entry in self.entries}


class FakeSchema:
    """Records the fields the adapter declares, in declaration order."""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.fields: list[dict[str, Any]] = []
        self.description = ""

    def add_field(self, field_name: str, datatype: Any, **kwargs: Any) -> None:
        self.fields.append({"name": field_name, "type": datatype, **kwargs})

    def named(self) -> dict[str, dict[str, Any]]:
        return {str(field["name"]): field for field in self.fields}


class FakeCursor:
    """A query iterator over pre-paged rows, which must be closed exactly once."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self._pages = list(pages)
        self.closed = 0

    def next(self) -> list[dict[str, Any]]:
        return self._pages.pop(0) if self._pages else []

    def close(self) -> None:
        self.closed += 1


class FakeClient:
    """An in-memory stand-in recording every call the adapter makes.

    Structure is modelled faithfully — collections, schemas, aliases, rows. Ranking is
    not: ``search`` replays scripted hits, since simulating ANN proves the simulation.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.collections: dict[str, dict[str, Any]] = {}
        self.indexes: dict[str, FakeIndexParams] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.aliases: dict[str, str] = {}
        self.hits: list[list[dict[str, Any]]] = []
        self.cursors: list[FakeCursor] = []
        self.loaded: list[str] = []
        self.version = "v2.6.0"
        self.fails: dict[str, BaseException] = {}

    def _record(self, name: str, **arguments: Any) -> None:
        failure = self.fails.get(name)
        self.calls.append((name, arguments))
        if failure is not None:
            raise failure

    def issued(self, name: str) -> list[dict[str, Any]]:
        return [arguments for called, arguments in self.calls if called == name]

    @staticmethod
    def create_schema(**options: Any) -> FakeSchema:
        return FakeSchema(**options)

    @staticmethod
    def prepare_index_params() -> FakeIndexParams:
        return FakeIndexParams()

    def create_collection(
        self,
        collection_name: str,
        schema: FakeSchema,
        index_params: FakeIndexParams,
        **kwargs: Any,
    ) -> None:
        self._record("create_collection", collection_name=collection_name, schema=schema, **kwargs)
        named = schema.named()
        self.collections[collection_name] = {
            "collection_name": collection_name,
            "description": schema.description,
            "fields": [
                {
                    "name": name,
                    "params": {"dim": field["dim"]} if "dim" in field else {},
                }
                for name, field in named.items()
            ],
        }
        self.indexes[collection_name] = index_params
        self.rows[collection_name] = []

    def has_collection(self, collection_name: str) -> bool:
        self._record("has_collection", collection_name=collection_name)
        return collection_name in self.collections

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        self._record("describe_collection", collection_name=collection_name)
        return self.collections[collection_name]

    def list_collections(self) -> list[str]:
        self._record("list_collections")
        return sorted(self.collections)

    def drop_collection(self, collection_name: str) -> None:
        self._record("drop_collection", collection_name=collection_name)
        self.collections.pop(collection_name, None)
        self.rows.pop(collection_name, None)

    def load_collection(self, collection_name: str) -> None:
        self._record("load_collection", collection_name=collection_name)
        self.loaded.append(collection_name)

    def upsert(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, int]:
        self._record("upsert", collection_name=collection_name, data=data)
        held = self.rows.setdefault(collection_name, [])
        for row in data:
            held[:] = [kept for kept in held if kept[PK_FIELD] != row[PK_FIELD]]
            held.append(dict(row))
        return {"upsert_count": len(data)}

    def get(
        self, collection_name: str, ids: list[str], output_fields: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]:
        self._record(
            "get",
            collection_name=collection_name,
            ids=ids,
            output_fields=output_fields,
            **kwargs,
        )
        wanted = set(ids)
        return [
            {key: value for key, value in row.items() if key in output_fields}
            for row in self.rows.get(collection_name, [])
            if row[PK_FIELD] in wanted
        ]

    def query(
        self, collection_name: str, output_fields: Any = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self._record(
            "query", collection_name=collection_name, output_fields=output_fields, **kwargs
        )
        expression = str(kwargs.get("filter", ""))
        held = self.rows.get(collection_name, [])
        if output_fields == ["count(*)"]:
            return [{"count(*)": sum(1 for row in held if _matches(row, expression))}]
        return [dict(row) for row in held if _matches(row, expression)]

    def query_iterator(
        self, collection_name: str, batch_size: int, output_fields: list[str], **kwargs: Any
    ) -> FakeCursor:
        self._record(
            "query_iterator",
            collection_name=collection_name,
            batch_size=batch_size,
            output_fields=output_fields,
            **kwargs,
        )
        held = self.rows.get(collection_name, [])
        pages = [
            [
                {key: value for key, value in row.items() if key in output_fields}
                for row in held[start : start + batch_size]
            ]
            for start in range(0, len(held), batch_size)
        ]
        cursor = FakeCursor(pages)
        self.cursors.append(cursor)
        return cursor

    def delete(
        self, collection_name: str, ids: list[str] | None = None, **kwargs: Any
    ) -> dict[str, int]:
        self._record("delete", collection_name=collection_name, ids=ids, **kwargs)
        held = self.rows.get(collection_name, [])
        if ids is not None:
            held[:] = [row for row in held if row[PK_FIELD] not in set(ids)]
        else:
            held.clear()
        return {"delete_count": 0}

    def search(self, collection_name: str, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._record("search", collection_name=collection_name, **kwargs)
        return [self.hits.pop(0)] if self.hits else [[]]

    def create_alias(self, collection_name: str, alias: str) -> None:
        self._record("create_alias", collection_name=collection_name, alias=alias)
        self.aliases[alias] = collection_name

    def alter_alias(self, collection_name: str, alias: str) -> None:
        self._record("alter_alias", collection_name=collection_name, alias=alias)
        self.aliases[alias] = collection_name

    def drop_alias(self, alias: str) -> None:
        self._record("drop_alias", alias=alias)
        self.aliases.pop(alias, None)

    def describe_alias(self, alias: str) -> dict[str, str]:
        self._record("describe_alias", alias=alias)
        return {"alias": alias, "collection_name": self.aliases[alias]}

    def list_aliases(self, collection_name: str = "") -> dict[str, list[str]]:
        self._record("list_aliases", collection_name=collection_name)
        names = [
            alias
            for alias, target in self.aliases.items()
            if not collection_name or target == collection_name
        ]
        return {"aliases": names}

    def get_server_version(self) -> str:
        self._record("get_server_version")
        return self.version

    def close(self) -> None:
        self._record("close")


def _matches(row: dict[str, Any], expression: str) -> bool:
    """Evaluate the one expression family the fake needs: ``array_contains`` and empty."""
    if not expression:
        return True
    if expression.startswith("array_contains("):
        term = int(expression.rsplit(",", 1)[1].strip(" )"))
        return term in (row.get(TERMS_FIELD) or [])
    raise AssertionError(f"the fake cannot evaluate {expression!r}")


@pytest.fixture
def client() -> FakeClient:
    """A fake Milvus holding no collections."""
    return FakeClient()


@pytest.fixture
def adapter(client: FakeClient) -> MilvusAdapter:
    """An adapter already holding the fake, so no channel is ever opened."""
    built = MilvusAdapter(settings())
    built._client = client
    return built


@pytest.fixture
def dense(client: FakeClient) -> str:
    """A dense-only collection the fake already holds."""
    _install(client, "docs", sparse=False)
    return "docs"


@pytest.fixture
def hybrid(client: FakeClient) -> str:
    """A collection carrying both legs."""
    _install(client, "docs", sparse=True)
    return "docs"


def _install(client: FakeClient, name: str, *, sparse: bool) -> str:
    """Create a collection in the fake exactly as the adapter would have."""
    physical = collection_name_for(name)
    fields: list[dict[str, Any]] = [
        {"name": PK_FIELD, "params": {}},
        {"name": DENSE_FIELD, "params": {"dim": DIMENSIONS}},
        {"name": PAYLOAD_FIELD, "params": {}},
    ]
    if sparse:
        fields.extend(({"name": SPARSE_FIELD, "params": {}}, {"name": TERMS_FIELD, "params": {}}))
    client.collections[physical] = {
        "collection_name": physical,
        "description": f'{{"v":1,"n":"{name}","d":"cosine","k":"c"}}',
        "fields": fields,
    }
    client.rows[physical] = []
    return physical


class TestDerivedNames:
    """A legal fasterRag collection name is not necessarily a legal Milvus identifier."""

    def test_a_collection_name_is_slugged_and_suffixed(self) -> None:
        assert collection_name_for("policies").startswith("frag_policies_")

    def test_a_hyphen_never_survives_into_an_identifier(self) -> None:
        """Milvus identifiers match ``[A-Za-z_][A-Za-z0-9_]*``; fasterRag names allow ``-``."""
        assert "-" not in collection_name_for("legal-2026")

    def test_a_name_starting_with_a_digit_yields_a_legal_identifier(self) -> None:
        derived = collection_name_for("2026-archive")

        assert derived[0].isalpha()

    def test_the_name_is_stable_across_calls(self) -> None:
        """A moving collection name orphans every point already written to the old one."""
        assert collection_name_for("policies") == collection_name_for("policies")

    def test_names_differing_only_in_case_never_share_a_collection(self) -> None:
        assert collection_name_for("policies") != collection_name_for("Policies")

    def test_a_name_of_pure_punctuation_still_yields_a_legal_identifier(self) -> None:
        assert collection_name_for("///").startswith("frag_collection_")

    def test_a_very_long_name_is_bounded(self) -> None:
        assert len(collection_name_for("x" * 500)) <= MILVUS_NAME_LIMIT

    def test_a_collection_is_never_mistaken_for_a_snapshot(self) -> None:
        """``list_collections`` filters on the prefix, so these must not share one."""
        assert not collection_name_for("x").startswith("fragsnap_")
        assert snapshot_name_for("x", "s").startswith("fragsnap_")
        assert alias_name_for("x").startswith("fragalias_")

    def test_one_snapshot_name_under_two_collections_is_two_collections(self) -> None:
        assert snapshot_name_for("a", "s") != snapshot_name_for("b", "s")


class TestFilterTranslation:
    """Filters are pushed down as Milvus expressions over the payload JSON column."""

    def test_no_filter_renders_as_no_expression(self) -> None:
        assert to_milvus_filter(None) == ""
        assert to_milvus_filter({}) == ""

    def test_a_scalar_renders_as_equality(self) -> None:
        assert to_milvus_filter({"department": "legal"}) == '(fr_payload["department"] == "legal")'

    def test_a_range_operator_renders_as_its_comparison(self) -> None:
        assert to_milvus_filter({"year": {"$gte": 2024}}) == '(fr_payload["year"] >= 2024)'

    def test_set_membership_renders_as_in(self) -> None:
        rendered = to_milvus_filter({"department": {"$in": ["legal", "hr"]}})

        assert rendered == '(fr_payload["department"] in ["legal", "hr"])'

    def test_set_exclusion_renders_as_not_in(self) -> None:
        rendered = to_milvus_filter({"department": {"$nin": ["legal"]}})

        assert rendered == '(fr_payload["department"] not in ["legal"])'

    def test_inequality_renders_as_not_equal(self) -> None:
        assert to_milvus_filter({"n": {"$ne": 1}}) == '(fr_payload["n"] != 1)'

    def test_several_keys_are_conjoined(self) -> None:
        rendered = to_milvus_filter({"a": 1, "b": 2})

        assert rendered == '(fr_payload["a"] == 1) and (fr_payload["b"] == 2)'

    def test_a_boolean_renders_in_milvus_spelling(self) -> None:
        assert to_milvus_filter({"reviewed": True}) == '(fr_payload["reviewed"] == true)'

    def test_a_quote_in_a_value_cannot_escape_the_expression(self) -> None:
        """A filter value is user input; unescaped it would be an expression injection."""
        rendered = to_milvus_filter({"title": 'a" or fr_payload["x"] == "b'})

        assert rendered == '(fr_payload["title"] == "a\\" or fr_payload[\\"x\\"] == \\"b")'

    def test_a_quote_in_a_key_cannot_escape_the_expression(self) -> None:
        rendered = to_milvus_filter({'a"]': 1})

        assert rendered == '(fr_payload["a\\"]"] == 1)'

    def test_a_value_with_no_literal_form_is_refused(self) -> None:
        with pytest.raises(FasterRagError) as caught:
            to_milvus_filter({"when": {"$eq": None}})

        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    def test_an_unsupported_operator_is_refused_by_the_shared_grammar(self) -> None:
        with pytest.raises(FasterRagError) as caught:
            to_milvus_filter({"a": {"$regex": "x"}})

        assert caught.value.code is ErrorCode.VALIDATION_FAILED


class TestErrorTranslation:
    """Milvus publishes few numeric codes, so both codes and wording are mapped."""

    @pytest.fixture
    def translate(self) -> Any:
        built = MilvusAdapter(settings())
        return lambda exc: built._translate(exc, "search")

    def test_a_missing_collection_maps_to_not_found(self, translate: Any) -> None:
        assert translate(server_error("gone", code=100)).code is ErrorCode.NOT_FOUND

    def test_a_missing_resource_by_wording_maps_to_not_found(self, translate: Any) -> None:
        assert translate(server_error("alias not found")).code is ErrorCode.NOT_FOUND

    def test_a_rate_limit_is_retryable(self, translate: Any) -> None:
        translated = translate(server_error("slow down", code=8))

        assert translated.retryable is True

    def test_a_denied_request_maps_to_the_auth_code(self, translate: Any) -> None:
        translated = translate(server_error("denied", code=9))

        assert translated.code is ErrorCode.VECTOR_DB_AUTH_FAILED
        assert translated.retryable is False

    def test_an_unauthenticated_request_maps_to_the_auth_code(self, translate: Any) -> None:
        translated = translate(server_error("auth check failure, please check your credential"))

        assert translated.code is ErrorCode.VECTOR_DB_AUTH_FAILED

    def test_an_existing_resource_maps_to_conflict(self, translate: Any) -> None:
        assert translate(server_error("collection already exist")).code is ErrorCode.CONFLICT

    def test_an_input_error_is_not_retryable(self, translate: Any) -> None:
        translated = translate(server_error("bad vector", is_input_error=True))

        assert translated.code is ErrorCode.VALIDATION_FAILED
        assert translated.retryable is False

    def test_the_servers_own_retry_verdict_is_carried(self, translate: Any) -> None:
        assert translate(server_error("busy", retriable=True)).retryable is True
        assert translate(server_error("broken", retriable=False)).retryable is False

    def test_a_socket_failure_is_retryable(self, translate: Any) -> None:
        assert translate(OSError("connection reset")).retryable is True

    def test_a_client_side_value_error_is_not_retryable(self, translate: Any) -> None:
        translated = translate(ValueError("that is not a sparse matrix"))

        assert translated.code is ErrorCode.VALIDATION_FAILED
        assert translated.retryable is False


class TestConversions:
    """The small functions that stand between a Milvus value and a neutral one."""

    def test_a_sparse_field_decodes_ascending_by_term(self) -> None:
        decoded = _to_sparse({9: 0.5, 2: 0.25, 5: 0.75})

        assert decoded is not None
        assert list(decoded.indices) == [2, 5, 9]
        assert list(decoded.values) == [0.25, 0.75, 0.5]

    def test_an_absent_or_empty_sparse_field_reads_as_no_keyword_leg(self) -> None:
        assert _to_sparse(None) is None
        assert _to_sparse({}) is None
        assert _to_sparse([1, 2]) is None

    def test_a_euclid_distance_is_negated_and_a_similarity_is_not(self) -> None:
        """The contract requires higher-is-better; only L2 counts the other way."""
        assert _score_of("euclid", 0.25) == -0.25
        assert _score_of("cosine", 0.9) == 0.9
        assert _score_of("dot", 3.0) == 3.0

    def test_idf_matches_the_formula_every_backend_uses(self) -> None:
        assert _idf(10, 2) == pytest.approx(math.log(1 + (10 - 2 + 0.5) / (2 + 0.5)))

    def test_idf_stays_positive_for_a_term_in_every_document(self) -> None:
        """A non-positive IDF would make a universal term subtract from a score."""
        assert _idf(10, 10) > 0

    def test_an_alias_listing_is_read_in_both_shapes_pymilvus_returns(self) -> None:
        """``MilvusClient.list_aliases`` is annotated ``list`` and returns the handler dict."""
        assert _alias_names({"aliases": ["a", "b"]}) == ["a", "b"]
        assert _alias_names(["a"]) == ["a"]
        assert _alias_names(None) == []

    def test_a_dense_row_carries_no_posting_fields(self) -> None:
        target = _Collection("docs", "frag_docs_x", DIMENSIONS, "cosine", sparse=False)
        row = _row_for(Point("c_a", "docs", [1.0, 0.0, 0.0, 0.0], {"n": 1}), target)

        assert set(row) == {PK_FIELD, DENSE_FIELD, PAYLOAD_FIELD}
        assert row[PAYLOAD_FIELD] == {"n": 1}

    def test_a_hybrid_row_carries_postings_and_a_sorted_term_list(self) -> None:
        target = _Collection("docs", "frag_docs_x", DIMENSIONS, "cosine", sparse=True)
        sparse = SparseVector(indices=[7, 3], values=[0.5, 0.25])
        row = _row_for(Point("c_a", "docs", [1.0, 0.0, 0.0, 0.0], {}, sparse), target)

        assert row[SPARSE_FIELD] == {7: 0.5, 3: 0.25}
        assert row[TERMS_FIELD] == [3, 7]

    def test_a_hybrid_row_without_postings_is_still_writable(self) -> None:
        """A dense-only point in a hybrid collection must not be rejected by shape."""
        target = _Collection("docs", "frag_docs_x", DIMENSIONS, "cosine", sparse=True)
        row = _row_for(Point("c_a", "docs", [1.0, 0.0, 0.0, 0.0]), target)

        assert row[SPARSE_FIELD] == {}
        assert row[TERMS_FIELD] == []

    def test_a_payload_key_colliding_with_a_field_is_not_a_collision(self) -> None:
        """The payload is one JSON column, so no key can shadow a declared field."""
        target = _Collection("docs", "frag_docs_x", DIMENSIONS, "cosine", sparse=False)
        row = _row_for(Point("c_a", "docs", [1.0, 0.0, 0.0, 0.0], {PK_FIELD: "spoof"}), target)

        assert row[PK_FIELD] == "c_a"
        assert row[PAYLOAD_FIELD] == {PK_FIELD: "spoof"}


class TestClientConstruction:
    """The channel the adapter opens is built from configuration and nothing else."""

    def test_the_uri_is_built_from_host_and_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.milvus.MilvusClient",
            lambda **kwargs: seen.update(kwargs) or object(),
        )

        MilvusAdapter(settings(host="milvus.internal", port=19531))._build_client()

        assert seen["uri"] == "http://milvus.internal:19531"

    def test_tls_is_requested_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.milvus.MilvusClient",
            lambda **kwargs: seen.update(kwargs) or object(),
        )

        MilvusAdapter(settings(https=True))._build_client()

        assert seen["uri"].startswith("https://")

    def test_the_token_is_read_from_the_named_variable_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A credential is named in config and lives in the environment, never in YAML."""
        seen: dict[str, Any] = {}
        monkeypatch.setenv("FASTERRAG_TEST_MILVUS_TOKEN", "root:hunter2")
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.milvus.MilvusClient",
            lambda **kwargs: seen.update(kwargs) or object(),
        )

        MilvusAdapter(settings(api_key_env="FASTERRAG_TEST_MILVUS_TOKEN"))._build_client()

        assert seen["token"] == "root:hunter2"


class TestCollectionLifecycle:
    """What the adapter declares at creation is what a Milvus collection is forever."""

    async def test_creation_declares_every_field_the_adapter_reads_back(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        await adapter.create_collection(CollectionSpec(name="docs", dimensions=DIMENSIONS))

        schema = client.issued("create_collection")[0]["schema"]
        assert set(schema.named()) == {PK_FIELD, DENSE_FIELD, PAYLOAD_FIELD}
        assert schema.named()[DENSE_FIELD]["dim"] == DIMENSIONS
        assert schema.options["enable_dynamic_field"] is False

    async def test_a_hybrid_collection_declares_the_posting_fields_too(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        await adapter.create_collection(
            CollectionSpec(name="docs", dimensions=DIMENSIONS, sparse=True)
        )

        schema = client.issued("create_collection")[0]["schema"]
        assert {SPARSE_FIELD, TERMS_FIELD} <= set(schema.named())
        assert schema.named()[TERMS_FIELD]["max_capacity"] == MAXIMUM_TERMS

    async def test_a_hybrid_collection_indexes_the_term_list(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        """``df`` is a server-side count over that array; unindexed it is a scan."""
        await adapter.create_collection(
            CollectionSpec(name="docs", dimensions=DIMENSIONS, sparse=True)
        )

        built = client.indexes[collection_name_for("docs")]
        assert built.fields() == {DENSE_FIELD, SPARSE_FIELD, TERMS_FIELD}
        assert {"field_name": TERMS_FIELD, "index_type": "INVERTED"} in built.entries

    async def test_the_sparse_index_asks_for_inner_product(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        await adapter.create_collection(
            CollectionSpec(name="docs", dimensions=DIMENSIONS, sparse=True)
        )

        built = client.indexes[collection_name_for("docs")]
        sparse = next(e for e in built.entries if e["field_name"] == SPARSE_FIELD)
        assert sparse["metric_type"] == "IP"
        assert sparse["index_type"] == "SPARSE_INVERTED_INDEX"

    async def test_the_distance_is_translated_to_a_milvus_metric(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        await adapter.create_collection(
            CollectionSpec(name="docs", dimensions=DIMENSIONS, distance="euclid")
        )

        built = client.indexes[collection_name_for("docs")]
        dense = next(e for e in built.entries if e["field_name"] == DENSE_FIELD)
        assert dense["metric_type"] == "L2"

    async def test_creation_asks_for_strong_consistency(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        """Under the default Bounded level a just-indexed point is not yet searchable."""
        await adapter.create_collection(CollectionSpec(name="docs", dimensions=DIMENSIONS))

        assert client.issued("create_collection")[0]["consistency_level"] == "Strong"

    async def test_creating_an_existing_collection_creates_nothing(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=dense, dimensions=DIMENSIONS))

        assert client.issued("create_collection") == []

    async def test_conflicting_dimensions_are_refused(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(CollectionSpec(name=dense, dimensions=DIMENSIONS + 1))

        assert caught.value.code is ErrorCode.CONFLICT

    async def test_asking_a_dense_collection_for_a_keyword_leg_is_refused(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        """A Milvus schema is fixed at creation, so this can only ever be a rebuild."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(
                CollectionSpec(name=dense, dimensions=DIMENSIONS, sparse=True)
            )

        assert caught.value.code is ErrorCode.CONFLICT

    async def test_the_listing_reports_the_logical_name_and_shape(
        self, adapter: MilvusAdapter, hybrid: str
    ) -> None:
        listed = await adapter.list_collections()

        assert [info.name for info in listed] == ["docs"]
        assert listed[0].dimensions == DIMENSIONS
        assert listed[0].sparse is True

    async def test_the_listing_counts_with_a_strong_read(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """Milvus row statistics lag behind writes; only an exact count is trustworthy."""
        await adapter.upsert([Point("c_a", dense, [1.0, 0.0, 0.0, 0.0])])

        listed = await adapter.list_collections()

        assert listed[0].vectors == 1
        assert client.issued("query")[-1]["consistency_level"] == "Strong"

    async def test_a_snapshot_is_not_listed_as_a_collection(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        taken = await adapter.snapshot(dense)

        assert [info.name for info in await adapter.list_collections()] == ["docs"]
        assert taken in await adapter.list_snapshots(dense)

    async def test_dropping_an_absent_collection_is_not_an_error(
        self, adapter: MilvusAdapter
    ) -> None:
        assert await adapter.drop_collection("never-existed") is False

    async def test_dropping_takes_the_snapshots_and_aliases_with_it(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """A snapshot of a dropped collection is unreachable; an alias to one is broken."""
        await adapter.snapshot(dense)
        await adapter.set_alias("live", dense)

        assert await adapter.drop_collection(dense) is True
        assert client.collections == {}
        assert client.aliases == {}


class TestAliases:
    """Aliases are native, under derived names, because a fasterRag alias may be hyphenated."""

    async def test_an_alias_resolves_to_its_collection(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        await adapter.set_alias("live-alias", dense)

        assert await adapter.alias_target("live-alias") == "docs"

    async def test_an_alias_is_created_under_a_derived_name(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.set_alias("live-alias", dense)

        assert client.issued("create_alias")[0]["alias"] == alias_name_for("live-alias")

    async def test_repointing_alters_rather_than_recreating(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """``create_alias`` on an existing alias fails; the swap must be one operation."""
        _install(client, "green", sparse=False)
        await adapter.set_alias("live", dense)
        await adapter.set_alias("live", "green")

        assert len(client.issued("create_alias")) == 1
        assert len(client.issued("alter_alias")) == 1
        assert await adapter.alias_target("live") == "green"

    async def test_an_unset_alias_resolves_to_nothing(self, adapter: MilvusAdapter) -> None:
        assert await adapter.alias_target("never-set") is None

    async def test_deleting_an_absent_alias_is_not_an_error(self, adapter: MilvusAdapter) -> None:
        assert await adapter.delete_alias("never-set") is False

    async def test_deleting_an_alias_leaves_the_collection(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.set_alias("live", dense)

        assert await adapter.delete_alias("live") is True
        assert collection_name_for(dense) in client.collections

    async def test_an_alias_cannot_shadow_a_collection(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        _install(client, "green", sparse=False)

        with pytest.raises(FasterRagError) as caught:
            await adapter.set_alias("green", dense)

        assert caught.value.code is ErrorCode.CONFLICT

    async def test_a_collection_is_reachable_through_its_alias(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.set_alias("live", dense)
        client.hits = [[{PK_FIELD: "c_a", "distance": 0.9, "entity": {}}]]

        hits = await adapter.search(
            SearchQuery(collection="live", vector=[1.0, 0.0, 0.0, 0.0], with_payload=False)
        )

        assert [hit.point_id for hit in hits] == ["c_a"]
        assert client.issued("search")[0]["collection_name"] == collection_name_for("docs")


class TestWrites:
    """What a point becomes on the way in, and what the adapter refuses outright."""

    async def test_a_point_is_written_with_its_payload_in_the_json_column(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        written = await adapter.upsert(
            [Point("c_a", dense, [1.0, 0.0, 0.0, 0.0], {"department": "legal"})]
        )

        row = client.issued("upsert")[0]["data"][0]
        assert written.upserted == 1
        assert row[PK_FIELD] == "c_a"
        assert row[PAYLOAD_FIELD] == {"department": "legal"}

    async def test_a_dimension_mismatch_is_refused_before_any_call(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        with pytest.raises(EmbedError) as caught:
            await adapter.upsert([Point("c_a", dense, [1.0, 0.0])])

        assert str(DIMENSIONS) in caught.value.detail
        assert caught.value.retryable is False
        assert client.issued("upsert") == []

    async def test_postings_into_a_dense_collection_are_refused(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        with pytest.raises(EmbedError):
            await adapter.upsert(
                [
                    Point(
                        "c_a",
                        dense,
                        [1.0, 0.0, 0.0, 0.0],
                        sparse=SparseVector(indices=[1], values=[0.5]),
                    )
                ]
            )

    async def test_more_terms_than_an_array_holds_is_refused_not_truncated(
        self, adapter: MilvusAdapter, hybrid: str
    ) -> None:
        """A truncated posting list would score wrong for the life of the collection."""
        oversized = SparseVector(
            indices=list(range(MAXIMUM_TERMS + 1)), values=[1.0] * (MAXIMUM_TERMS + 1)
        )

        with pytest.raises(EmbedError) as caught:
            await adapter.upsert([Point("c_a", hybrid, [1.0, 0.0, 0.0, 0.0], sparse=oversized)])

        assert str(MAXIMUM_TERMS) in caught.value.detail

    async def test_writing_to_an_unknown_collection_raises_a_typed_error(
        self, adapter: MilvusAdapter
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.upsert([Point("c_a", "nope", [1.0, 0.0, 0.0, 0.0])])

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_an_update_reads_the_point_back_and_rewrites_it_whole(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """Milvus has no partial update, so a merge must not drop the vector."""
        await adapter.upsert([Point("c_a", dense, [1.0, 0.0, 0.0, 0.0], {"department": "legal"})])

        await adapter.update([PointUpdate("c_a", dense, {"reviewed": True})])

        rewritten = client.issued("upsert")[-1]["data"][0]
        assert rewritten[DENSE_FIELD] == [1.0, 0.0, 0.0, 0.0]
        assert rewritten[PAYLOAD_FIELD] == {"department": "legal", "reviewed": True}

    async def test_updating_a_point_that_is_not_there_is_an_error(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        """A silent no-op here looks exactly like a successful metadata write."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.update([PointUpdate("c_missing", dense, {"reviewed": True})])

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_a_delete_by_id_passes_the_ids_through(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.delete(PointSelector(collection=dense, point_ids=["c_a"]))

        assert client.issued("delete")[0]["ids"] == ["c_a"]

    async def test_a_delete_by_filter_pushes_the_expression_down(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.delete(PointSelector(collection=dense, filters={"department": "legal"}))

        assert client.issued("delete")[0]["filter"] == '(fr_payload["department"] == "legal")'

    async def test_an_empty_delete_filter_is_refused(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """An empty expression would match everything, which is not what was asked for."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.delete(PointSelector(collection=dense, filters={}))

        assert caught.value.code is ErrorCode.VALIDATION_FAILED
        assert client.issued("delete") == []


class TestDenseSearch:
    """What a dense query becomes on the wire, and what comes back from it."""

    async def test_the_dense_leg_searches_the_dense_field_with_its_metric(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        client.hits = [[{PK_FIELD: "c_a", "distance": 0.9, "entity": {PAYLOAD_FIELD: {"n": 1}}}]]

        await adapter.search(SearchQuery(collection=dense, vector=[1.0, 0.0, 0.0, 0.0], limit=3))

        issued = client.issued("search")[0]
        assert issued["anns_field"] == DENSE_FIELD
        assert issued["search_params"] == {"metric_type": "COSINE"}
        assert issued["limit"] == 3
        assert issued["consistency_level"] == "Strong"

    async def test_a_hit_carries_its_payload_back_out_of_the_json_column(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        client.hits = [[{PK_FIELD: "c_a", "distance": 0.9, "entity": {PAYLOAD_FIELD: {"n": 1}}}]]

        hits = await adapter.search(SearchQuery(collection=dense, vector=[1.0, 0.0, 0.0, 0.0]))

        assert hits[0].payload == {"n": 1}
        assert hits[0].vector is None

    async def test_vectors_come_back_only_when_asked_for(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        client.hits = [[{PK_FIELD: "c_a", "distance": 0.9, "entity": {DENSE_FIELD: [1.0, 0.0]}}]]

        hits = await adapter.search(
            SearchQuery(collection=dense, vector=[1.0, 0.0, 0.0, 0.0], with_vectors=True)
        )

        assert DENSE_FIELD in client.issued("search")[0]["output_fields"]
        assert hits[0].vector == [1.0, 0.0]

    async def test_a_euclid_collection_reports_a_higher_is_better_score(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        physical = _install(client, "docs", sparse=False)
        client.collections[physical]["description"] = '{"v":1,"n":"docs","d":"euclid","k":"c"}'
        client.hits = [[{PK_FIELD: "c_a", "distance": 0.25, "entity": {}}]]

        hits = await adapter.search(
            SearchQuery(collection="docs", vector=[1.0, 0.0, 0.0, 0.0], with_payload=False)
        )

        assert hits[0].score == -0.25

    async def test_searching_an_unknown_collection_raises_a_typed_error(
        self, adapter: MilvusAdapter
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(SearchQuery(collection="nope", vector=[1.0, 0.0, 0.0, 0.0]))

        assert caught.value.code is ErrorCode.NOT_FOUND


class TestKeywordSearch:
    """IDF is assembled from server-side counts and carried in the query vector."""

    async def seed(self, adapter: MilvusAdapter, hybrid: str) -> None:
        """Three documents: one rare term, one shared by all three."""
        await adapter.upsert(
            [
                Point(
                    f"c_{index}",
                    hybrid,
                    [1.0, 0.0, 0.0, 0.0],
                    sparse=SparseVector(indices=[11], values=[0.5]),
                )
                for index in range(3)
            ]
        )
        await adapter.upsert(
            [
                Point(
                    "c_rare",
                    hybrid,
                    [1.0, 0.0, 0.0, 0.0],
                    sparse=SparseVector(indices=[11, 22], values=[0.5, 0.25]),
                )
            ]
        )

    async def test_the_query_vector_carries_tf_times_idf(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """Milvus scores sparse by dot product, so IDF has to arrive on the query side."""
        await self.seed(adapter, hybrid)
        client.hits = [[]]

        await adapter.search(
            SearchQuery(collection=hybrid, sparse=SparseVector(indices=[11, 22], values=[1.0, 1.0]))
        )

        sent = client.issued("search")[0]["data"][0]
        assert sent[11] == pytest.approx(_idf(4, 4))
        assert sent[22] == pytest.approx(_idf(4, 1))
        assert sent[22] > sent[11]

    async def test_the_keyword_leg_searches_the_sparse_field_by_inner_product(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        await self.seed(adapter, hybrid)
        client.hits = [[]]

        await adapter.search(
            SearchQuery(collection=hybrid, sparse=SparseVector(indices=[22], values=[1.0]))
        )

        issued = client.issued("search")[0]
        assert issued["anns_field"] == SPARSE_FIELD
        assert issued["search_params"] == {"metric_type": "IP"}

    async def test_document_frequency_is_counted_per_term_on_the_server(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """One count per term, not a scan of the postings: that is the scalability claim."""
        await self.seed(adapter, hybrid)
        client.hits = [[]]
        client.calls.clear()

        await adapter.search(
            SearchQuery(collection=hybrid, sparse=SparseVector(indices=[11, 22], values=[1.0, 1.0]))
        )

        counted = [issued["filter"] for issued in client.issued("query")]
        assert counted == ["", "array_contains(fr_terms, 11)", "array_contains(fr_terms, 22)"]

    async def test_a_term_in_no_document_contributes_nothing(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        await self.seed(adapter, hybrid)
        client.hits = [[]]

        await adapter.search(
            SearchQuery(
                collection=hybrid, sparse=SparseVector(indices=[11, 999], values=[1.0, 1.0])
            )
        )

        assert set(client.issued("search")[0]["data"][0]) == {11}

    async def test_a_query_matching_nothing_never_reaches_the_index(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        await self.seed(adapter, hybrid)

        hits = await adapter.search(
            SearchQuery(collection=hybrid, sparse=SparseVector(indices=[999], values=[1.0]))
        )

        assert hits == []
        assert client.issued("search") == []

    async def test_an_empty_corpus_scores_nothing(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """IDF over zero documents is undefined, not zero."""
        hits = await adapter.search(
            SearchQuery(collection=hybrid, sparse=SparseVector(indices=[11], values=[1.0]))
        )

        assert hits == []
        assert client.issued("search") == []

    async def test_a_zero_scoring_hit_is_dropped(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """A hit sharing no weighted term is not a keyword match, whatever is returned."""
        await self.seed(adapter, hybrid)
        client.hits = [
            [
                {PK_FIELD: "c_rare", "distance": 1.5, "entity": {}},
                {PK_FIELD: "c_0", "distance": 0.0, "entity": {}},
            ]
        ]

        hits = await adapter.search(
            SearchQuery(
                collection=hybrid,
                sparse=SparseVector(indices=[22], values=[1.0]),
                with_payload=False,
            )
        )

        assert [hit.point_id for hit in hits] == ["c_rare"]

    async def test_a_keyword_search_on_a_dense_collection_is_refused(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(collection=dense, sparse=SparseVector(indices=[1], values=[1.0]))
            )

        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    async def test_a_filtered_keyword_search_pushes_the_payload_filter_down(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        await self.seed(adapter, hybrid)
        client.hits = [[]]

        await adapter.search(
            SearchQuery(
                collection=hybrid,
                sparse=SparseVector(indices=[22], values=[1.0]),
                filters={"department": "legal"},
            )
        )

        assert client.issued("search")[0]["filter"] == '(fr_payload["department"] == "legal")'


class TestWalk:
    """The portability read side: what ``iterate_points`` carries, and what it does not."""

    async def test_a_walk_carries_the_stored_postings_ascending(
        self, adapter: MilvusAdapter, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    "c_a",
                    hybrid,
                    [1.0, 0.0, 0.0, 0.0],
                    {"n": 1},
                    SparseVector(indices=[9, 2], values=[0.5, 0.25]),
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(hybrid, with_vectors=True)]

        carried = walked[0].sparse
        assert carried is not None
        assert list(carried.indices) == [2, 9]
        assert list(carried.values) == [0.25, 0.5]
        assert walked[0].payload == {"n": 1}

    async def test_a_walk_without_vectors_carries_no_postings(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """``with_vectors`` is the one gate; a payload-only walk must not pay for postings."""
        await adapter.upsert(
            [
                Point(
                    "c_a",
                    hybrid,
                    [1.0, 0.0, 0.0, 0.0],
                    sparse=SparseVector(indices=[9], values=[0.5]),
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(hybrid)]

        assert [found.sparse for found in walked] == [None]
        assert SPARSE_FIELD not in client.issued("query_iterator")[0]["output_fields"]

    async def test_a_dense_collection_walks_without_a_sparse_field(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """Asking Milvus for a field a collection does not declare is an error, not an empty."""
        await adapter.upsert([Point("c_a", dense, [1.0, 0.0, 0.0, 0.0])])

        walked = [found async for found in adapter.iterate_points(dense, with_vectors=True)]

        assert [found.sparse for found in walked] == [None]
        assert SPARSE_FIELD not in client.issued("query_iterator")[0]["output_fields"]

    async def test_a_walk_pages_and_closes_its_cursor(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.upsert(
            [Point(f"c_{index}", dense, [1.0, 0.0, 0.0, 0.0]) for index in range(5)]
        )

        walked = [
            found async for found in adapter.iterate_points(dense, with_vectors=True, batch_size=2)
        ]

        assert len(walked) == 5
        assert client.cursors[0].closed == 1

    async def test_a_walk_reads_strongly(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        """An export that silently misses the newest points is the worst kind of backup."""
        [found async for found in adapter.iterate_points(dense)]

        assert client.issued("query_iterator")[0]["consistency_level"] == "Strong"

    async def test_walking_an_unknown_collection_raises_a_typed_error(
        self, adapter: MilvusAdapter
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            [found async for found in adapter.iterate_points("nope")]

        assert caught.value.code is ErrorCode.NOT_FOUND


class TestSnapshots:
    """A snapshot is a collection copy, so the copy is the adapter's responsibility."""

    async def test_a_snapshot_copies_the_rows(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    "c_a",
                    hybrid,
                    [1.0, 0.0, 0.0, 0.0],
                    {"n": 1},
                    SparseVector(indices=[9], values=[0.5]),
                )
            ]
        )

        taken = await adapter.snapshot(hybrid)

        copied = client.rows[snapshot_name_for("docs", taken)]
        assert [row[PK_FIELD] for row in copied] == ["c_a"]
        assert copied[0][SPARSE_FIELD] == {9: 0.5}

    async def test_a_snapshot_of_a_hybrid_collection_declares_both_legs(
        self, adapter: MilvusAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """A snapshot that drops the keyword leg restores a collection that ranks differently."""
        taken = await adapter.snapshot(hybrid)

        schema = client.issued("create_collection")[0]["schema"]
        assert {SPARSE_FIELD, TERMS_FIELD} <= set(schema.named())
        assert taken in await adapter.list_snapshots(hybrid)

    async def test_restoring_replaces_the_live_rows(
        self, adapter: MilvusAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.upsert([Point("c_a", dense, [1.0, 0.0, 0.0, 0.0])])
        taken = await adapter.snapshot(dense)
        await adapter.upsert([Point("c_b", dense, [0.0, 1.0, 0.0, 0.0])])

        await adapter.restore_snapshot(dense, taken)

        assert [row[PK_FIELD] for row in client.rows[collection_name_for("docs")]] == ["c_a"]

    async def test_restoring_an_absent_snapshot_is_a_typed_error(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.restore_snapshot(dense, "never-taken")

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_deleting_a_snapshot_is_reported_and_then_tolerated(
        self, adapter: MilvusAdapter, dense: str
    ) -> None:
        taken = await adapter.snapshot(dense)

        assert await adapter.delete_snapshot(dense, taken) is True
        assert await adapter.delete_snapshot(dense, taken) is False


class TestHealthAndClose:
    """A probe renders a failure; it never becomes one."""

    async def test_a_reachable_server_is_healthy(self, adapter: MilvusAdapter) -> None:
        status = await adapter.health()

        assert status.healthy is True
        assert status.latency_ms is not None

    async def test_an_unreachable_server_is_reported_not_raised(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        client.fails["get_server_version"] = OSError("connection refused")

        status = await adapter.health()

        assert status.healthy is False
        assert status.detail is not None

    async def test_closing_releases_the_channel_once(
        self, adapter: MilvusAdapter, client: FakeClient
    ) -> None:
        await adapter.close()
        await adapter.close()

        assert len(client.issued("close")) == 1


class TestDriverAbsence:
    """A missing driver is a configuration error with an install hint, not an ImportError."""

    def test_a_missing_driver_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.milvus._DRIVER_IMPORT_ERROR",
            ImportError("No module named 'pymilvus'"),
        )

        with pytest.raises(FasterRagError) as caught:
            MilvusAdapter(settings())

        assert caught.value.code is ErrorCode.CONFIG_INVALID
        assert "fasterrag[milvus]" in caught.value.detail
