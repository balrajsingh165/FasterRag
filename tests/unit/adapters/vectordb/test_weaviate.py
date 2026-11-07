"""Weaviate adapter unit tests, and an explicit statement of what they cannot prove.

There is no Weaviate on this machine and no Docker daemon to start one (B7.2), so the
integration contract suite under ``tests/integration/weaviate`` skips in full. These cases
run against a fake client and therefore prove exactly one class of thing: that the adapter
*issues the calls it believes it issues* — the properties it declares, the distance metric
it asks for, the filter objects it pushes down, the object ids it derives, the BM25
arithmetic it performs over what comes back, and the shape it converts results into.

They cannot prove that a real Weaviate agrees with the fake about what those calls do.
Whether the batch endpoint overwrites an object already holding a derived id, whether
auto-schema creates a flattened payload property with a usable type, whether
``ContainsAny`` over an ``INT_ARRAY`` matches the way the fake matches, whether
``fetch_objects`` paging with an offset is stable under concurrent writes, and whether a
``near_vector`` hit reports cosine *distance* rather than similarity are all assertions
about the server, and no test here touches a server.

The fake does evaluate real ``weaviate.classes.query.Filter`` objects rather than
recording them, because a filter the adapter builds wrongly and the fake ignores is a test
that passes for the wrong reason. Where a pure function can be pinned with no client at
all — name derivation, id derivation, payload splitting, filter translation, error
translation, BM25 ranking — it is, because those are the parts a fake cannot flatter.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from weaviate.classes.query import Filter as WeaviateFilter
from weaviate.collections.classes.filters import _FilterAnd, _FilterOr, _FilterValue
from weaviate.exceptions import (
    AuthenticationFailedError,
    ObjectAlreadyExistsError,
    UnexpectedStatusCodeError,
    WeaviateBaseError,
    WeaviateInvalidInputError,
    WeaviateQueryError,
)

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    PointUpdate,
    SearchQuery,
    SparseVector,
)
from fasterrag.adapters.vectordb.weaviate import (
    ALIAS_COLLECTION,
    EXTRA_PROPERTY,
    PK_PROPERTY,
    TERMS_PROPERTY,
    WEIGHTS_PROPERTY,
    WeaviateAdapter,
    _idf,
    _payload_of,
    _properties_for,
    _score_of,
    _to_sparse,
    alias_uuid_for,
    collection_name_for,
    point_uuid_for,
    snapshot_name_for,
    to_weaviate_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError

DIMENSIONS = 4
WEAVIATE_NAME = r"^[A-Z][_0-9A-Za-z]*$"


def settings(**vector_db: Any) -> Settings:
    """Return settings selecting Weaviate, with no credential named."""
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "weaviate",
                "mode": "external",
                "host": "localhost",
                "port": 8080,
                "grpc_port": 50051,
                "api_key_env": None,
                **vector_db,
            }
        }
    )


def matches(where: Any, properties: dict[str, Any], key: str) -> bool:
    """Evaluate one real Weaviate filter object against one fake object."""
    if where is None:
        return True
    if isinstance(where, _FilterAnd):
        return all(matches(member, properties, key) for member in where.filters)
    if isinstance(where, _FilterOr):
        return any(matches(member, properties, key) for member in where.filters)
    if not isinstance(where, _FilterValue):
        raise AssertionError(f"the adapter built a filter the fake cannot evaluate: {where!r}")

    held: Any = key if where.target == "_id" else properties.get(str(where.target))
    wanted: Any = where.value
    operator = where.operator.name
    if operator == "CONTAINS_ANY":
        candidates = list(wanted) if isinstance(wanted, list) else [wanted]
        if isinstance(held, list):
            return any(member in held for member in candidates)
        return bool(held in candidates)
    if held is None:
        return operator == "NOT_EQUAL"

    comparisons = {
        "EQUAL": held == wanted,
        "NOT_EQUAL": held != wanted,
        "GREATER_THAN": held > wanted,
        "GREATER_THAN_EQUAL": held >= wanted,
        "LESS_THAN": held < wanted,
        "LESS_THAN_EQUAL": held <= wanted,
    }
    if operator not in comparisons:
        raise AssertionError(f"the adapter used an operator the fake cannot evaluate: {operator}")
    return bool(comparisons[operator])


@dataclass
class FakeMetadata:
    """The subset of ``MetadataReturn`` the adapter reads."""

    distance: float | None = None


@dataclass
class FakeObject:
    """One stored object, in the shape the client hands back."""

    uuid: str
    properties: dict[str, Any]
    vector: Any = None
    metadata: FakeMetadata = field(default_factory=FakeMetadata)


@dataclass
class FakeProperty:
    """One declared property, as ``config.get(simple=True)`` reports it."""

    name: str
    data_type: Any = None


@dataclass
class FakeConfig:
    """One collection's configuration, as the client reports it."""

    name: str
    description: str
    properties: list[FakeProperty]
    metric: Any = None
    shards: Any = None
    replication: Any = None


@dataclass
class FakeReturn:
    """A query result carrying objects."""

    objects: list[FakeObject]


@dataclass
class FakeAggregate:
    """An aggregate result carrying a total."""

    total_count: int


@dataclass
class FakeBatchReturn:
    """A batch write result, carrying whatever the write refused."""

    errors: dict[int, Any] = field(default_factory=dict)


@dataclass
class FakeError:
    """One per-object batch failure."""

    message: str


class FakeStore:
    """One Weaviate collection's objects and declared configuration."""

    def __init__(self, config: FakeConfig) -> None:
        self.config_value = config
        self.objects: dict[str, FakeObject] = {}


class FakeData:
    """The ``collection.data`` surface the adapter uses."""

    def __init__(self, client: FakeClient, store: FakeStore) -> None:
        self._client = client
        self._store = store

    def insert_many(self, objects: Sequence[Any]) -> FakeBatchReturn:
        self._client.record("insert_many", collection=self._store.config_value.name)
        errors: dict[int, Any] = {}
        for position, entry in enumerate(objects):
            key = str(entry.uuid)
            if key in self._store.objects and self._client.batch_refuses_existing:
                errors[position] = FakeError("id 'x' already exists")
                continue
            self._store.objects[key] = FakeObject(
                uuid=key, properties=dict(entry.properties), vector=list(entry.vector or [])
            )
        return FakeBatchReturn(errors=errors)

    def replace(self, uuid: str, properties: dict[str, Any], vector: Any = None) -> None:
        self._client.record("replace", collection=self._store.config_value.name)
        self._store.objects[str(uuid)] = FakeObject(
            uuid=str(uuid), properties=dict(properties), vector=list(vector or [])
        )

    def update(self, uuid: str, properties: dict[str, Any]) -> None:
        self._client.record("update", collection=self._store.config_value.name)
        held = self._store.objects[str(uuid)]
        held.properties.update(properties)

    def delete_by_id(self, uuid: str) -> bool:
        self._client.record("delete_by_id", collection=self._store.config_value.name)
        return self._store.objects.pop(str(uuid), None) is not None

    def delete_many(self, where: Any) -> None:
        self._client.record("delete_many", collection=self._store.config_value.name, where=where)
        doomed = [
            key for key, held in self._store.objects.items() if matches(where, held.properties, key)
        ]
        for key in doomed:
            del self._store.objects[key]


class FakeQuery:
    """The ``collection.query`` surface the adapter uses."""

    def __init__(self, client: FakeClient, store: FakeStore) -> None:
        self._client = client
        self._store = store

    def near_vector(
        self,
        near_vector: Sequence[float],
        limit: int = 10,
        filters: Any = None,
        include_vector: bool = False,
        return_metadata: Any = None,
        return_properties: Any = None,
    ) -> FakeReturn:
        self._client.record(
            "near_vector",
            collection=self._store.config_value.name,
            asked=list(near_vector),
            filters=filters,
            include_vector=include_vector,
            return_properties=return_properties,
        )
        allowed = [
            held
            for key, held in self._store.objects.items()
            if matches(filters, held.properties, key)
        ]
        ranked = sorted(allowed, key=lambda held: _measured(near_vector, held.vector))
        return FakeReturn(
            objects=[
                _rendered(held, _measured(near_vector, held.vector), include_vector)
                for held in ranked[:limit]
            ]
        )

    def fetch_objects(
        self,
        filters: Any = None,
        limit: int | None = None,
        offset: int = 0,
        include_vector: bool = False,
    ) -> FakeReturn:
        self._client.record(
            "fetch_objects",
            collection=self._store.config_value.name,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        allowed = [
            _rendered(held, None, include_vector)
            for key, held in self._store.objects.items()
            if matches(filters, held.properties, key)
        ]
        window = allowed[offset:]
        return FakeReturn(objects=window[:limit] if limit is not None else window)

    def fetch_object_by_id(self, uuid: str) -> FakeObject | None:
        self._client.record("fetch_object_by_id", collection=self._store.config_value.name)
        held = self._store.objects.get(str(uuid))
        return None if held is None else _rendered(held, None, True)


class FakeAggregation:
    """The ``collection.aggregate`` surface the adapter uses."""

    def __init__(self, client: FakeClient, store: FakeStore) -> None:
        self._client = client
        self._store = store

    def over_all(self, total_count: bool = True, filters: Any = None) -> FakeAggregate:
        self._client.record("over_all", collection=self._store.config_value.name)
        return FakeAggregate(
            total_count=sum(
                1
                for key, held in self._store.objects.items()
                if matches(filters, held.properties, key)
            )
        )


class FakeConfigSurface:
    """The ``collection.config`` surface the adapter uses."""

    def __init__(self, client: FakeClient, store: FakeStore) -> None:
        self._client = client
        self._store = store

    def get(self, simple: bool = False) -> FakeConfig:
        self._client.record("config_get", collection=self._store.config_value.name)
        return self._store.config_value


class FakeCollection:
    """One collection handle, with the four surfaces the adapter touches."""

    def __init__(self, client: FakeClient, store: FakeStore) -> None:
        self._client = client
        self._store = store
        self.data = FakeData(client, store)
        self.query = FakeQuery(client, store)
        self.aggregate = FakeAggregation(client, store)
        self.config = FakeConfigSurface(client, store)

    def iterator(
        self, include_vector: bool = False, cache_size: int | None = None
    ) -> Iterator[Any]:
        self._client.record(
            "iterator", collection=self._store.config_value.name, cache_size=cache_size
        )
        return iter(
            [_rendered(held, None, include_vector) for held in self._store.objects.values()]
        )


class FakeCollections:
    """The ``client.collections`` surface the adapter uses."""

    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def create(
        self,
        name: str,
        description: str | None = None,
        properties: Sequence[Any] | None = None,
        vector_config: Any = None,
        sharding_config: Any = None,
        replication_config: Any = None,
    ) -> None:
        self._client.record("create", name=name, vector_config=vector_config)
        if name in self._client.stores:
            raise ObjectAlreadyExistsError(f"collection {name} already exists")
        self._client.stores[name] = FakeStore(
            FakeConfig(
                name=name,
                description=description or "",
                properties=[
                    FakeProperty(name=str(entry.name), data_type=getattr(entry, "dataType", None))
                    for entry in properties or []
                ],
                metric=vector_config,
                shards=sharding_config,
                replication=replication_config,
            )
        )

    def exists(self, name: str) -> bool:
        self._client.record("exists", name=name)
        return name in self._client.stores

    def get(self, name: str) -> FakeCollection:
        store = self._client.stores.get(name)
        if store is None:
            raise WeaviateQueryError(f"collection {name} does not exist", "GRPC")
        return FakeCollection(self._client, store)

    def delete(self, name: str) -> None:
        self._client.record("delete", name=name)
        self._client.stores.pop(name, None)

    def list_all(self, simple: bool = True) -> dict[str, FakeConfig]:
        self._client.record("list_all")
        return {name: store.config_value for name, store in self._client.stores.items()}


class FakeClient:
    """An in-memory stand-in recording every call the adapter makes.

    Structure is modelled faithfully — collections, declared properties, objects,
    filters. Ranking is modelled only as far as ordering: a squared distance stands in
    for HNSW, since simulating ANN proves the simulation.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stores: dict[str, FakeStore] = {}
        self.collections = FakeCollections(self)
        self.closed = 0
        self.meta = {"version": "1.34.0"}
        self.batch_refuses_existing = False
        self.fails: dict[str, BaseException] = {}

    def record(self, call: str, **arguments: Any) -> None:
        failure = self.fails.get(call)
        self.calls.append((call, arguments))
        if failure is not None:
            raise failure

    def issued(self, call: str) -> list[dict[str, Any]]:
        return [arguments for called, arguments in self.calls if called == call]

    def get_meta(self) -> dict[str, Any]:
        self.record("get_meta")
        return self.meta

    def close(self) -> None:
        self.closed += 1


def _measured(asked: Sequence[float], stored: Any) -> float:
    """Return a stand-in distance that orders hits the way a cosine index would."""
    held = list(stored or [])
    if len(held) != len(asked):
        return float("inf")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(asked, held, strict=True))


def _rendered(held: FakeObject, measured: float | None, include_vector: bool) -> FakeObject:
    """Return the object as the client would hand it back for one query."""
    return FakeObject(
        uuid=held.uuid,
        properties=dict(held.properties),
        vector={"default": list(held.vector or [])} if include_vector else None,
        metadata=FakeMetadata(distance=measured),
    )


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def adapter(client: FakeClient) -> WeaviateAdapter:
    built = WeaviateAdapter(settings())
    built._client = client
    return built


@pytest.fixture
async def dense(adapter: WeaviateAdapter) -> str:
    await adapter.create_collection(CollectionSpec(name="notes-dense", dimensions=DIMENSIONS))
    return "notes-dense"


@pytest.fixture
async def hybrid(adapter: WeaviateAdapter) -> str:
    await adapter.create_collection(
        CollectionSpec(name="notes-hybrid", dimensions=DIMENSIONS, sparse=True)
    )
    return "notes-hybrid"


def point(point_id: str, collection: str, vector: list[float], **payload: Any) -> Point:
    return Point(point_id=point_id, collection=collection, vector=vector, payload=payload)


VECTORS: dict[str, list[float]] = {
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0],
    "c": [0.0, 0.0, 1.0, 0.0],
}


class TestDerivedNames:
    """A legal fasterRag collection name is not necessarily a legal Weaviate one."""

    def test_a_collection_name_starts_with_a_capital(self) -> None:
        import re

        assert re.match(WEAVIATE_NAME, collection_name_for("notes"))

    def test_a_hyphen_never_survives_into_an_identifier(self) -> None:
        assert "-" not in collection_name_for("blue-green-2026")

    def test_a_name_starting_with_a_digit_yields_a_legal_identifier(self) -> None:
        import re

        assert re.match(WEAVIATE_NAME, collection_name_for("2026-archive"))

    def test_the_name_is_stable_across_calls(self) -> None:
        assert collection_name_for("notes") == collection_name_for("notes")

    def test_names_differing_only_in_case_never_share_a_collection(self) -> None:
        assert collection_name_for("Notes") != collection_name_for("notes")

    def test_a_name_of_pure_punctuation_still_yields_a_legal_identifier(self) -> None:
        import re

        assert re.match(WEAVIATE_NAME, collection_name_for("---"))

    def test_a_collection_is_never_mistaken_for_a_snapshot(self) -> None:
        assert collection_name_for("notes") != snapshot_name_for("notes", "notes")

    def test_one_snapshot_name_under_two_collections_is_two_collections(self) -> None:
        assert snapshot_name_for("a", "nightly") != snapshot_name_for("b", "nightly")

    def test_a_point_id_derives_a_stable_uuid(self) -> None:
        assert point_uuid_for("c_1") == point_uuid_for("c_1")
        assert uuid.UUID(point_uuid_for("c_1")).version == 5

    def test_two_point_ids_never_collide(self) -> None:
        assert point_uuid_for("c_1") != point_uuid_for("c_2")

    def test_an_alias_and_a_point_of_one_name_derive_different_ids(self) -> None:
        """The two namespaces exist so an alias record cannot overwrite a chunk."""
        assert alias_uuid_for("live") != point_uuid_for("live")


class TestFilterTranslation:
    """Every supported operator, and what happens to the ones that are not."""

    def test_no_filter_renders_as_no_filter(self) -> None:
        assert to_weaviate_filter(None) is None
        assert to_weaviate_filter({}) is None

    def test_a_scalar_renders_as_equality(self) -> None:
        built = to_weaviate_filter({"department": "legal"})
        assert matches(built, {"department": "legal"}, "k")
        assert not matches(built, {"department": "finance"}, "k")

    def test_a_range_operator_renders_as_its_comparison(self) -> None:
        built = to_weaviate_filter({"year": {"$gte": 2024}})
        assert matches(built, {"year": 2024}, "k")
        assert not matches(built, {"year": 2023}, "k")

    def test_set_membership_expands_into_a_disjunction_of_equalities(self) -> None:
        """``contains_any`` is Weaviate's array operator; a scalar needs ``any_of``."""
        built = to_weaviate_filter({"department": {"$in": ["finance", "hr"]}})
        assert isinstance(built, _FilterOr)
        assert matches(built, {"department": "hr"}, "k")
        assert not matches(built, {"department": "legal"}, "k")

    def test_set_exclusion_expands_into_a_conjunction_of_inequalities(self) -> None:
        built = to_weaviate_filter({"department": {"$nin": ["finance", "hr"]}})
        assert isinstance(built, _FilterAnd)
        assert matches(built, {"department": "legal"}, "k")
        assert not matches(built, {"department": "hr"}, "k")

    def test_several_keys_are_conjoined(self) -> None:
        built = to_weaviate_filter({"department": "legal", "year": {"$lt": 2025}})
        assert matches(built, {"department": "legal", "year": 2024}, "k")
        assert not matches(built, {"department": "legal", "year": 2026}, "k")

    def test_a_reserved_key_is_refused(self) -> None:
        """``fr_terms`` is the adapter's own property, never a caller's payload key."""
        with pytest.raises(FasterRagError) as caught:
            to_weaviate_filter({TERMS_PROPERTY: 1})
        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    def test_a_key_that_is_not_a_legal_property_name_is_refused_not_ignored(self) -> None:
        """Such a key lives in the JSON remainder, so a pushed-down filter would miss."""
        with pytest.raises(FasterRagError) as caught:
            to_weaviate_filter({"source-uri": "x"})
        assert caught.value.code is ErrorCode.VALIDATION_FAILED
        assert "source-uri" in caught.value.detail

    def test_an_unsupported_operator_is_refused_by_the_shared_grammar(self) -> None:
        with pytest.raises(FasterRagError) as caught:
            to_weaviate_filter({"year": {"$regex": "2024"}})
        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    def test_set_membership_without_a_list_is_refused(self) -> None:
        with pytest.raises(FasterRagError):
            to_weaviate_filter({"department": {"$in": "finance"}})


class TestPayloadSplitting:
    """A payload must round-trip exactly, whatever Weaviate will accept as a property."""

    def test_a_scalar_key_is_flattened_so_it_can_be_filtered(self) -> None:
        assert _properties_for({"department": "legal"})["department"] == "legal"

    def test_a_nested_value_goes_into_the_json_remainder(self) -> None:
        built = _properties_for({"span": {"start": 0, "end": 9}})
        assert "span" not in built
        assert "start" in built[EXTRA_PROPERTY]

    def test_an_illegal_property_name_goes_into_the_remainder(self) -> None:
        built = _properties_for({"source-uri": "file:///a"})
        assert "source-uri" not in built
        assert "source-uri" in built[EXTRA_PROPERTY]

    def test_a_key_colliding_with_the_reserved_namespace_goes_into_the_remainder(self) -> None:
        built = _properties_for({PK_PROPERTY: "spoofed"})
        assert built[PK_PROPERTY] != "spoofed" if PK_PROPERTY in built else True
        assert PK_PROPERTY in built[EXTRA_PROPERTY]

    def test_a_null_goes_into_the_remainder_rather_than_vanishing(self) -> None:
        built = _properties_for({"section": None})
        assert "section" in built[EXTRA_PROPERTY]

    def test_a_mixed_payload_round_trips_exactly(self) -> None:
        original = {
            "department": "legal",
            "year": 2024,
            "reviewed": True,
            "score": 0.5,
            "span": {"start": 0, "end": 9},
            "tags": ["a", "b"],
            "section": None,
            "source-uri": "file:///a",
        }
        stored = {**_properties_for(original), PK_PROPERTY: "c_1"}

        assert _payload_of(stored) == original

    def test_the_reserved_properties_never_reach_a_caller(self) -> None:
        stored = {
            PK_PROPERTY: "c_1",
            TERMS_PROPERTY: [1],
            WEIGHTS_PROPERTY: [1.0],
            EXTRA_PROPERTY: "",
            "department": "legal",
        }
        assert _payload_of(stored) == {"department": "legal"}


class TestConversions:
    """The pure translations a fake cannot flatter."""

    def test_a_posting_pair_decodes_ascending_by_term(self) -> None:
        carried = _to_sparse({TERMS_PROPERTY: [9, 2, 5], WEIGHTS_PROPERTY: [0.9, 0.2, 0.5]})
        assert carried is not None
        assert list(carried.indices) == [2, 5, 9]
        assert list(carried.values) == [0.2, 0.5, 0.9]

    def test_an_absent_or_empty_posting_list_reads_as_no_keyword_leg(self) -> None:
        assert _to_sparse({}) is None
        assert _to_sparse({TERMS_PROPERTY: [], WEIGHTS_PROPERTY: []}) is None

    def test_mismatched_postings_are_refused_rather_than_truncated(self) -> None:
        with pytest.raises(FasterRagError) as caught:
            _to_sparse({TERMS_PROPERTY: [1, 2], WEIGHTS_PROPERTY: [1.0]})
        assert caught.value.code is ErrorCode.INTERNAL

    def test_a_cosine_distance_becomes_a_similarity(self) -> None:
        assert _score_of("cosine", 0.25) == pytest.approx(0.75)

    def test_a_negated_dot_product_becomes_the_dot_product(self) -> None:
        assert _score_of("dot", -3.0) == pytest.approx(3.0)

    def test_a_squared_euclid_distance_is_negated(self) -> None:
        assert _score_of("euclid", 4.0) == pytest.approx(-4.0)

    def test_idf_matches_the_formula_every_backend_uses(self) -> None:
        assert _idf(10, 2) == pytest.approx(math.log(1 + (10 - 2 + 0.5) / (2 + 0.5)))

    def test_idf_stays_positive_for_a_term_in_every_document(self) -> None:
        assert _idf(5, 5) > 0


class TestErrorTranslation:
    """Every vendor failure crosses the boundary as a typed error, never as itself."""

    @pytest.fixture
    def translate(self, adapter: WeaviateAdapter) -> Any:
        return lambda exc: adapter._translate(exc, "search")

    def test_a_rejected_credential_is_not_retryable(self, translate: Any) -> None:
        caught = translate(AuthenticationFailedError("bad key"))
        assert caught.code is ErrorCode.VECTOR_DB_AUTH_FAILED
        assert caught.retryable is False

    def test_an_anonymous_rejection_by_wording_maps_to_the_auth_code(self, translate: Any) -> None:
        caught = translate(WeaviateQueryError("anonymous access is disabled", "GRPC"))
        assert caught.code is ErrorCode.VECTOR_DB_AUTH_FAILED

    def test_an_existing_object_maps_to_conflict(self, translate: Any) -> None:
        assert translate(ObjectAlreadyExistsError("x already exists")).code is ErrorCode.CONFLICT

    def test_a_missing_resource_by_wording_maps_to_not_found(self, translate: Any) -> None:
        caught = translate(WeaviateQueryError("class Frag_x does not exist", "GRPC"))
        assert caught.code is ErrorCode.NOT_FOUND

    def test_an_invalid_input_is_not_retryable(self, translate: Any) -> None:
        caught = translate(WeaviateInvalidInputError("limit must be positive"))
        assert caught.code is ErrorCode.VALIDATION_FAILED
        assert caught.retryable is False

    def test_a_socket_failure_is_retryable(self, translate: Any) -> None:
        assert translate(ConnectionResetError("reset")).retryable is True

    def test_a_client_side_value_error_is_not_retryable(self, translate: Any) -> None:
        caught = translate(ValueError("not a valid uuid"))
        assert caught.code is ErrorCode.VALIDATION_FAILED
        assert caught.retryable is False

    def test_an_unclassified_server_error_stays_retryable(self, translate: Any) -> None:
        caught = translate(WeaviateQueryError("shard is backing up", "GRPC"))
        assert caught.code is ErrorCode.EMBED_PROVIDER_ERROR
        assert caught.retryable is True

    def test_every_translated_error_is_a_fasterrag_error(self, translate: Any) -> None:
        for exc in (
            AuthenticationFailedError("a"),
            ObjectAlreadyExistsError("b"),
            WeaviateQueryError("c", "GRPC"),
            ValueError("d"),
            OSError("e"),
        ):
            assert isinstance(translate(exc), FasterRagError)
            assert not isinstance(translate(exc), WeaviateBaseError)


class TestClientConstruction:
    """What the adapter asks the client for, and where the credential comes from."""

    def test_construction_opens_no_connection(self) -> None:
        assert WeaviateAdapter(settings())._client is None

    def test_both_ports_are_carried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weaviate needs HTTP and gRPC; the v4 client refuses to work without both."""
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.weaviate.weaviate.connect_to_custom",
            lambda **kwargs: seen.update(kwargs) or object(),
        )

        WeaviateAdapter(settings(port=8080, grpc_port=50051))._build_client()

        assert seen["http_port"] == 8080
        assert seen["grpc_port"] == 50051
        assert seen["http_secure"] is False
        assert seen["grpc_secure"] is False

    def test_tls_applies_to_both_channels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.weaviate.weaviate.connect_to_custom",
            lambda **kwargs: seen.update(kwargs) or object(),
        )

        WeaviateAdapter(settings(https=True))._build_client()

        assert seen["http_secure"] is True
        assert seen["grpc_secure"] is True

    def test_the_key_is_read_from_the_named_variable_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.weaviate.weaviate.connect_to_custom",
            lambda **kwargs: seen.update(kwargs) or object(),
        )
        monkeypatch.setenv("WEAVIATE_TEST_KEY", "secret")
        monkeypatch.setenv("WEAVIATE_API_KEY", "the-wrong-one")

        WeaviateAdapter(settings(api_key_env="WEAVIATE_TEST_KEY"))._build_client()

        assert seen["auth_credentials"] is not None
        WeaviateAdapter(settings(api_key_env=None))._build_client()
        assert seen["auth_credentials"] is None


class TestCollectionLifecycle:
    """What a created collection declares, and what an existing one refuses."""

    async def test_a_dense_collection_declares_only_the_two_base_properties(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        await adapter.create_collection(CollectionSpec(name="notes", dimensions=DIMENSIONS))

        declared = {
            entry.name
            for entry in client.stores[collection_name_for("notes")].config_value.properties
        }
        assert declared == {PK_PROPERTY, EXTRA_PROPERTY}

    async def test_a_hybrid_collection_declares_both_posting_properties(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        await adapter.create_collection(
            CollectionSpec(name="notes", dimensions=DIMENSIONS, sparse=True)
        )

        declared = {
            entry.name
            for entry in client.stores[collection_name_for("notes")].config_value.properties
        }
        assert {TERMS_PROPERTY, WEIGHTS_PROPERTY} <= declared

    async def test_the_declared_width_is_recorded_because_weaviate_does_not_declare_it(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        """HNSW infers its width from the first vector, so the record is the only check."""
        await adapter.create_collection(CollectionSpec(name="notes", dimensions=DIMENSIONS))

        description = client.stores[collection_name_for("notes")].config_value.description
        assert f'"m":{DIMENSIONS}' in description

    async def test_creating_an_existing_collection_is_idempotent(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        spec = CollectionSpec(name="notes", dimensions=DIMENSIONS)
        await adapter.create_collection(spec)
        await adapter.create_collection(spec)

        assert len(client.issued("create")) == 1

    async def test_a_conflicting_width_is_refused(self, adapter: WeaviateAdapter) -> None:
        await adapter.create_collection(CollectionSpec(name="notes", dimensions=DIMENSIONS))

        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(CollectionSpec(name="notes", dimensions=DIMENSIONS + 1))
        assert caught.value.code is ErrorCode.CONFLICT

    async def test_a_conflicting_distance_is_refused(self, adapter: WeaviateAdapter) -> None:
        await adapter.create_collection(CollectionSpec(name="notes", dimensions=DIMENSIONS))

        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(
                CollectionSpec(name="notes", dimensions=DIMENSIONS, distance="dot")
            )
        assert caught.value.code is ErrorCode.CONFLICT

    async def test_bolting_a_keyword_leg_onto_a_dense_collection_is_refused(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(
                CollectionSpec(name=dense, dimensions=DIMENSIONS, sparse=True)
            )
        assert caught.value.code is ErrorCode.CONFLICT

    async def test_the_listing_reports_the_logical_name_and_a_count(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"])])

        listed = {info.name: info for info in await adapter.list_collections()}

        assert listed[dense].vectors == 1
        assert listed[dense].dimensions == DIMENSIONS

    async def test_a_foreign_collection_is_not_listed(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        """Weaviate is somebody else's database too; nothing unmarked is ours."""
        client.stores["Frag_someone_elses"] = FakeStore(
            FakeConfig(name="Frag_someone_elses", description="", properties=[])
        )

        assert [info.name for info in await adapter.list_collections()] == [dense]

    async def test_a_snapshot_is_not_listed_as_a_collection(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        taken = await adapter.snapshot(dense)

        assert taken not in {info.name for info in await adapter.list_collections()}

    async def test_dropping_an_absent_collection_is_not_an_error(
        self, adapter: WeaviateAdapter
    ) -> None:
        assert await adapter.drop_collection("never-existed") is False

    async def test_dropping_a_collection_takes_its_snapshots_and_aliases(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.snapshot(dense)
        await adapter.set_alias("live", dense)

        assert await adapter.drop_collection(dense) is True
        assert collection_name_for(dense) not in client.stores
        assert not [name for name in client.stores if name.startswith("Fragsnap_")]
        assert await adapter.alias_target("live") is None


class TestAliases:
    """Aliases are records, so every alias property has to be built here."""

    async def test_an_alias_resolves_to_its_collection(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.set_alias("live", dense)

        assert await adapter.alias_target("live") == dense

    async def test_an_unset_alias_resolves_to_nothing(self, adapter: WeaviateAdapter) -> None:
        assert await adapter.alias_target("never-set") is None

    async def test_repointing_an_alias_replaces_its_target_in_one_record(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name="green", dimensions=DIMENSIONS))
        await adapter.set_alias("live", dense)
        await adapter.set_alias("live", "green")

        assert await adapter.alias_target("live") == "green"
        assert len(client.stores[ALIAS_COLLECTION].objects) == 1

    async def test_deleting_an_alias_leaves_the_collection(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.set_alias("live", dense)

        assert await adapter.delete_alias("live") is True
        assert await adapter.alias_target("live") is None
        assert dense in {info.name for info in await adapter.list_collections()}

    async def test_deleting_an_absent_alias_is_not_an_error(self, adapter: WeaviateAdapter) -> None:
        assert await adapter.delete_alias("never-set") is False

    async def test_an_alias_cannot_shadow_a_collection(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.set_alias(dense, dense)
        assert caught.value.code is ErrorCode.CONFLICT

    async def test_an_alias_is_searchable_as_if_it_were_the_collection(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"])])
        await adapter.set_alias("live", dense)

        hits = await adapter.search(SearchQuery(collection="live", vector=VECTORS["a"], limit=3))

        assert [hit.point_id for hit in hits] == ["c_a"]

    async def test_the_alias_collection_is_not_listed_as_a_collection(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.set_alias("live", dense)

        assert [info.name for info in await adapter.list_collections()] == [dense]


class TestWrites:
    """What an upsert writes, and what it refuses to write."""

    async def test_a_point_is_stored_under_its_derived_id_with_its_own_id_beside_it(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"], department="legal")])

        stored = client.stores[collection_name_for(dense)].objects
        assert set(stored) == {point_uuid_for("c_a")}
        assert stored[point_uuid_for("c_a")].properties[PK_PROPERTY] == "c_a"

    async def test_upsert_is_idempotent(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        batch = [point("c_a", dense, VECTORS["a"])]
        await adapter.upsert(batch)
        await adapter.upsert(batch)

        assert len(client.stores[collection_name_for(dense)].objects) == 1

    async def test_a_batch_that_refuses_an_existing_id_falls_back_to_replace(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        """Whether Weaviate's batch overwrites is the server's business, not ours."""
        await adapter.upsert([point("c_a", dense, VECTORS["a"], version=1)])
        client.batch_refuses_existing = True

        await adapter.upsert([point("c_a", dense, VECTORS["a"], version=2)])

        stored = client.stores[collection_name_for(dense)].objects[point_uuid_for("c_a")]
        assert stored.properties["version"] == 2
        assert len(client.issued("replace")) == 1

    async def test_a_batch_that_overwrites_never_falls_back(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"])])
        await adapter.upsert([point("c_a", dense, VECTORS["a"])])

        assert client.issued("replace") == []

    async def test_upsert_reports_how_many_points_it_wrote(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        result = await adapter.upsert(
            [point(f"c_{index}", dense, VECTORS["a"]) for index in range(5)]
        )

        assert result.upserted == 5

    async def test_an_empty_upsert_touches_nothing(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        assert (await adapter.upsert([])).upserted == 0
        assert client.issued("insert_many") == []

    async def test_a_dimension_mismatch_is_refused_before_the_server_sees_it(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        with pytest.raises(EmbedError) as caught:
            await adapter.upsert([point("c_bad", dense, [1.0, 0.0])])

        assert caught.value.retryable is False
        assert str(DIMENSIONS) in caught.value.detail
        assert client.issued("insert_many") == []

    async def test_postings_for_a_dense_collection_are_refused(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        with pytest.raises(EmbedError) as caught:
            await adapter.upsert(
                [
                    Point(
                        point_id="c_a",
                        collection=dense,
                        vector=VECTORS["a"],
                        sparse=encode_document("termination notice"),
                    )
                ]
            )
        assert caught.value.retryable is False

    async def test_postings_are_written_ascending_by_term(
        self, adapter: WeaviateAdapter, client: FakeClient, hybrid: str
    ) -> None:
        written = SparseVector(indices=[9, 2, 5], values=[0.9, 0.2, 0.5])
        await adapter.upsert(
            [Point(point_id="c_a", collection=hybrid, vector=VECTORS["a"], sparse=written)]
        )

        stored = client.stores[collection_name_for(hybrid)].objects[point_uuid_for("c_a")]
        assert stored.properties[TERMS_PROPERTY] == [2, 5, 9]
        assert stored.properties[WEIGHTS_PROPERTY] == [0.2, 0.5, 0.9]

    async def test_a_hybrid_point_without_postings_is_still_writable(
        self, adapter: WeaviateAdapter, client: FakeClient, hybrid: str
    ) -> None:
        await adapter.upsert([point("c_a", hybrid, VECTORS["a"])])

        stored = client.stores[collection_name_for(hybrid)].objects[point_uuid_for("c_a")]
        assert stored.properties[TERMS_PROPERTY] == []

    async def test_an_unknown_collection_raises_a_typed_error(
        self, adapter: WeaviateAdapter
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.upsert([point("c_a", "never-created", VECTORS["a"])])
        assert caught.value.code is ErrorCode.NOT_FOUND


class TestDenseSearch:
    """The dense leg, and the filters it pushes down."""

    async def seed(self, adapter: WeaviateAdapter, collection: str) -> None:
        await adapter.upsert(
            [
                point("c_a", collection, VECTORS["a"], department="legal", year=2024),
                point("c_b", collection, VECTORS["b"], department="legal", year=2020),
                point("c_c", collection, VECTORS["c"], department="finance", year=2024),
            ]
        )

    async def test_a_hit_carries_its_own_id_and_payload(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(SearchQuery(collection=dense, vector=VECTORS["a"], limit=3))

        assert hits[0].point_id == "c_a"
        assert hits[0].payload["department"] == "legal"

    async def test_scores_rank_higher_is_better(self, adapter: WeaviateAdapter, dense: str) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(SearchQuery(collection=dense, vector=VECTORS["a"], limit=3))

        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)

    async def test_the_limit_is_respected(self, adapter: WeaviateAdapter, dense: str) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(SearchQuery(collection=dense, vector=VECTORS["a"], limit=2))

        assert len(hits) == 2

    async def test_a_zero_limit_asks_the_server_nothing(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        asked = SearchQuery(collection=dense, vector=VECTORS["a"], limit=0)

        assert await adapter.search(asked) == []
        assert client.issued("near_vector") == []

    async def test_an_equality_filter_is_pushed_down(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(
            SearchQuery(
                collection=dense,
                vector=VECTORS["a"],
                limit=10,
                filters={"department": "finance"},
            )
        )

        assert [hit.point_id for hit in hits] == ["c_c"]

    async def test_a_range_filter_is_pushed_down(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(
            SearchQuery(
                collection=dense, vector=VECTORS["a"], limit=10, filters={"year": {"$gte": 2024}}
            )
        )

        assert {hit.point_id for hit in hits} == {"c_a", "c_c"}

    async def test_a_set_membership_filter_is_pushed_down(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(
            SearchQuery(
                collection=dense,
                vector=VECTORS["a"],
                limit=10,
                filters={"department": {"$in": ["finance", "hr"]}},
            )
        )

        assert {hit.point_id for hit in hits} == {"c_c"}

    async def test_vectors_come_back_only_when_asked(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        with_vectors = await adapter.search(
            SearchQuery(collection=dense, vector=VECTORS["a"], limit=1, with_vectors=True)
        )
        without = await adapter.search(SearchQuery(collection=dense, vector=VECTORS["a"], limit=1))

        assert with_vectors[0].vector is not None
        assert len(with_vectors[0].vector) == DIMENSIONS
        assert without[0].vector is None

    async def test_a_payload_free_search_asks_only_for_the_id(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        hits = await adapter.search(
            SearchQuery(collection=dense, vector=VECTORS["a"], limit=1, with_payload=False)
        )

        assert hits[0].payload == {}
        assert client.issued("near_vector")[-1]["return_properties"] == [PK_PROPERTY]

    async def test_searching_an_unknown_collection_raises_a_typed_error(
        self, adapter: WeaviateAdapter
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(SearchQuery(collection="never-created", vector=VECTORS["a"]))
        assert caught.value.code is ErrorCode.NOT_FOUND


class TestKeywordSearch:
    """The leg Weaviate's native BM25 is deliberately not used for."""

    async def test_a_keyword_search_finds_the_document_carrying_the_term(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("Either party may terminate with written notice."),
                ),
                Point(
                    point_id="c_payment",
                    collection=hybrid,
                    vector=VECTORS["b"],
                    sparse=encode_document("Invoices are payable within thirty days."),
                ),
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid, sparse=encode_query("terminate"), limit=5)
        )

        assert [hit.point_id for hit in hits] == ["c_notice"]

    async def test_a_keyword_search_matches_across_inflections(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        """Stemming is fasterRag's half of the split, so it works on every backend."""
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("The agreement covers terminations and notices."),
                )
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid, sparse=encode_query("terminate"), limit=5)
        )

        assert [hit.point_id for hit in hits] == ["c_notice"]

    async def test_a_rarer_term_outranks_a_common_one(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        """Without IDF these would tie; the ranking is the whole proof it is applied."""
        await adapter.upsert(
            [
                Point(
                    point_id=f"c_common_{index}",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("notice notice notice"),
                )
                for index in range(5)
            ]
            + [
                Point(
                    point_id="c_rare",
                    collection=hybrid,
                    vector=VECTORS["b"],
                    sparse=encode_document("notice indemnification"),
                )
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid, sparse=encode_query("notice indemnification"), limit=10)
        )

        assert hits[0].point_id == "c_rare"

    async def test_a_term_absent_from_the_corpus_scores_nothing(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("termination notice period"),
                )
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid, sparse=encode_query("velociraptor"), limit=5)
        )

        assert hits == []

    async def test_filters_are_pushed_down_on_the_keyword_leg_too(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_legal",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    payload={"department": "legal"},
                    sparse=encode_document("termination notice"),
                ),
                Point(
                    point_id="c_finance",
                    collection=hybrid,
                    vector=VECTORS["b"],
                    payload={"department": "finance"},
                    sparse=encode_document("termination notice"),
                ),
            ]
        )

        hits = await adapter.search(
            SearchQuery(
                collection=hybrid,
                sparse=encode_query("termination"),
                filters={"department": "finance"},
                limit=5,
            )
        )

        assert [hit.point_id for hit in hits] == ["c_finance"]

    async def test_a_filter_narrows_the_result_without_narrowing_the_corpus(
        self, adapter: WeaviateAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """IDF is the live corpus's statistic, so the count must not see the filter."""
        await adapter.upsert(
            [
                Point(
                    point_id=f"c_{index}",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    payload={"department": "legal" if index else "finance"},
                    sparse=encode_document("termination notice"),
                )
                for index in range(4)
            ]
        )

        await adapter.search(
            SearchQuery(
                collection=hybrid,
                sparse=encode_query("termination"),
                filters={"department": "finance"},
                limit=5,
            )
        )

        assert client.issued("over_all")

    async def test_a_keyword_search_on_a_dense_collection_is_refused(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(collection=dense, sparse=encode_query("notice"), limit=5)
            )
        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    async def test_the_native_bm25_endpoint_is_never_called(
        self, adapter: WeaviateAdapter, client: FakeClient, hybrid: str
    ) -> None:
        """The fake has no ``bm25``; calling it would be an AttributeError, not a pass."""
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("termination notice"),
                )
            ]
        )

        await adapter.search(SearchQuery(collection=hybrid, sparse=encode_query("notice"), limit=5))

        assert not any(called in ("bm25", "hybrid") for called, _ in client.calls)


class TestWalk:
    """What ``iterate_points`` carries, which is what an archive can carry."""

    async def test_a_walk_carries_every_point(self, adapter: WeaviateAdapter, dense: str) -> None:
        await adapter.upsert([point(f"c_{index}", dense, VECTORS["a"]) for index in range(7)])

        walked = [found async for found in adapter.iterate_points(dense, batch_size=2)]

        assert {found.point_id for found in walked} == {f"c_{index}" for index in range(7)}

    async def test_a_walk_carries_the_stored_sparse_vector_exactly(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        written = encode_document("Either party may terminate with written notice.")
        await adapter.upsert(
            [Point(point_id="c_a", collection=hybrid, vector=VECTORS["a"], sparse=written)]
        )

        walked = [found async for found in adapter.iterate_points(hybrid, with_vectors=True)]

        carried = walked[0].sparse
        assert carried is not None
        assert dict(zip(carried.indices, carried.values, strict=True)) == pytest.approx(
            dict(zip(written.indices, written.values, strict=True))
        )

    async def test_a_walk_returns_terms_ascending(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        """The ordering is the contract every backend agrees on (TASK-0291)."""
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("notice termination period written party"),
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(hybrid, with_vectors=True)]

        carried = walked[0].sparse
        assert carried is not None
        assert list(carried.indices) == sorted(carried.indices)

    async def test_a_walk_without_vectors_carries_no_sparse_vector(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    sparse=encode_document("termination notice"),
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(hybrid)]

        assert [found.sparse for found in walked] == [None]
        assert [list(found.vector) for found in walked] == [[]]

    async def test_a_dense_collection_walks_without_a_sparse_vector(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"])])

        walked = [found async for found in adapter.iterate_points(dense, with_vectors=True)]

        assert [found.sparse for found in walked] == [None]

    async def test_a_walked_point_round_trips_into_a_second_collection(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        """A portability archive is exactly this loop, so it is tested as one."""
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    payload={"department": "legal", "span": {"start": 0, "end": 9}},
                    sparse=encode_document("Either party may terminate with written notice."),
                )
            ]
        )
        query = encode_query("terminate notice")
        before = await adapter.search(SearchQuery(collection=hybrid, sparse=query, limit=5))

        copy = f"{hybrid}-copy"
        await adapter.create_collection(
            CollectionSpec(name=copy, dimensions=DIMENSIONS, sparse=True)
        )
        await adapter.upsert(
            [
                Point(
                    point_id=found.point_id,
                    collection=copy,
                    vector=found.vector,
                    payload=found.payload,
                    sparse=found.sparse,
                )
                async for found in adapter.iterate_points(hybrid, with_vectors=True)
            ]
        )
        after = await adapter.search(SearchQuery(collection=copy, sparse=query, limit=5))

        assert [hit.point_id for hit in after] == [hit.point_id for hit in before]
        assert [hit.score for hit in after] == pytest.approx([hit.score for hit in before])
        assert after[0].payload == before[0].payload


class TestUpdates:
    """A metadata merge must not disturb the vector or the postings."""

    async def test_an_update_merges_rather_than_replaces(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"], department="legal")])

        await adapter.update(
            [PointUpdate(point_id="c_a", collection=dense, payload={"reviewed": True})]
        )

        hits = await adapter.search(SearchQuery(collection=dense, vector=VECTORS["a"], limit=1))
        assert hits[0].payload == {"department": "legal", "reviewed": True}

    async def test_an_update_leaves_the_vector_and_the_postings_alone(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        written = encode_document("termination notice")
        await adapter.upsert(
            [Point(point_id="c_a", collection=hybrid, vector=VECTORS["a"], sparse=written)]
        )

        await adapter.update(
            [PointUpdate(point_id="c_a", collection=hybrid, payload={"reviewed": True})]
        )

        walked = [found async for found in adapter.iterate_points(hybrid, with_vectors=True)]
        assert list(walked[0].vector) == VECTORS["a"]
        assert walked[0].sparse is not None
        assert list(walked[0].sparse.indices) == sorted(written.indices)

    async def test_a_nested_value_survives_a_merge(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        """The remainder is rewritten whole, so a merge is where it would be lost."""
        await adapter.upsert([point("c_a", dense, VECTORS["a"], span={"start": 0, "end": 9})])

        await adapter.update(
            [PointUpdate(point_id="c_a", collection=dense, payload={"reviewed": True})]
        )

        walked = [found async for found in adapter.iterate_points(dense)]
        assert walked[0].payload == {"span": {"start": 0, "end": 9}, "reviewed": True}

    async def test_updating_an_absent_point_raises_rather_than_creating_one(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.update(
                [PointUpdate(point_id="c_missing", collection=dense, payload={"a": 1})]
            )
        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_an_empty_update_touches_nothing(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        await adapter.update([])
        assert client.issued("update") == []


class TestDeletes:
    """What a delete removes, and the one it refuses to perform."""

    async def seed(self, adapter: WeaviateAdapter, collection: str) -> None:
        await adapter.upsert(
            [
                point("c_a", collection, VECTORS["a"], department="legal"),
                point("c_b", collection, VECTORS["b"], department="legal"),
                point("c_c", collection, VECTORS["c"], department="finance"),
            ]
        )

    async def test_delete_by_ids_removes_only_those_points(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        await adapter.delete(PointSelector(collection=dense, point_ids=["c_a"]))

        walked = {found.point_id async for found in adapter.iterate_points(dense)}
        assert walked == {"c_b", "c_c"}

    async def test_delete_by_filter_removes_matching_points(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        await adapter.delete(PointSelector(collection=dense, filters={"department": "legal"}))

        walked = {found.point_id async for found in adapter.iterate_points(dense)}
        assert walked == {"c_c"}

    async def test_an_empty_id_list_deletes_nothing(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        await self.seed(adapter, dense)

        await adapter.delete(PointSelector(collection=dense, point_ids=[]))

        assert client.issued("delete_many") == []
        assert len(client.stores[collection_name_for(dense)].objects) == 3


class TestSnapshots:
    """Snapshots are a collection copy, so their contents are this adapter's problem."""

    async def test_a_snapshot_holds_what_the_collection_held(
        self, adapter: WeaviateAdapter, hybrid: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=hybrid,
                    vector=VECTORS["a"],
                    payload={"n": 1},
                    sparse=encode_document("termination notice"),
                )
            ]
        )

        taken = await adapter.snapshot(hybrid)
        await adapter.delete(PointSelector(collection=hybrid, point_ids=["c_a"]))
        await adapter.restore_snapshot(hybrid, taken)

        walked = [found async for found in adapter.iterate_points(hybrid, with_vectors=True)]
        assert [found.point_id for found in walked] == ["c_a"]
        assert walked[0].payload == {"n": 1}
        assert walked[0].sparse is not None

    async def test_a_snapshot_is_listed_and_then_is_not(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        taken = await adapter.snapshot(dense)

        assert taken in await adapter.list_snapshots(dense)
        assert await adapter.delete_snapshot(dense, taken) is True
        assert taken not in await adapter.list_snapshots(dense)
        assert await adapter.delete_snapshot(dense, taken) is False

    async def test_a_restore_replaces_rather_than_merges(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.upsert([point("c_a", dense, VECTORS["a"])])
        taken = await adapter.snapshot(dense)
        await adapter.upsert([point("c_later", dense, VECTORS["b"])])

        await adapter.restore_snapshot(dense, taken)

        walked = {found.point_id async for found in adapter.iterate_points(dense)}
        assert walked == {"c_a"}

    async def test_restoring_an_unknown_snapshot_raises_a_typed_error(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.restore_snapshot(dense, "never-taken")
        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_one_collections_snapshot_is_not_anothers(
        self, adapter: WeaviateAdapter, dense: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name="other", dimensions=DIMENSIONS))
        taken = await adapter.snapshot(dense)

        assert await adapter.list_snapshots("other") == []
        assert await adapter.delete_snapshot("other", taken) is False

    async def test_the_native_backup_api_is_never_called(
        self, adapter: WeaviateAdapter, client: FakeClient, dense: str
    ) -> None:
        """A backup module the server may not have enabled is not a snapshot."""
        await adapter.snapshot(dense)

        assert not any(called.startswith("backup") for called, _ in client.calls)


class TestHealthAndClose:
    """A probe must render a failure rather than raise one."""

    async def test_a_reachable_server_is_healthy_and_timed(self, adapter: WeaviateAdapter) -> None:
        status = await adapter.health()

        assert status.healthy is True
        assert status.latency_ms is not None

    async def test_an_unreachable_server_is_reported_rather_than_raised(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        client.fails["get_meta"] = WeaviateQueryError("connection refused", "GRPC")

        status = await adapter.health()

        assert status.healthy is False
        assert status.detail is not None

    async def test_close_releases_the_client_once(
        self, adapter: WeaviateAdapter, client: FakeClient
    ) -> None:
        await adapter.close()
        await adapter.close()

        assert client.closed == 1
        assert adapter._client is None


class TestDriverAbsence:
    """A missing driver names the extra that installs it."""

    def test_a_missing_driver_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fasterrag.adapters.vectordb.weaviate._DRIVER_IMPORT_ERROR",
            ImportError("no module named weaviate"),
        )

        with pytest.raises(FasterRagError) as caught:
            WeaviateAdapter(settings())

        assert "fasterrag[weaviate]" in caught.value.detail


def test_the_unexpected_status_error_is_importable() -> None:
    """Pins the vendor exception the translator's named cases are chosen against."""
    assert issubclass(UnexpectedStatusCodeError, WeaviateBaseError)


def test_a_filter_the_adapter_cannot_build_fails_the_fake_loudly() -> None:
    """The fake's evaluator must never silently pass a filter it does not understand."""
    with pytest.raises(AssertionError):
        matches(WeaviateFilter.by_property("x").like("a*"), {"x": "abc"}, "k")
