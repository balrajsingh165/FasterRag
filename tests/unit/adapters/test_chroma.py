"""Chroma adapter unit tests.

Everything here runs without opening a Chroma directory. The integration suite in
``tests/integration/chroma`` owns whether the adapter is *correct* against a real store;
these own the logic around it — the collection-name derivation Chroma's stricter naming
rules force, the exception-to-taxonomy mapping, filter translation, the lossless payload
flattening that scalar-only metadata forces, and the BM25 ranking the adapter has to do
itself because Chroma has no corpus-rarity primitive.
"""

import re
from typing import Any

import pytest
from chromadb import errors as chroma_errors

from fasterrag.adapters.vectordb.base import Point, SparseVector
from fasterrag.adapters.vectordb.chroma import (
    ChromaAdapter,
    _Collection,
    _from_metadata,
    _rank,
    _score_of,
    _to_metadata,
    collection_name_for,
    snapshot_name_for,
    to_chroma_where,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError

CHROMA_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


def settings(**chroma: Any) -> Settings:
    """Return settings selecting Chroma; the directory is named but never opened."""
    return Settings.model_validate(
        {"vector_db": {"provider": "chroma", "chroma": {"path": ".fasterrag/unit", **chroma}}}
    )


@pytest.fixture
def adapter() -> ChromaAdapter:
    """An adapter that never opens its persistent directory."""
    return ChromaAdapter(settings())


def described(**overrides: Any) -> _Collection:
    """Return a resolved collection descriptor for the guards that take one."""
    fields: dict[str, Any] = {
        "name": "notes",
        "physical": collection_name_for("notes"),
        "dimensions": 4,
        "distance": "cosine",
        "sparse": False,
    }
    fields.update(overrides)
    return _Collection(**fields)


class TestDerivedNames:
    """Chroma names are stricter than fasterRag ones, so the mapping has to absorb it."""

    @pytest.mark.parametrize(
        "collection", ["a", "ab", "my_collection", "UPPER", "-leading", "trailing-", "x" * 64]
    )
    def test_every_legal_fasterrag_name_derives_a_legal_chroma_name(self, collection: str) -> None:
        derived = collection_name_for(collection)

        assert CHROMA_NAME.match(derived), derived
        assert len(derived) >= 3

    def test_the_derivation_is_stable(self) -> None:
        assert collection_name_for("notes") == collection_name_for("notes")

    def test_names_that_slug_alike_still_separate(self) -> None:
        assert collection_name_for("my_notes") != collection_name_for("my-notes")

    def test_a_snapshot_name_is_distinct_from_its_collection(self) -> None:
        assert snapshot_name_for("notes", "notes-1") != collection_name_for("notes")

    def test_snapshots_of_different_collections_do_not_collide(self) -> None:
        assert snapshot_name_for("a", "snap") != snapshot_name_for("b", "snap")


class TestErrorTranslation:
    """Every Chroma failure must leave this package as a typed fasterRag error."""

    @pytest.mark.parametrize(
        ("raised", "expected"),
        [
            (chroma_errors.NotFoundError("gone"), ErrorCode.NOT_FOUND),
            (chroma_errors.UniqueConstraintError("twice"), ErrorCode.CONFLICT),
            (chroma_errors.VersionMismatchError("stale"), ErrorCode.CONFLICT),
            (chroma_errors.InvalidArgumentError("no"), ErrorCode.VALIDATION_FAILED),
            (ValueError("client-side validation"), ErrorCode.VALIDATION_FAILED),
            (chroma_errors.AuthorizationError("nope"), ErrorCode.VECTOR_DB_AUTH_FAILED),
            (chroma_errors.RateLimitError("slow down"), ErrorCode.EMBED_PROVIDER_ERROR),
            (OSError("disk"), ErrorCode.EMBED_PROVIDER_ERROR),
        ],
    )
    def test_each_failure_maps_to_its_code(
        self, adapter: ChromaAdapter, raised: BaseException, expected: ErrorCode
    ) -> None:
        translated = adapter._translate(raised, "search")

        assert translated.code is expected

    def test_a_rejected_request_is_not_retryable(self, adapter: ChromaAdapter) -> None:
        translated = adapter._translate(chroma_errors.InvalidArgumentError("no"), "upsert")

        assert translated.retryable is False

    def test_an_unreachable_store_is_retryable(self, adapter: ChromaAdapter) -> None:
        translated = adapter._translate(OSError("disk"), "upsert")

        assert translated.retryable is True

    def test_the_operation_is_named_in_the_detail(self, adapter: ChromaAdapter) -> None:
        translated = adapter._translate(chroma_errors.NotFoundError("gone"), "restore_snapshot")

        assert "restore_snapshot" in translated.detail

    async def test_a_vendor_exception_never_escapes_the_adapter(
        self, adapter: ChromaAdapter
    ) -> None:
        def explode() -> None:
            raise chroma_errors.InternalError("boom")

        with pytest.raises(FasterRagError):
            await adapter._call("search", explode)


class TestFilterTranslation:
    """The neutral grammar maps one-to-one onto Chroma's, which is the whole point."""

    def test_no_filter_translates_to_no_where(self) -> None:
        assert to_chroma_where(None) is None
        assert to_chroma_where({}) is None

    def test_a_scalar_becomes_an_explicit_equality(self) -> None:
        assert to_chroma_where({"department": "legal"}) == {"department": {"$eq": "legal"}}

    def test_an_operator_is_passed_through(self) -> None:
        assert to_chroma_where({"year": {"$gte": 2024}}) == {"year": {"$gte": 2024}}

    def test_several_keys_become_an_explicit_conjunction(self) -> None:
        translated = to_chroma_where({"department": "legal", "year": {"$gte": 2024}})

        assert translated == {"$and": [{"department": {"$eq": "legal"}}, {"year": {"$gte": 2024}}]}

    def test_an_unsupported_operator_is_refused(self) -> None:
        with pytest.raises(FasterRagError) as caught:
            to_chroma_where({"text": {"$regex": "^a"}})

        assert caught.value.code is ErrorCode.VALIDATION_FAILED


class TestPayloadFlattening:
    """Chroma metadata holds scalars only, so the encoding has to be lossless."""

    def test_a_scalar_payload_round_trips(self) -> None:
        payload = {"department": "legal", "year": 2024, "reviewed": True, "score": 0.5}

        assert _from_metadata(_to_metadata(payload, None)) == payload

    def test_a_nested_payload_round_trips(self) -> None:
        payload: dict[str, Any] = {"span": {"start": 0, "end": 9}, "tags": ["a", "b"]}

        assert _from_metadata(_to_metadata(payload, None)) == payload

    def test_a_null_value_round_trips(self) -> None:
        assert _from_metadata(_to_metadata({"section": None}, None)) == {"section": None}

    def test_a_payload_key_colliding_with_the_reserved_namespace_round_trips(self) -> None:
        payload = {"fr.t.1": "not a posting"}

        assert _from_metadata(_to_metadata(payload, None)) == payload

    def test_scalars_stay_filterable_as_their_own_keys(self) -> None:
        stored = _to_metadata({"department": "legal", "span": {"start": 0}}, None)

        assert stored["department"] == "legal"

    def test_postings_never_reach_a_caller_payload(self) -> None:
        stored = _to_metadata({"department": "legal"}, SparseVector(indices=[7], values=[1.5]))

        assert stored["fr.t.7"] == 1.5
        assert _from_metadata(stored) == {"department": "legal"}

    def test_corrupt_stored_json_is_a_typed_error(self) -> None:
        with pytest.raises(FasterRagError) as caught:
            _from_metadata({"fr.j.span": "{not json"})

        assert caught.value.code is ErrorCode.INTERNAL


class TestScoring:
    """``score`` must rank higher-is-better whatever Chroma's distance means."""

    def test_cosine_inverts_the_distance(self) -> None:
        assert _score_of("cosine", 0.0) > _score_of("cosine", 1.0)

    def test_dot_inverts_the_distance(self) -> None:
        assert _score_of("dot", 0.0) > _score_of("dot", 1.0)

    def test_euclid_negates_the_distance(self) -> None:
        assert _score_of("euclid", 0.5) > _score_of("euclid", 2.0)
        assert _score_of("euclid", 2.0) == -2.0


class TestKeywordRanking:
    """Chroma has no corpus-rarity primitive, so the adapter owns IDF — and must earn it."""

    def test_a_rare_term_outweighs_a_common_one(self) -> None:
        query = SparseVector(indices=[1, 2], values=[1.0, 1.0])
        candidates = ["rare", "common_a", "common_b"]
        metadatas: list[dict[str, Any]] = [
            {"fr.t.1": 1.0},
            {"fr.t.2": 1.0},
            {"fr.t.2": 1.0},
        ]

        ranked = _rank(query, candidates, metadatas, total=3, allowed=None)

        assert ranked[0][0] == "rare"

    def test_a_document_matching_nothing_scores_zero(self) -> None:
        query = SparseVector(indices=[1], values=[1.0])
        ranked = _rank(query, ["a", "b"], [{"fr.t.1": 1.0}, {}], total=2, allowed=None)

        assert dict(ranked)["b"] == 0.0

    def test_scores_come_back_in_descending_order(self) -> None:
        query = SparseVector(indices=[1], values=[1.0])
        candidates = ["low", "high"]
        metadatas: list[dict[str, Any]] = [{"fr.t.1": 0.2}, {"fr.t.1": 0.9}]

        ranked = _rank(query, candidates, metadatas, total=2, allowed=None)

        assert [point_id for point_id, _ in ranked] == ["high", "low"]

    def test_a_pushed_down_filter_removes_a_candidate_without_changing_idf(self) -> None:
        query = SparseVector(indices=[1], values=[1.0])
        candidates = ["a", "b"]
        metadatas: list[dict[str, Any]] = [{"fr.t.1": 1.0}, {"fr.t.1": 1.0}]

        unfiltered = _rank(query, candidates, metadatas, total=2, allowed=None)
        filtered = _rank(query, candidates, metadatas, total=2, allowed={"b"})

        assert [point_id for point_id, _ in filtered] == ["b"]
        assert dict(filtered)["b"] == dict(unfiltered)["b"]


class TestWriteGuards:
    """The guards that refuse before anything reaches the store."""

    def test_a_mismatched_dimension_is_refused_with_the_expected_size(self) -> None:
        batch = [Point(point_id="c_a", collection="notes", vector=[1.0, 0.0])]

        with pytest.raises(EmbedError) as caught:
            ChromaAdapter._require_matching_dimensions(described(), batch)

        assert "4" in caught.value.detail
        assert caught.value.retryable is False

    def test_a_posting_without_a_sparse_index_is_refused(self) -> None:
        batch = [
            Point(
                point_id="c_a",
                collection="notes",
                vector=[1.0, 0.0, 0.0, 0.0],
                sparse=SparseVector(indices=[1], values=[1.0]),
            )
        ]

        with pytest.raises(EmbedError):
            ChromaAdapter._require_sparse_index(described(), batch)

    def test_a_posting_is_accepted_by_a_sparse_collection(self) -> None:
        batch = [
            Point(
                point_id="c_a",
                collection="notes",
                vector=[1.0, 0.0, 0.0, 0.0],
                sparse=SparseVector(indices=[1], values=[1.0]),
            )
        ]

        ChromaAdapter._require_sparse_index(described(sparse=True), batch)
