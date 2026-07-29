import pytest

from fasterrag.adapters.vectordb.base import (
    PointSelector,
    validate_filter,
)
from fasterrag.errors import ErrorCode, FasterRagError


def test_no_filter_is_valid() -> None:
    validate_filter(None)
    validate_filter({})


def test_scalar_equality_is_valid() -> None:
    validate_filter({"department": "legal", "year": 2024})


@pytest.mark.parametrize("operator", ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"])
def test_supported_scalar_operators(operator: str) -> None:
    validate_filter({"year": {operator: 2024}})


@pytest.mark.parametrize("operator", ["$in", "$nin"])
def test_supported_set_operators(operator: str) -> None:
    validate_filter({"tag": {operator: ["a", "b"]}})


def test_unsupported_operator_is_rejected() -> None:
    with pytest.raises(FasterRagError, match="unsupported operators") as caught:
        validate_filter({"year": {"$regex": ".*"}})
    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert caught.value.status == 422


def test_multiple_operators_on_one_field_are_rejected() -> None:
    with pytest.raises(FasterRagError, match="exactly one operator"):
        validate_filter({"year": {"$gte": 2024, "$lte": 2025}})


def test_set_operator_requires_a_list() -> None:
    with pytest.raises(FasterRagError, match="requires a list"):
        validate_filter({"tag": {"$in": "a"}})


def test_selector_accepts_ids_only() -> None:
    selector = PointSelector(collection="default", point_ids=["c_1"])
    assert selector.point_ids == ["c_1"]


def test_selector_accepts_filters_only() -> None:
    selector = PointSelector(collection="default", filters={"tenant": "acme"})
    assert selector.filters == {"tenant": "acme"}


def test_selector_rejects_both_ids_and_filters() -> None:
    with pytest.raises(FasterRagError, match="exactly one"):
        PointSelector(collection="default", point_ids=["c_1"], filters={"tenant": "acme"})


def test_selector_rejects_neither() -> None:
    with pytest.raises(FasterRagError, match="exactly one"):
        PointSelector(collection="default")
