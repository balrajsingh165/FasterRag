import pytest

from fasterrag.adapters.embeddings import huggingface
from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder
from fasterrag.adapters.embeddings.tiering import create_embedding_router, matches
from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError
from tests.unit.adapters.embeddings.conftest import FakeModel


def tiered_settings(rules: list[dict[str, object]]) -> Settings:
    return Settings.model_validate(
        {
            "embeddings": {
                "provider": "huggingface",
                "model": "default-model",
                "tiering": {"enabled": True, "rules": rules},
            }
        }
    )


@pytest.fixture(autouse=True)
def local_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    fake = FakeModel()
    monkeypatch.setattr(huggingface, "load_model", lambda name: fake)
    return fake


def test_equality_matching() -> None:
    assert matches({"priority_class": "archive"}, {"priority_class": "archive"})
    assert not matches({"priority_class": "archive"}, {"priority_class": "legal"})


def test_a_missing_key_never_matches() -> None:
    assert not matches({"priority_class": "archive"}, {"department": "legal"})


@pytest.mark.parametrize(
    ("condition", "value", "expected"),
    [
        ({"$gte": 2024}, 2024, True),
        ({"$gte": 2024}, 2023, False),
        ({"$lt": 2024}, 2023, True),
        ({"$in": ["a", "b"]}, "b", True),
        ({"$in": ["a", "b"]}, "c", False),
        ({"$nin": ["a"]}, "b", True),
        ({"$ne": "a"}, "b", True),
        ({"$ne": "a"}, "a", False),
    ],
)
def test_operator_matching(condition: dict[str, object], value: object, expected: bool) -> None:
    assert matches({"year": condition}, {"year": value}) is expected


def test_a_range_operator_on_a_non_number_does_not_match() -> None:
    assert not matches({"year": {"$gte": 2024}}, {"year": "recent"})


def test_all_conditions_must_hold() -> None:
    filters = {"department": "legal", "year": {"$gte": 2024}}

    assert matches(filters, {"department": "legal", "year": 2025})
    assert not matches(filters, {"department": "legal", "year": 2020})


def test_a_disabled_router_always_returns_the_default() -> None:
    router = create_embedding_router(Settings.model_validate({"embeddings": {"model": "only"}}))

    assert router.enabled is False
    assert router.select({"priority_class": "archive"}).model == "only"


def test_the_first_matching_rule_wins() -> None:
    router = create_embedding_router(
        tiered_settings(
            [
                {
                    "match": {"priority_class": "archive"},
                    "provider": "huggingface",
                    "model": "cheap-model",
                },
                {
                    "match": {"priority_class": "archive"},
                    "provider": "huggingface",
                    "model": "never-reached",
                },
            ]
        )
    )

    assert router.select({"priority_class": "archive"}).model == "cheap-model"


def test_unmatched_metadata_falls_through_to_the_default() -> None:
    router = create_embedding_router(
        tiered_settings(
            [
                {
                    "match": {"priority_class": "archive"},
                    "provider": "huggingface",
                    "model": "cheap-model",
                }
            ]
        )
    )

    assert router.select({"priority_class": "legal"}).model == "default-model"
    assert router.select(None).model == "default-model"
    assert router.select({}).model == "default-model"


def test_each_rule_gets_its_own_adapter() -> None:
    router = create_embedding_router(
        tiered_settings(
            [
                {"match": {"tier": "a"}, "provider": "huggingface", "model": "model-a"},
                {"match": {"tier": "b"}, "provider": "huggingface", "model": "model-b"},
            ]
        )
    )

    assert {adapter.model for adapter in router.adapters()} == {
        "default-model",
        "model-a",
        "model-b",
    }


def test_a_rule_with_an_unsupported_operator_is_rejected() -> None:
    settings = tiered_settings(
        [{"match": {"year": {"$regex": ".*"}}, "provider": "huggingface", "model": "m"}]
    )

    with pytest.raises(FasterRagError, match="unsupported operators"):
        create_embedding_router(settings)


def test_a_rule_may_route_to_a_different_provider() -> None:
    settings = Settings.model_validate(
        {
            "embeddings": {
                "provider": "huggingface",
                "tiering": {
                    "enabled": True,
                    "rules": [{"match": {"tier": "a"}, "provider": "openai", "model": "m"}],
                },
            }
        }
    )
    router = create_embedding_router(settings)
    routed = router.select({"tier": "a"})

    assert routed.model == "m"
    assert routed.provider == "openai"


async def test_closing_the_router_closes_every_adapter() -> None:
    router = create_embedding_router(
        tiered_settings([{"match": {"tier": "a"}, "provider": "huggingface", "model": "model-a"}])
    )

    local = [adapter for adapter in router.adapters() if isinstance(adapter, HuggingFaceEmbedder)]
    for adapter in local:
        await adapter.embed_documents(["warm up"])

    await router.close()

    assert len(local) == 2
    assert all(adapter._model is None for adapter in local)


def test_the_router_reports_the_configured_rule_count() -> None:
    router = create_embedding_router(
        tiered_settings([{"match": {"tier": "a"}, "provider": "huggingface", "model": "model-a"}])
    )

    assert router.enabled is True
    assert len(router.tiers) == 1
