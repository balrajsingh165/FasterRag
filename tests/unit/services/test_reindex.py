from collections.abc import AsyncIterator
from typing import Any

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.reindex import (
    GREEN_SUFFIX,
    green_name,
    plan_reindex,
    retire,
    rollback,
    swap,
)


class AliasAdapter(VectorDBAdapter):
    """An in-memory backend with collections and aliases."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.collections: dict[str, int] = {}
        self.aliases: dict[str, str] = {}
        self.snapshots: dict[str, list[str]] = {}
        self.restored: list[tuple[str, str]] = []
        self.swaps: list[tuple[str, str]] = []

    async def create_collection(self, spec: CollectionSpec) -> None:
        self.collections.setdefault(spec.name, 0)

    async def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(name=name, vectors=count) for name, count in self.collections.items()
        ]

    async def drop_collection(self, name: str) -> bool:
        return self.collections.pop(name, None) is not None

    async def snapshot(self, collection: str) -> str:
        self.snapshots.setdefault(collection, []).append(f"{collection}-snap")
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        return list(self.snapshots.get(collection, []))

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        self.restored.append((collection, snapshot))

    async def set_alias(self, alias: str, collection: str) -> None:
        self.aliases[alias] = collection
        self.swaps.append((alias, collection))

    async def alias_target(self, alias: str) -> str | None:
        return self.aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        return self.aliases.pop(alias, None) is not None

    async def upsert(self, points: list[Point]) -> UpsertResult:
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield nothing; these doubles hold no scannable state."""
        empty: list[Point] = []
        for point in empty:
            yield point

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        return []

    async def update(self, updates: list[PointUpdate]) -> None:
        return None

    async def delete(self, selector: PointSelector) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def settings(**overrides: Any) -> Settings:
    return Settings.model_validate(overrides)


def adapter(**collections: int) -> AliasAdapter:
    built = AliasAdapter(settings())
    built.collections.update(collections)
    return built


def test_a_green_name_is_derived_from_the_alias() -> None:
    assert green_name("docs", "20260801120000") == f"docs{GREEN_SUFFIX}20260801120000"


def test_successive_builds_get_distinguishable_names() -> None:
    assert green_name("docs", "20260801120000") != green_name("docs", "20260801120001")


async def test_a_first_build_has_no_blue() -> None:
    plan = await plan_reindex("docs", settings(), adapter())

    assert plan.first_build is True
    assert plan.blue is None
    assert plan.green.startswith(f"docs{GREEN_SUFFIX}")


async def test_a_rebuild_records_the_live_collection_as_blue() -> None:
    backend = adapter(**{"docs-v1": 10})
    backend.aliases["docs"] = "docs-v1"

    plan = await plan_reindex("docs", settings(), backend)

    assert plan.blue == "docs-v1"
    assert plan.first_build is False


async def test_a_physical_collection_on_the_served_name_is_refused() -> None:
    backend = adapter(docs=5)

    with pytest.raises(FasterRagError) as failure:
        await plan_reindex("docs", settings(), backend)

    assert failure.value.code is ErrorCode.CONFLICT
    assert "alias" in failure.value.detail


async def test_the_plan_carries_the_configured_policy() -> None:
    configured = settings(index={"reindex": {"eval_gate": False, "rollback_retention_hours": 12}})

    plan = await plan_reindex("docs", configured, adapter())

    assert plan.eval_gate is False
    assert plan.rollback_retention_hours == 12
    assert plan.strategy == "blue_green"


async def test_a_passing_gate_swaps_the_alias() -> None:
    backend = adapter()
    plan = await plan_reindex("docs", settings(), backend)

    result = await swap(plan, backend, eval_passed=True)

    assert result.swapped is True
    assert backend.aliases["docs"] == plan.green


async def test_a_failing_gate_leaves_the_previous_index_live() -> None:
    backend = adapter(**{"docs-v1": 10})
    backend.aliases["docs"] = "docs-v1"
    plan = await plan_reindex("docs", settings(), backend)

    result = await swap(plan, backend, eval_passed=False)

    assert result.swapped is False
    assert backend.aliases["docs"] == "docs-v1"
    assert "eval gate" in result.reason


async def test_a_blocked_swap_is_a_result_not_an_exception() -> None:
    backend = adapter()
    plan = await plan_reindex("docs", settings(), backend)

    result = await swap(plan, backend, eval_passed=False)

    assert result.swapped is False
    assert result.as_dict()["swapped"] is False


async def test_a_disabled_gate_swaps_even_on_a_failing_eval() -> None:
    configured = settings(index={"reindex": {"eval_gate": False}})
    backend = adapter()
    plan = await plan_reindex("docs", configured, backend)

    result = await swap(plan, backend, eval_passed=False)

    assert result.swapped is True


async def test_the_swap_is_one_operation() -> None:
    backend = adapter(**{"docs-v1": 1})
    backend.aliases["docs"] = "docs-v1"
    plan = await plan_reindex("docs", settings(), backend)

    await swap(plan, backend)

    assert len(backend.swaps) == 1


async def test_the_eval_report_is_carried_into_the_result() -> None:
    backend = adapter()
    plan = await plan_reindex("docs", settings(), backend)

    result = await swap(plan, backend, eval_passed=True, eval_report={"recall_at_10": 0.91})

    assert result.eval_report == {"recall_at_10": 0.91}


async def test_rollback_restores_the_previous_build() -> None:
    backend = adapter(**{"docs-v20260101000000": 1, "docs-v20260102000000": 1})
    backend.aliases["docs"] = "docs-v20260102000000"

    result = await rollback("docs", backend)

    assert result.restored == "docs-v20260101000000"
    assert result.replaced == "docs-v20260102000000"
    assert backend.aliases["docs"] == "docs-v20260101000000"


async def test_rollback_can_target_a_named_build() -> None:
    backend = adapter(**{"docs-v20260101000000": 1, "docs-v20260103000000": 1})
    backend.aliases["docs"] = "docs-v20260103000000"

    result = await rollback("docs", backend, to="docs-v20260101000000")

    assert result.restored == "docs-v20260101000000"


async def test_rollback_with_nothing_retained_is_refused() -> None:
    backend = adapter(**{"docs-v20260101000000": 1})
    backend.aliases["docs"] = "docs-v20260101000000"

    with pytest.raises(FasterRagError) as failure:
        await rollback("docs", backend)

    assert failure.value.code is ErrorCode.NOT_FOUND


async def test_rollback_never_targets_the_live_collection() -> None:
    backend = adapter(**{"docs-v20260101000000": 1, "docs-v20260102000000": 1})
    backend.aliases["docs"] = "docs-v20260101000000"

    result = await rollback("docs", backend)

    assert result.restored == "docs-v20260102000000"


async def test_retiring_drops_builds_past_the_window() -> None:
    import time

    backend = adapter(**{"docs-v20200101000000": 1, "docs-v20260801000000": 1})
    backend.aliases["docs"] = "docs-v20260801000000"

    dropped = await retire("docs", backend, settings(), now=time.time())

    assert dropped == ["docs-v20200101000000"]
    assert "docs-v20200101000000" not in backend.collections


async def test_retiring_never_drops_the_live_collection() -> None:
    import time

    backend = adapter(**{"docs-v20200101000000": 1})
    backend.aliases["docs"] = "docs-v20200101000000"

    dropped = await retire("docs", backend, settings(), now=time.time())

    assert dropped == []
    assert "docs-v20200101000000" in backend.collections


async def test_retiring_ignores_collections_it_did_not_create() -> None:
    import time

    backend = adapter(**{"docs-vNOTASTAMP": 1, "unrelated": 1})
    backend.aliases["docs"] = "docs-v20260801000000"

    dropped = await retire("docs", backend, settings(), now=time.time())

    assert dropped == []
    assert "unrelated" in backend.collections


async def test_a_zero_retention_window_still_spares_the_live_collection() -> None:
    import time

    configured = settings(index={"reindex": {"rollback_retention_hours": 0}})
    backend = adapter(**{"docs-v20200101000000": 1, "docs-v20260801000000": 1})
    backend.aliases["docs"] = "docs-v20260801000000"

    dropped = await retire("docs", backend, configured, now=time.time())

    assert dropped == ["docs-v20200101000000"]
    assert "docs-v20260801000000" in backend.collections
