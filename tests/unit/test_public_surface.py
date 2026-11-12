"""A snapshot of the surface a user or a plugin author builds against (TASK-0486).

``docs/python-api.md`` promises SemVer stability for the names ``fasterrag`` exports and for
the adapter base classes. Until this snapshot, membership tests checked that some names were
present, so a name could be removed or renamed with the suite green. Removing a name below,
or changing what a plugin must implement without moving ``ADAPTER_API_VERSION``, now fails.

Adding a name is not a break and does not fail here: extend the snapshot when you add one, so
its removal is caught later. Changing an abstract-member set is always a break for plugins:
bump the base module's ``ADAPTER_API_VERSION`` major (or minor, for an addition with a
default) and record the new set under the new version.
"""

from __future__ import annotations

from importlib import import_module

import pytest

import fasterrag

EXPORTS: dict[str, frozenset[str]] = {
    "fasterrag": frozenset({"FasterRag", "__version__"}),
    "fasterrag.adapters.vectordb.base": frozenset(
        {
            "ADAPTER_API_VERSION",
            "COMPARISON_OPERATORS",
            "CollectionInfo",
            "CollectionSpec",
            "Distance",
            "Filter",
            "HealthStatus",
            "POINT_ID_PAYLOAD_KEY",
            "Point",
            "PointSelector",
            "PointUpdate",
            "ScoredPoint",
            "SearchQuery",
            "SparseVector",
            "UpsertResult",
            "VectorDBAdapter",
            "validate_filter",
        }
    ),
    "fasterrag.adapters.embeddings.base": frozenset(
        {
            "ADAPTER_API_VERSION",
            "EmbeddingAdapter",
            "EmbeddingResult",
            "classify_provider_failure",
            "require_extra",
            "require_key",
            "sum_reported",
        }
    ),
    "fasterrag.adapters.llm.base": frozenset(
        {
            "ADAPTER_API_VERSION",
            "Completion",
            "LLMAdapter",
            "classify_llm_failure",
            "require_llm_extra",
            "require_llm_key",
        }
    ),
}

CONTRACTS: dict[tuple[str, str], tuple[str, frozenset[str]]] = {
    ("fasterrag.adapters.vectordb.base", "VectorDBAdapter"): (
        "1.0",
        frozenset(
            {
                "alias_target",
                "close",
                "create_collection",
                "delete",
                "delete_alias",
                "delete_snapshot",
                "drop_collection",
                "health",
                "iterate_points",
                "list_collections",
                "list_snapshots",
                "restore_snapshot",
                "search",
                "set_alias",
                "snapshot",
                "update",
                "upsert",
            }
        ),
    ),
    ("fasterrag.adapters.embeddings.base", "EmbeddingAdapter"): (
        "1.0",
        frozenset(
            {
                "close",
                "dimensions",
                "embed_documents",
                "embed_query",
                "health",
                "model",
                "model_version",
            }
        ),
    ),
    ("fasterrag.adapters.llm.base", "LLMAdapter"): (
        "1.0",
        frozenset({"close", "complete", "health", "stream"}),
    ),
}

ENTRY_POINT_GROUPS: dict[str, str] = {
    "fasterrag.adapters.vectordb.factory": "fasterrag.vectordb",
    "fasterrag.adapters.embeddings.factory": "fasterrag.embeddings",
    "fasterrag.adapters.llm.factory": "fasterrag.llm",
}


@pytest.mark.parametrize("module_name", sorted(EXPORTS))
def test_no_documented_name_has_been_removed(module_name: str) -> None:
    """The defect: a stable name could disappear and only a downstream user would notice."""
    module = import_module(module_name)
    exported = frozenset(module.__all__)

    assert EXPORTS[module_name] - exported == frozenset()
    for name in EXPORTS[module_name]:
        assert hasattr(module, name) or name in dir(module)


def test_every_lazy_top_level_export_resolves() -> None:
    """``FasterRag`` is imported on first access, so membership alone would not prove it."""
    for name in fasterrag.__all__:
        assert getattr(fasterrag, name) is not None


@pytest.mark.parametrize(("location", "snapshot"), sorted(CONTRACTS.items()))
def test_what_a_plugin_must_implement_matches_the_version_it_is_told_to_target(
    location: tuple[str, str], snapshot: tuple[str, frozenset[str]]
) -> None:
    """A changed abstract-member set under an unchanged version breaks plugins silently."""
    module_name, class_name = location
    module = import_module(module_name)
    version, members = snapshot

    assert version == module.ADAPTER_API_VERSION
    assert frozenset(getattr(module, class_name).__abstractmethods__) == members


@pytest.mark.parametrize(("module_name", "group"), sorted(ENTRY_POINT_GROUPS.items()))
def test_the_entry_point_group_names_are_unchanged(module_name: str, group: str) -> None:
    """Renaming a group orphans every installed plugin without an error anywhere."""
    assert group == import_module(module_name).ENTRY_POINT_GROUP
