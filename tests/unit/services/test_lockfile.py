from pathlib import Path

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.lockfile import (
    LOCK_VERSION,
    IndexLock,
    LockStore,
    build_lock,
    create_lock_store,
    detect_drift,
)


def lock(settings: Settings | None = None, **overrides: object) -> IndexLock:
    built = build_lock(
        "docs",
        settings or Settings(),
        embedding_model="bge-small",
        embedding_model_version="bge-small-v1",
        dimensions=384,
        document_hashes={"d_1": "hash-a", "d_2": "hash-b"},
    )
    return built if not overrides else IndexLock(**{**built.as_dict(), **overrides})


def test_a_lock_records_what_produced_the_index() -> None:
    built = lock()

    assert built.collection == "docs"
    assert built.embedding_model == "bge-small"
    assert built.embedding_model_version == "bge-small-v1"
    assert built.dimensions == 384
    assert built.lock_version == LOCK_VERSION
    assert built.built_by


def test_a_lock_records_the_chunker_parameters() -> None:
    built = lock()

    assert built.chunker_strategy == "recursive"
    assert built.chunk_size == 768
    assert built.overlap == 64


def test_a_lock_round_trips_through_its_persisted_form() -> None:
    restored = IndexLock.from_dict(lock().as_dict())

    assert restored.config_hash == lock().config_hash
    assert restored.document_hashes == {"d_1": "hash-a", "d_2": "hash-b"}


def test_the_summary_omits_per_document_hashes() -> None:
    summary = lock().summary()

    assert "document_hashes" not in summary
    assert summary["documents"] == 2


def test_an_unchanged_config_reports_no_drift() -> None:
    settings = Settings()

    report = detect_drift(lock(settings), settings, collection="docs")

    assert report.detected is False
    assert report.fields == []


def test_a_changed_embedding_model_is_named() -> None:
    settings = Settings()

    report = detect_drift(
        lock(settings), settings, collection="docs", embedding_model="text-embedding-3-small"
    )

    assert report.detected is True
    assert "embedding_model" in report.fields


def test_a_changed_model_version_is_named() -> None:
    settings = Settings()

    report = detect_drift(
        lock(settings), settings, collection="docs", embedding_model_version="bge-small-v2"
    )

    assert "embedding_model_version" in report.fields


def test_a_changed_chunk_size_is_named() -> None:
    original = Settings()
    changed = Settings.model_validate({"chunking": {"chunk_size": 512}})

    report = detect_drift(lock(original), changed, collection="docs")

    assert "chunk_size" in report.fields
    assert "config_hash" in report.fields


def test_the_drift_detail_carries_both_values() -> None:
    original = Settings()
    changed = Settings.model_validate({"chunking": {"chunk_size": 512}})

    report = detect_drift(lock(original), changed, collection="docs")
    detail = next(item for item in report.details if item["field"] == "chunk_size")

    assert detail["locked"] == 768
    assert detail["live"] == 512


def test_an_unrelated_config_change_is_not_drift() -> None:
    original = Settings()
    changed = Settings.model_validate({"app": {"port": 9999}})

    report = detect_drift(lock(original), changed, collection="docs")

    assert report.detected is False


def test_a_new_document_is_reported_as_added() -> None:
    settings = Settings()

    report = detect_drift(
        lock(settings),
        settings,
        collection="docs",
        document_hashes={"d_1": "hash-a", "d_2": "hash-b", "d_3": "hash-c"},
    )

    assert report.documents_added == ["d_3"]
    assert report.detected is True


def test_a_deleted_document_is_reported_as_removed() -> None:
    settings = Settings()

    report = detect_drift(
        lock(settings), settings, collection="docs", document_hashes={"d_1": "hash-a"}
    )

    assert report.documents_removed == ["d_2"]


def test_edited_content_is_reported_as_changed() -> None:
    settings = Settings()

    report = detect_drift(
        lock(settings),
        settings,
        collection="docs",
        document_hashes={"d_1": "hash-a", "d_2": "DIFFERENT"},
    )

    assert report.documents_changed == ["d_2"]
    assert report.documents_added == []


def test_a_missing_lock_is_reported_as_missing_not_as_drift() -> None:
    report = detect_drift(None, Settings(), collection="docs")

    assert report.missing_lock is True
    assert report.detected is False


def test_the_report_serializes_for_the_status_surfaces() -> None:
    payload = detect_drift(None, Settings(), collection="docs").as_dict()

    assert set(payload) == {
        "collection",
        "detected",
        "missing_lock",
        "fields",
        "details",
        "documents",
    }


def test_a_written_lock_reads_back(tmp_path: Path) -> None:
    store = LockStore(tmp_path)
    store.write(lock())

    restored = store.read("docs")

    assert restored is not None
    assert restored.embedding_model == "bge-small"


def test_a_disabled_store_writes_nothing(tmp_path: Path) -> None:
    store = LockStore(tmp_path, enabled=False)
    store.write(lock())

    assert store.read("docs") is None
    assert list(tmp_path.glob("*.json")) == []


def test_an_absent_lock_reads_as_none(tmp_path: Path) -> None:
    assert LockStore(tmp_path).read("never") is None


def test_a_corrupt_lock_reads_as_none_rather_than_raising(tmp_path: Path) -> None:
    store = LockStore(tmp_path)
    store.write(lock())
    (tmp_path / "docs.lock.json").write_text("{not json", encoding="utf-8")

    assert store.read("docs") is None


def test_an_unwritable_root_never_fails_the_build(tmp_path: Path) -> None:
    blocker = tmp_path / "locks"
    blocker.write_text("not a directory", encoding="utf-8")

    LockStore(blocker).write(lock())


def test_rewriting_replaces_the_previous_lock(tmp_path: Path) -> None:
    store = LockStore(tmp_path)
    store.write(lock())
    store.write(
        build_lock(
            "docs",
            Settings(),
            embedding_model="new-model",
            embedding_model_version="v2",
            dimensions=1536,
        )
    )

    restored = store.read("docs")

    assert restored is not None
    assert restored.embedding_model == "new-model"
    assert restored.document_hashes == {}


def test_deleting_reports_whether_one_was_there(tmp_path: Path) -> None:
    store = LockStore(tmp_path)
    store.write(lock())

    assert store.delete("docs") is True
    assert store.delete("docs") is False


def test_the_store_honours_the_lockfile_setting() -> None:
    settings = Settings.model_validate({"index": {"lockfile": False}})

    assert create_lock_store(settings).enabled is False


@pytest.mark.parametrize("field_name", ["contextual_enrichment", "overlap"])
def test_every_lock_field_that_changes_chunks_is_compared(field_name: str) -> None:
    changed = Settings.model_validate(
        {"chunking": {field_name: True if field_name == "contextual_enrichment" else 128}}
    )

    report = detect_drift(lock(Settings()), changed, collection="docs")

    assert field_name in report.fields
