from fasterrag.config.schema import Settings
from fasterrag.core.identity import (
    chunk_id,
    chunker_config_hash,
    collection_id,
    content_hash,
    document_id,
    job_id,
    retrieval_config_hash,
)


def test_document_ids_are_deterministic() -> None:
    assert document_id("s3://docs/a.pdf") == document_id("s3://docs/a.pdf")


def test_document_ids_distinguish_sources_and_tenants() -> None:
    assert document_id("a.pdf") != document_id("b.pdf")
    assert document_id("a.pdf", "acme") != document_id("a.pdf", "globex")
    assert document_id("a.pdf") != document_id("a.pdf", "acme")


def test_document_ids_carry_their_prefix() -> None:
    assert document_id("a.pdf").startswith("d_")


def test_chunk_ids_are_a_pure_function_of_their_inputs() -> None:
    first = chunk_id("d_1", 3, "hash")
    second = chunk_id("d_1", 3, "hash")

    assert first == second
    assert first.startswith("c_")


def test_chunk_ids_change_with_index_document_or_chunker() -> None:
    baseline = chunk_id("d_1", 3, "hash")

    assert chunk_id("d_1", 4, "hash") != baseline
    assert chunk_id("d_2", 3, "hash") != baseline
    assert chunk_id("d_1", 3, "other") != baseline


def test_separators_prevent_field_boundary_collisions() -> None:
    assert chunk_id("d_1", 23, "h") != chunk_id("d_12", 3, "h")


def test_job_ids_are_unique_and_time_ordered() -> None:
    ids = [job_id() for _ in range(5)]

    assert len(set(ids)) == 5
    assert ids == sorted(ids)
    assert all(identifier.startswith("job_") for identifier in ids)


def test_collection_ids_are_random() -> None:
    assert collection_id() != collection_id()
    assert collection_id().startswith("col_")


def test_content_hashes_are_stable_and_distinguishing() -> None:
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")
    assert len(content_hash(b"abc")) == 64


def test_the_chunker_hash_tracks_boundary_settings() -> None:
    baseline = chunker_config_hash(Settings())

    assert (
        chunker_config_hash(Settings.model_validate({"chunking": {"chunk_size": 512}})) != baseline
    )
    assert chunker_config_hash(Settings.model_validate({"chunking": {"overlap": 32}})) != baseline
    assert (
        chunker_config_hash(Settings.model_validate({"chunking": {"strategy": "fixed"}}))
        != baseline
    )


def test_enrichment_does_not_renumber_chunks() -> None:
    enriched = Settings.model_validate({"chunking": {"contextual_enrichment": True}})

    assert chunker_config_hash(enriched) == chunker_config_hash(Settings())


def test_unrelated_settings_do_not_change_the_chunker_hash() -> None:
    unrelated = Settings.model_validate({"app": {"port": 9000}})

    assert chunker_config_hash(unrelated) == chunker_config_hash(Settings())


def test_the_retrieval_hash_tracks_retrieval_affecting_settings() -> None:
    baseline = retrieval_config_hash(Settings())

    assert (
        retrieval_config_hash(Settings.model_validate({"embeddings": {"model": "other"}}))
        != baseline
    )
    assert retrieval_config_hash(Settings.model_validate({"retrieval": {"rrf_k": 30}})) != baseline
    assert (
        retrieval_config_hash(Settings.model_validate({"chunking": {"chunk_size": 512}}))
        != baseline
    )


def test_unrelated_settings_do_not_report_index_drift() -> None:
    for override in ({"app": {"port": 9000}}, {"observability": {"dashboard": True}}):
        assert retrieval_config_hash(Settings.model_validate(override)) == retrieval_config_hash(
            Settings()
        )
