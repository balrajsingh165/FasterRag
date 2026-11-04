import re
import time
from collections.abc import Sequence

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core import identity
from fasterrag.core.identity import (
    _stable,
    chunk_id,
    chunker_config_hash,
    collection_id,
    content_hash,
    document_id,
    job_id,
    normalise_source,
    retrieval_config_hash,
)
from fasterrag.errors import ErrorCode, IngestionError


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


def test_a_url_is_left_exactly_as_it_was_given() -> None:
    """Object keys are case- and byte-sensitive: rewriting one addresses a different object."""
    assert normalise_source("https://Example.COM/A.PDF") == "https://Example.COM/A.PDF"
    assert normalise_source(r"s3://Bucket/Dir\A.PDF") == r"s3://Bucket/Dir\A.PDF"


def test_two_spellings_of_one_windows_file_are_one_document() -> None:
    """The whole path is case-insensitive on Windows, not only its drive letter."""
    assert normalise_source(r"D:\Docs\A.PDF") == "d:/docs/a.pdf"
    assert document_id(r"D:\Docs\A.PDF") == document_id("d:/docs/a.pdf")


def test_a_colon_slash_that_is_no_drive_letter_keeps_its_case() -> None:
    """Only a single alphabetic drive letter makes a path case-insensitive; POSIX is not."""
    assert normalise_source("file:/Tmp/A.TXT") == "file:/Tmp/A.TXT"
    assert normalise_source("1:/Dir/A.TXT") == "1:/Dir/A.TXT"
    assert document_id("file:/Tmp/A.TXT") != document_id("file:/tmp/a.txt")


def test_a_path_without_a_drive_is_only_separator_normalised() -> None:
    assert normalise_source(r"Docs\Sub\A.PDF") == "Docs/Sub/A.PDF"
    assert document_id(r"Docs\Sub\A.PDF") != document_id("docs/sub/a.pdf")


def test_a_nul_in_a_source_is_a_permanent_validation_failure() -> None:
    """Retrying it would requeue an input no filesystem or URL scheme can ever accept."""
    with pytest.raises(IngestionError) as caught:
        normalise_source("docs/a\x00.pdf")

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert caught.value.retryable is False


class _Clock:
    def __init__(self, ticks: Sequence[float]) -> None:
        self._ticks = list(ticks)

    def __call__(self) -> float:
        if len(self._ticks) > 1:
            return self._ticks.pop(0)
        return self._ticks[0]


def _install_clock(monkeypatch: pytest.MonkeyPatch, *ticks: float) -> None:
    """Pin the wall clock and the minting state so job ids are reproducible."""
    monkeypatch.setattr(identity, "_last_millis", 0)
    monkeypatch.setattr(identity, "_sequence", 0)
    monkeypatch.setattr(time, "time", _Clock(ticks))


_JOB_ID = re.compile(r"^job_(?P<stamp>\d{13})(?P<tie_break>\d{4})[0-9a-f]{10}$")
_COLLECTION_ID = re.compile(r"^col_[0-9a-f]{10}$")


def test_job_and_collection_ids_have_the_documented_shape() -> None:
    """The layout is public — it reaches callers through the REST API, journal and logs."""
    assert _JOB_ID.fullmatch(job_id())
    assert _COLLECTION_ID.fullmatch(collection_id())


def test_a_job_id_carries_the_epoch_millisecond_it_was_minted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamp is read back as a mint time, so its unit and scale are part of the id."""
    _install_clock(monkeypatch, 1793000000.5)

    matched = _JOB_ID.fullmatch(job_id())

    assert matched is not None
    assert matched.group("stamp") == "1793000000500"


def test_the_tie_break_counts_within_a_millisecond_and_resets_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tie-break that never reset would outgrow its four digits and break the ordering."""
    _install_clock(monkeypatch, 1793000000.5, 1793000000.5, 1793000001.5)

    crowded = [job_id(), job_id()]
    later = job_id()

    assert [identifier[17:21] for identifier in crowded] == ["0000", "0001"]
    assert later[17:21] == "0000"
    assert sorted([*crowded, later]) == [*crowded, later]


def test_stable_serialisation_ignores_the_order_keys_were_built_in() -> None:
    """Two equal mappings must hash equally, or an unchanged config reports index drift."""
    assert _stable({"a": 1, "b": 2}) == _stable({"b": 2, "a": 1})
    assert _stable({"outer": {"a": 1, "b": 2}}) == _stable({"outer": {"b": 2, "a": 1}})
