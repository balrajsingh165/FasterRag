"""The archive test helper must fail loudly, since it replaced an asserted-away ``None``.

``tests/support/archive.py`` exists because five call sites wrote
``tar.extractfile(member).read()`` behind ``# type: ignore[union-attr]``. Moving the deref
into one helper is only an improvement if the helper's failure says what went wrong, so
the two shapes ``extractfile`` can produce — ``None`` for a non-regular member, ``KeyError``
for an absent name — are exercised here rather than trusted.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from tests.support.archive import archive_members, member_bytes


def _archive(tmp_path: Path, *, with_directory: bool = False) -> Path:
    """Write a small gzipped tar holding one file member, optionally plus a directory."""
    payload = tmp_path / "manifest.json"
    payload.write_text('{"format_version": 1}', encoding="utf-8")
    target = tmp_path / "corpus.fragx"
    with tarfile.open(target, "w:gz") as tar:
        tar.add(payload, arcname="manifest.json")
        if with_directory:
            info = tarfile.TarInfo("vectors")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
    return target


def test_a_members_bytes_come_back_whole(tmp_path: Path) -> None:
    with tarfile.open(_archive(tmp_path)) as tar:
        assert member_bytes(tar, "manifest.json") == b'{"format_version": 1}'


def test_every_member_is_returned_by_name(tmp_path: Path) -> None:
    with tarfile.open(_archive(tmp_path)) as tar:
        assert archive_members(tar) == {"manifest.json": b'{"format_version": 1}'}


def test_an_absent_member_names_itself_and_what_the_archive_holds(tmp_path: Path) -> None:
    """A bare ``KeyError`` from ``extractfile`` never says which archive was short."""
    with tarfile.open(_archive(tmp_path)) as tar, pytest.raises(AssertionError) as caught:
        member_bytes(tar, "vectors.jsonl")

    assert "vectors.jsonl" in str(caught.value)
    assert "manifest.json" in str(caught.value)


def test_a_member_with_no_stream_is_reported_rather_than_dereferenced(tmp_path: Path) -> None:
    """The latent deref this helper replaced: ``extractfile`` returns ``None`` here."""
    archive = _archive(tmp_path, with_directory=True)

    with tarfile.open(archive) as tar, pytest.raises(AssertionError) as caught:
        member_bytes(tar, "vectors")

    assert "no readable stream" in str(caught.value)


def test_a_non_regular_member_fails_the_whole_read_rather_than_vanishing(tmp_path: Path) -> None:
    """Skipping it would hide exactly the malformed archive D11's readers must reject."""
    archive = _archive(tmp_path, with_directory=True)

    with tarfile.open(archive) as tar, pytest.raises(AssertionError):
        archive_members(tar)
