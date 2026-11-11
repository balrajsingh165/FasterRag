"""Read portability-archive (``.fragx``) members without the latent ``None`` deref.

``TarFile.extractfile`` returns ``IO[bytes] | None`` — ``None`` for a member that is not a
regular file — and raises ``KeyError`` for a name the archive does not hold. Five call
sites across the archive tests wrote ``tar.extractfile(member).read()`` behind a
``# type: ignore[union-attr]``, so a directory member silently became an ``AttributeError``
about ``NoneType`` and a missing member a bare ``KeyError``, neither naming the archive.
D11's writer (``services/archive.py::_write_tar``) emits regular files only, so either
shape means the archive is malformed; these helpers say so in the failure message instead
of asserting the ``None`` away.
"""

from __future__ import annotations

import tarfile


def member_bytes(tar: tarfile.TarFile, member: str | tarfile.TarInfo) -> bytes:
    """Return one archive member's bytes, failing loudly when it carries no stream."""
    name = member if isinstance(member, str) else member.name
    try:
        stream = tar.extractfile(member)
    except KeyError as exc:
        raise AssertionError(
            f"archive member {name!r} is absent; it holds {sorted(tar.getnames())}"
        ) from exc
    if stream is None:
        raise AssertionError(
            f"archive member {name!r} has no readable stream, so it is not a regular file; "
            "the archive writer emits regular files only"
        )
    with stream:
        return stream.read()


def archive_members(tar: tarfile.TarFile) -> dict[str, bytes]:
    """Return every member of an open archive as a name-to-bytes mapping."""
    return {member.name: member_bytes(tar, member) for member in tar.getmembers()}
