"""Stamp a release version everywhere it must agree, or fail when the places disagree.

Step 3 of ``docs/release.md`` touches four files that each carry the version in a different
shape — ``__version__`` in PEP 440 form, the CHANGELOG heading in SemVer form with a date,
the Trove classifier as a development-status word, and three pre-release banners that promise
to delete themselves on the release that makes them false (TASK-0256). Four hand edits made
under release-day pressure are four chances for one to be missed, and a wheel whose
``__version__`` says beta while its own README says "not yet on PyPI" is the kind of drift a
reader cannot tell from a lie.

So the stamp is one command, and the check is the gate that says whether the tree is in
*either* of the two coherent states:

* **development** — ``__version__`` is a ``.devN`` version, the CHANGELOG has only an
  ``[Unreleased]`` section above its history, the classifier is Pre-Alpha, and the banners
  are present;
* **released** — ``__version__`` is a release version, the CHANGELOG's first versioned
  heading matches it and carries a date, the classifier matches the version's maturity, and
  the banners are gone.

Anything in between is a half-done stamp, and the check names every place that is out of
step so the commit that finishes it cannot be made blind. It runs in CI and in the unit suite.

Decided 2026-09-13 (TASK-0020): the first beta is ``0.1.0-beta.1`` — tag ``v0.1.0-beta.1``,
PEP 440 ``0.1.0b1``. The stamp itself happens at release step 3, after the readiness gate and
the freeze, never before: a beta label on an unfrozen tree would name builds that are not the
beta, and a dated CHANGELOG heading for a release that has not happened is a false record.

Usage::

    python scripts/stamp_release.py --check
    python scripts/stamp_release.py 0.1.0-beta.1 --date 2026-10-01

Exit codes: ``0`` coherent (or stamped), ``1`` drift reported, ``2`` bad arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

# The SemVer shape the release procedure uses for tags and CHANGELOG headings. Deliberately
# narrower than SemVer: only the pre-release labels PEP 440 has a direct spelling for, so the
# two forms convert without a normalisation library and without ambiguity.
_SEMVER: Final = re.compile(
    r"^(?P<release>\d+\.\d+\.\d+)(?:-(?P<label>alpha|beta|rc)\.(?P<number>\d+))?$"
)
_PEP440_LABEL: Final = {"alpha": "a", "beta": "b", "rc": "rc"}
_PEP440: Final = re.compile(r"^(?P<release>\d+\.\d+\.\d+)(?:(?P<label>a|b|rc)(?P<number>\d+))?$")
_SEMVER_LABEL: Final = {"a": "alpha", "b": "beta", "rc": "rc"}
_VERSION_LINE: Final = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)
_UNRELEASED: Final = re.compile(r"^## \[Unreleased\][ \t]*$", re.MULTILINE)
_HEADING: Final = re.compile(r"^## \[(?P<version>[^\]]+)\](?: — (?P<date>\d{4}-\d{2}-\d{2}))?\s*$")
_CLASSIFIER: Final = re.compile(
    r'^(?P<indent>\s*)"Development Status :: (?P<status>[^"]+)",$', re.MULTILINE
)
_STATUS_FOR: Final = {
    "alpha": "3 - Alpha",
    "beta": "4 - Beta",
    "rc": "4 - Beta",
    None: "5 - Production/Stable",
}
_DEVELOPMENT_STATUS: Final = "2 - Pre-Alpha"

# Each banner is identified by a phrase that exists only to be deleted at release; the check
# looks for the phrase, not the sentence, so rewording around it does not defeat the gate.
_BANNERS: Final = (
    ("README.md", "not yet on PyPI"),
    ("docs/python-api.md", "Not yet published"),
    ("docs/how-to-use.md", "Until the first release lands"),
)


@dataclass(frozen=True)
class Tree:
    """The files a stamp touches, resolved against one root so tests can use a fixture."""

    root: Path

    @property
    def init(self) -> Path:
        """The package init that hatch reads the version from."""
        return self.root / "src" / "fasterrag" / "__init__.py"

    @property
    def changelog(self) -> Path:
        """The Keep-a-Changelog file whose Unreleased section becomes the release."""
        return self.root / "docs" / "CHANGELOG.md"

    @property
    def pyproject(self) -> Path:
        """The project metadata carrying the development-status classifier."""
        return self.root / "pyproject.toml"


def _read(path: Path) -> str:
    """Read with newlines normalised to LF, so one regex serves both line-ending conventions."""
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    """Write back with the line ending the file already used.

    A Windows checkout under ``core.autocrlf`` holds CRLF; CI holds LF. ``write_text`` picks
    the platform's ending, not the file's, which on the wrong side turns a one-line stamp
    into a whole-file diff and a spurious modification of every other line.
    """
    newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def pep440(semver: str) -> str:
    """Return the PEP 440 spelling of a release-procedure version, or raise ``ValueError``."""
    match = _SEMVER.match(semver)
    if match is None:
        raise ValueError(
            f"{semver!r} is not a release version; expected X.Y.Z or X.Y.Z-(alpha|beta|rc).N"
        )
    if match.group("label") is None:
        return match.group("release")
    return f"{match.group('release')}{_PEP440_LABEL[match.group('label')]}{match.group('number')}"


def _maturity(version: str) -> str | None:
    """Return the pre-release label of a version in either spelling, None for a final."""
    match = _SEMVER.match(version)
    if match is not None:
        return match.group("label")
    match = _PEP440.match(version)
    if match is None:
        raise ValueError(f"{version!r} is not a release version")
    label = match.group("label")
    return None if label is None else _SEMVER_LABEL[label]


def _current_version(tree: Tree) -> str:
    match = _VERSION_LINE.search(_read(tree.init))
    if match is None:
        raise ValueError(f"{tree.init} has no __version__ line")
    return match.group("version")


def _first_heading(changelog: str) -> tuple[str, str | None] | None:
    """Return the first versioned heading after Unreleased, or None when history is empty."""
    for line in changelog.splitlines():
        heading = _HEADING.match(line)
        if heading is None or heading.group("version") == "Unreleased":
            continue
        return heading.group("version"), heading.group("date")
    return None


def _classifier(pyproject: str) -> str | None:
    match = _CLASSIFIER.search(pyproject)
    return None if match is None else match.group("status")


def check(tree: Tree | None = None) -> list[str]:
    """Return one message per place out of step with ``__version__``; empty when coherent."""
    tree = tree or Tree(REPOSITORY_ROOT)
    problems: list[str] = []
    version = _current_version(tree)
    changelog = _read(tree.changelog)
    heading = _first_heading(changelog)
    status = _classifier(_read(tree.pyproject))
    banners = [
        f"{path} still carries the pre-release banner {phrase!r}"
        for path, phrase in _BANNERS
        if (tree.root / path).exists() and phrase in _read(tree.root / path)
    ]

    if _UNRELEASED.search(changelog) is None:
        problems.append(
            "CHANGELOG has no [Unreleased] section; keep an empty one above the history"
        )

    if ".dev" in version:
        if heading is not None:
            problems.append(
                f"__version__ is {version} but the CHANGELOG already has a release heading "
                f"[{heading[0]}]; a dev tree has only Unreleased above its history"
            )
        if status != _DEVELOPMENT_STATUS:
            problems.append(
                f"__version__ is {version} but the classifier says {status!r}; "
                f"expected {_DEVELOPMENT_STATUS!r}"
            )
        if len(banners) != len(_BANNERS):
            present = {path for path, _ in _BANNERS} - {entry.split(" ")[0] for entry in banners}
            problems.append(
                f"__version__ is {version} but the pre-release banner is missing from "
                f"{sorted(present)}; it is deleted by the release that makes it false"
            )
        return problems

    try:
        expected_status = _STATUS_FOR[_maturity(version)]
    except ValueError as error:
        problems.append(f"__version__ is neither a dev nor a release version: {error}")
    else:
        if status != expected_status:
            problems.append(f"classifier says {status!r} but {version} is {expected_status!r}")

    if heading is None:
        problems.append(f"__version__ is {version} but the CHANGELOG has no release heading")
    else:
        heading_version, heading_date = heading
        try:
            expected = pep440(heading_version)
        except ValueError as error:
            problems.append(
                f"CHANGELOG heading [{heading_version}] is not a release version: {error}"
            )
        else:
            if expected != version:
                problems.append(
                    f"__version__ is {version} but the CHANGELOG's latest heading is "
                    f"[{heading_version}] ({expected})"
                )
            if heading_date is None:
                problems.append(f"CHANGELOG heading [{heading_version}] carries no release date")
    problems.extend(banners)
    return problems


def stamp(semver: str, date: dt.date, tree: Tree | None = None) -> str:
    """Write the version into every mechanical place and return its PEP 440 form.

    The banners are prose and are not rewritten here — ``check`` reports them until the
    release commit removes them by hand, which is the point: the stamp commit cannot be made
    while a document still says the package is unpublished.
    """
    tree = tree or Tree(REPOSITORY_ROOT)
    version = pep440(semver)

    init = _read(tree.init)
    if _VERSION_LINE.search(init) is None:
        raise ValueError(f"{tree.init} has no __version__ line")
    _write(tree.init, _VERSION_LINE.sub(f'__version__ = "{version}"', init, count=1))

    changelog = _read(tree.changelog)
    if _UNRELEASED.search(changelog) is None:
        raise ValueError(f"{tree.changelog} has no [Unreleased] section to convert")
    _write(
        tree.changelog,
        _UNRELEASED.sub(
            f"## [Unreleased]\n\n## [{semver}] — {date.isoformat()}", changelog, count=1
        ),
    )

    pyproject = _read(tree.pyproject)
    if _CLASSIFIER.search(pyproject) is None:
        raise ValueError(f"{tree.pyproject} has no Development Status classifier")
    status = _STATUS_FOR[_maturity(semver)]
    _write(
        tree.pyproject,
        _CLASSIFIER.sub(f'\\g<indent>"Development Status :: {status}",', pyproject, count=1),
    )
    return version


def main(argv: list[str] | None = None) -> int:
    """Run the check, or stamp the given version, per the docstring's usage."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", nargs="?", help="release version, e.g. 0.1.0-beta.1")
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--check", action="store_true", help="report drift instead of stamping")
    args = parser.parse_args(argv)

    if args.check == bool(args.version):
        parser.error("give exactly one of --check or a version")

    if args.check:
        problems = check()
        for problem in problems:
            sys.stderr.write(f"stamp_release: {problem}\n")
        return 1 if problems else 0

    try:
        version = stamp(args.version, args.date)
    except ValueError as error:
        parser.error(str(error))
    sys.stdout.write(f"stamped {args.version} ({version}) dated {args.date.isoformat()}\n")
    remaining = check()
    for problem in remaining:
        sys.stderr.write(f"stamp_release: still to do by hand: {problem}\n")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
