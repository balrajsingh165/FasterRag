"""The decisions `.github/workflows/release.yml` makes, as code that can be run on a laptop.

A release publishes to PyPI, and a PyPI version can never be reused, so the three places
where the workflow decides something rather than runs something live here instead of in
shell inside YAML — where nothing lints them and nobody can execute them before the day
they matter:

* ``decide`` — is this run a release or a rehearsal? It reads the trigger, the ref and
  ``__version__`` and prints ``publish=true`` only for a ``v*`` tag whose name is the tree's
  version, pushed or dispatched with ``dry_run`` switched off. Everything else is either a
  rehearsal or a refusal, and a refusal exits 1 before any later job starts.
* ``notes`` — the GitHub release body is the CHANGELOG section, verbatim. GitHub refuses a
  body over 125,000 characters, and the first release's section is longer than that, so the
  body is cut at a paragraph boundary with a pointer to the complete section, which is
  written beside it and attached to the release.
* ``sbom`` — ``uv export`` writes the project's own component without a version, because the
  version is read from the source at build time. An SBOM that does not say what it describes
  is not much of one, so the version is stamped in and the document's shape is checked.

Standard library only: the ``verify`` job runs this before anything is installed.

Usage::

    python .github/scripts/release.py decide --event push --ref refs/tags/v0.1.0-beta.1
    python .github/scripts/release.py notes --label 0.1.0-beta.1 --ref v0.1.0-beta.1 \
        --repository owner/name --full CHANGELOG-0.1.0-beta.1.md --body release-body.md
    python .github/scripts/release.py sbom fasterrag.cdx.json --version 0.1.0b1

Exit codes: ``0`` done, ``1`` refused (the reason is printed as a workflow error), ``2`` bad
arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
TAG_PREFIX: Final = "refs/tags/v"
BODY_LIMIT: Final = 120_000
"""Characters a release body may hold here; GitHub's own limit is 125,000."""

_VERSION_LINE: Final = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)
_HEADING: Final = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)
_PRERELEASE: Final = re.compile(r"(a|b|rc)\d+$|\.dev\d+$")


class RefusedError(Exception):
    """A reason this run must stop, worded for the person who started it."""


def tree_version(root: Path = REPOSITORY_ROOT) -> str:
    """Return ``__version__`` as the package init states it."""
    found = _VERSION_LINE.search((root / "src" / "fasterrag" / "__init__.py").read_text("utf-8"))
    if found is None:
        raise RefusedError("src/fasterrag/__init__.py has no __version__ line")
    return found.group("version")


def _pep440(semver: str) -> str:
    """Spell a tag's version the way ``__version__`` does, using the stamp script's rule."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    from stamp_release import pep440

    return pep440(semver)


def decide(event: str, ref: str, dry_run: str, version: str) -> dict[str, str]:
    """Return the outputs of the verify job, or raise ``RefusedError``."""
    tagged = ref.startswith(TAG_PREFIX)
    label = version
    if tagged:
        label = ref.removeprefix(TAG_PREFIX)
        try:
            expected = _pep440(label)
        except ValueError as error:
            raise RefusedError(f"tag v{label} is not a release tag: {error}") from error
        if expected != version:
            raise RefusedError(
                f"tag v{label} is {expected} in PEP 440 form but __version__ is {version}; "
                f"run `python scripts/stamp_release.py {label}`, commit, and tag that commit"
            )

    if event == "push":
        if not tagged:
            raise RefusedError(f"a push to {ref} is not a release; only a v* tag is")
        publish = True
    elif event == "workflow_dispatch":
        if dry_run not in {"true", "false"}:
            raise RefusedError(f"dry_run must be true or false, got {dry_run!r}")
        if dry_run == "false" and not tagged:
            raise RefusedError(
                f"dry_run is off but this run is on {ref}, which is not a v* tag; a release "
                "is published from a tag, so dispatch on the tag or leave dry_run on"
            )
        publish = dry_run == "false"
    else:
        raise RefusedError(f"the release workflow does not run on a {event!r} event")

    return {
        "version": version,
        "label": label,
        "publish": "true" if publish else "false",
        "prerelease": "true" if _PRERELEASE.search(version) else "false",
    }


def changelog_section(changelog: str, heading: str) -> str:
    """Return the body under ``## [heading]``, up to the next version heading."""
    headings = list(_HEADING.finditer(changelog))
    for index, found in enumerate(headings):
        if found.group("version") != heading:
            continue
        start = changelog.index("\n", found.start()) + 1
        end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        return changelog[start:end].strip() + "\n"
    raise RefusedError(f"docs/CHANGELOG.md has no `## [{heading}]` section")


def release_body(section: str, pointer: str, limit: int = BODY_LIMIT) -> str:
    """Return the section itself when it fits, else its head and a pointer to the rest."""
    if len(section) <= limit:
        return section
    blocks = section[: limit - len(pointer) - 2].split("\n\n")[:-1]
    while blocks and blocks[-1].lstrip().startswith("#"):
        blocks.pop()
    if not blocks:
        raise RefusedError("the changelog section has no paragraph boundary to cut the body at")
    return "\n\n".join([*blocks, pointer]) + "\n"


def stamp_sbom(document: dict[str, Any], version: str) -> list[str]:
    """Write the version onto the root component; return what is wrong with the document."""
    problems = []
    if document.get("bomFormat") != "CycloneDX":
        problems.append(f"bomFormat is {document.get('bomFormat')!r}, expected 'CycloneDX'")
    root = document.get("metadata", {}).get("component")
    if not isinstance(root, dict) or root.get("name") != "fasterrag":
        problems.append("metadata.component does not describe fasterrag")
    else:
        root["version"] = version
    components = document.get("components") or []
    if not components:
        problems.append("the document lists no components")
    unpinned = [entry.get("name", "?") for entry in components if not entry.get("version")]
    if unpinned:
        problems.append(f"components without a version: {sorted(unpinned)}")
    return problems


def _run_decide(arguments: argparse.Namespace) -> None:
    outputs = decide(arguments.event, arguments.ref, arguments.dry_run, tree_version())
    for key, value in outputs.items():
        sys.stdout.write(f"{key}={value}\n")


def _run_notes(arguments: argparse.Namespace) -> None:
    changelog = (REPOSITORY_ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    section = changelog_section(changelog, arguments.label)
    if not section.strip() and arguments.label != "Unreleased":
        raise RefusedError(f"the `## [{arguments.label}]` section of the changelog is empty")
    target = f"https://github.com/{arguments.repository}/blob/{arguments.ref}/docs/CHANGELOG.md"
    pointer = (
        "_This section is longer than a GitHub release body may be. It continues in "
        f"[docs/CHANGELOG.md]({target}), and the complete section is attached to this "
        f"release as `{Path(arguments.full).name}`._"
    )
    body = release_body(section, pointer)
    Path(arguments.full).write_text(section, encoding="utf-8", newline="\n")
    Path(arguments.body).write_text(body, encoding="utf-8", newline="\n")
    cut = "cut to fit, pointer added" if body != section else "whole section"
    sys.stdout.write(f"release body: {len(body)} of {len(section)} characters ({cut})\n")


def _run_sbom(arguments: argparse.Namespace) -> None:
    path = Path(arguments.path)
    document = json.loads(path.read_text(encoding="utf-8"))
    problems = stamp_sbom(document, arguments.version)
    if problems:
        raise RefusedError(f"{path.name} is not a usable SBOM: " + "; ".join(problems))
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(
        f"{path.name}: CycloneDX {document.get('specVersion')}, "
        f"{len(document['components'])} components, fasterrag {arguments.version}\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; print a refusal as a workflow error and exit 1."""
    parser = argparse.ArgumentParser(description="Decisions the release workflow makes.")
    commands = parser.add_subparsers(dest="command", required=True)

    deciding = commands.add_parser("decide", help="say whether this run publishes")
    deciding.add_argument("--event", required=True)
    deciding.add_argument("--ref", required=True)
    deciding.add_argument("--dry-run", default="", help="the dry_run input; empty on a push")
    deciding.set_defaults(run=_run_decide)

    noting = commands.add_parser("notes", help="write the release body from the changelog")
    noting.add_argument("--label", required=True, help="a changelog heading, e.g. 0.1.0-beta.1")
    noting.add_argument("--ref", required=True, help="the tag or branch the pointer links to")
    noting.add_argument("--repository", required=True, help="owner/name")
    noting.add_argument("--full", required=True, help="where to write the whole section")
    noting.add_argument("--body", required=True, help="where to write the release body")
    noting.set_defaults(run=_run_notes)

    stamping = commands.add_parser("sbom", help="stamp and check a CycloneDX document")
    stamping.add_argument("path")
    stamping.add_argument("--version", required=True)
    stamping.set_defaults(run=_run_sbom)

    arguments = parser.parse_args(argv)
    try:
        arguments.run(arguments)
    except RefusedError as refusal:
        sys.stdout.write(f"::error::{refusal}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
