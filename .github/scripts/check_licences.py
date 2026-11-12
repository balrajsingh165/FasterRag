"""Fail when a locked dependency's licence is not one fasterRag has agreed to depend on.

fasterRag is Apache-2.0 (ADR-0008). Nothing checked what its dependencies were licensed
under: a transitive package could move to a licence the project cannot ship beside, the
lockfile would be refreshed, and every gate would stay green (TASK-0496).

The check reads the *lockfile*, as the dependency audit beside it does, and for the same
reason: an installed environment is whatever one platform resolved today, while ``uv.lock``
names every package any platform can be given. The Linux-only CUDA runtime that ``torch``
pulls in is the case that matters — it is absent from a Windows or macOS environment and is
the least permissively licensed thing in the set.

For each ``name==version`` in the export it asks PyPI what that release declares, in the
order a reader would trust it:

1. ``License-Expression`` (PEP 639) — an SPDX expression, evaluated as written: every side
   of an ``AND`` must be allowed, one side of an ``OR`` is enough, and a ``WITH`` exception
   only ever adds permissions, so it is judged by the licence it modifies.
2. the free-text ``License`` field, when it is an SPDX expression or a spelling in
   ``ALIASES``. A whole licence text pasted into that field is neither, and falls through.
3. the ``License ::`` Trove classifiers, all of which must be allowed. A classifier pair
   does not say whether it means *and* or *or*, so it is read as *and*.

A package is then one of four things, and only the first two pass:

* **allowed** — everything it declares is in ``ALLOWED``;
* **reviewed** — named in ``REVIEWED`` with the licence it declared when it was reviewed
  and the reason it is accepted. The exception holds only while the declaration is
  unchanged, and an entry whose package has left the lockfile fails as stale;
* **refused** — it declares a licence that is not allowed;
* **undetermined** — it declares nothing this script can read. That is a failure, not a
  pass: "the tool could not tell" is how an unlicensed package gets in.

``ALLOWED`` is deliberately short. It holds what the current dependency set uses, not every
permissive licence that exists, so a licence nobody here has looked at arrives as a red
build and a one-line decision rather than silently.

Usage::

    uv export --format requirements-txt --locked --all-extras --no-extra dev \
        --no-emit-project --no-hashes -o locked.txt
    python .github/scripts/check_licences.py locked.txt [--summary report.md]

Exit codes: ``0`` every package is allowed or reviewed, ``1`` at least one is refused,
undetermined or stale, ``2`` bad arguments, ``3`` PyPI could not be asked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "CC0-1.0",
        "CNRI-Python",
        "ISC",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "MPL-2.0",
        "PSF-2.0",
        "Zlib",
    }
)
"""SPDX identifiers a dependency may carry.

All permissive except MPL-2.0, which is copyleft per *file*: it binds whoever modifies or
redistributes those files, and fasterRag does neither — it names the package as a
dependency and pip fetches it.
"""

ALIASES: Final[dict[str, str]] = {
    "3-clause bsd license": "BSD-3-Clause",
    "apache 2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license version 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "isc license": "ISC",
    "mit license": "MIT",
}
"""Free-text spellings of an allowed licence, lower-cased. Each one was seen in the lockfile.

A bare ``BSD`` does not say which; every variant it can mean is allowed.
"""

CLASSIFIERS: Final[dict[str, str]] = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}
"""Trove classifiers and the SPDX identifier each is judged as.

``BSD License`` does not say which BSD; every variant it can mean is allowed, so it is
judged as the three-clause one.
"""

_NVIDIA: Final = (
    "NVIDIA's proprietary CUDA runtime, pulled in by torch on Linux only and only by the "
    "huggingface, rerank and all extras. fasterRag does not distribute it: pip downloads it "
    "from PyPI under NVIDIA's own terms, and the published container image is built without "
    "extras, so it holds none of it. Most of these packages declare no licence at all."
)
_PSYCOPG: Final = (
    "LGPL-3.0, which is not on the allow-list. Accepted because it is reached only through "
    "the optional pgvector extra, is imported and never copied or modified, and is not in "
    "the published container image; the LGPL's terms fall on whoever redistributes psycopg."
)


@dataclass(frozen=True)
class Review:
    """An accepted exception: what the package declared when reviewed, and why it passes."""

    declared: frozenset[str]
    reason: str


REVIEWED: Final[dict[str, Review]] = {
    "psycopg": Review(frozenset({"LGPL-3.0-only"}), _PSYCOPG),
    "psycopg-binary": Review(frozenset({"LGPL-3.0-only"}), _PSYCOPG),
    "psycopg-pool": Review(frozenset({"LGPL-3.0-only"}), _PSYCOPG),
    "pypdfium2": Review(
        frozenset({"BSD-3-Clause, Apache-2.0, dependency licenses"}),
        "Declared in prose rather than SPDX: the bindings are BSD-3-Clause or Apache-2.0, and "
        "the PDFium binary in the wheel carries its own third-party notices, which this "
        "script cannot read. A core dependency, through pdfplumber.",
    ),
}
"""Exceptions by exact package name. Delete an entry when its package leaves the lockfile."""

REVIEWED_FAMILIES: Final[dict[str, Review]] = {
    "nvidia-": Review(
        frozenset({"", "NVIDIA Proprietary Software", "LicenseRef-NVIDIA-Proprietary"}), _NVIDIA
    ),
    "cuda-toolkit": Review(frozenset({""}), _NVIDIA),
}
"""Exceptions by name prefix, for a vendor family torch renames with every CUDA release."""

_REQUIREMENT: Final = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)")
_TOKEN: Final = re.compile(r"\s*(\(|\)|[A-Za-z0-9.+-]+)")
_OPERATORS: Final = frozenset({"AND", "OR", "WITH"})
_ATTEMPTS: Final = 4
_INDEX: Final = "https://pypi.org/pypi"


class UnreadableError(Exception):
    """A string that is not an SPDX expression."""


class UnreachableError(Exception):
    """PyPI could not be asked about a release, so nothing is known about it."""


@dataclass(frozen=True)
class Verdict:
    """What one locked release declares and what the policy makes of it."""

    name: str
    version: str
    declared: str
    status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Whether this release may stay in the lockfile."""
        return self.status in {"allowed", "reviewed"}


def normalise(name: str) -> str:
    """Return a distribution name in its canonical form (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_releases(export: str) -> list[tuple[str, str]]:
    """Return every ``(name, version)`` a requirements export pins, once each."""
    found = {
        (normalise(match.group("name")), match.group("version"))
        for match in map(_REQUIREMENT.match, export.splitlines())
        if match is not None
    }
    return sorted(found)


def _tokens(expression: str) -> list[str]:
    tokens = []
    position = 0
    while position < len(expression.rstrip()):
        found = _TOKEN.match(expression, position)
        if found is None:
            raise UnreadableError(expression)
        tokens.append(found.group(1))
        position = found.end()
    return tokens


def disallowed(expression: str) -> list[str]:
    """Return the identifiers that stop an SPDX expression being allowed; empty when it is."""
    tokens = _tokens(expression)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        if position >= len(tokens):
            raise UnreadableError(expression)
        position += 1
        return tokens[position - 1]

    def licence() -> list[str]:
        token = take()
        if token == "(":
            inner = either()
            if take() != ")":
                raise UnreadableError(expression)
            return inner
        if token == ")" or token.upper() in _OPERATORS:
            raise UnreadableError(expression)
        if (peek() or "").upper() == "WITH":
            take()
            if take().upper() in _OPERATORS | {"(", ")"}:
                raise UnreadableError(expression)
        return [] if token in ALLOWED else [token]

    def every() -> list[str]:
        problems = licence()
        while (peek() or "").upper() == "AND":
            take()
            problems = [*problems, *licence()]
        return problems

    def either() -> list[str]:
        problems = every()
        while (peek() or "").upper() == "OR":
            take()
            alternative = every()
            problems = [] if not problems or not alternative else [*problems, *alternative]
        return problems

    result = either()
    if position != len(tokens) or not tokens:
        raise UnreadableError(expression)
    return result


def declaration(metadata: dict[str, Any]) -> tuple[str, str | None]:
    """Return what a release declares, verbatim, and the SPDX expression it is judged as."""
    expression = (metadata.get("license_expression") or "").strip()
    if expression:
        return expression, expression

    text = (metadata.get("license") or "").strip()
    classifiers = [
        entry for entry in metadata.get("classifiers") or [] if entry.startswith("License ::")
    ]
    if text.lower() in ALIASES:
        return text, ALIASES[text.lower()]
    if text and "\n" not in text:
        try:
            disallowed(text)
        except UnreadableError:
            if not classifiers:
                return text, None
        else:
            return text, text

    if classifiers:
        shown = "; ".join(classifiers)
        if all(entry in CLASSIFIERS for entry in classifiers):
            return shown, " AND ".join(CLASSIFIERS[entry] for entry in classifiers)
        return shown, None
    return (text.splitlines()[0][:60] + " ...", None) if text else ("", None)


def review_for(name: str) -> Review | None:
    """Return the exception that covers a package, by exact name first and family second."""
    if name in REVIEWED:
        return REVIEWED[name]
    return next(
        (review for prefix, review in REVIEWED_FAMILIES.items() if name.startswith(prefix)), None
    )


def judge(name: str, version: str, metadata: dict[str, Any]) -> Verdict:
    """Return the verdict on one release from the metadata PyPI holds for it."""
    declared, expression = declaration(metadata)
    review = review_for(name)
    if review is not None:
        if declared in review.declared:
            return Verdict(name, version, declared, "reviewed", review.reason)
        return Verdict(
            name,
            version,
            declared,
            "refused",
            f"reviewed when it declared {sorted(review.declared)}; it now declares something "
            "else, so the exception no longer applies and the review must be repeated",
        )
    if expression is None:
        return Verdict(
            name,
            version,
            declared,
            "undetermined",
            "declares no licence this check can read; read the project's own licence file, "
            "then add an alias or a reviewed exception",
        )
    try:
        problems = disallowed(expression)
    except UnreadableError:
        return Verdict(name, version, declared, "undetermined", "not an SPDX expression")
    if problems:
        return Verdict(
            name, version, declared, "refused", f"not on the allow-list: {sorted(set(problems))}"
        )
    return Verdict(name, version, declared, "allowed")


def stale_reviews(names: set[str]) -> list[str]:
    """Return the exceptions that no longer match any locked package."""
    stale = [name for name in REVIEWED if name not in names]
    stale.extend(
        prefix for prefix in REVIEWED_FAMILIES if not any(name.startswith(prefix) for name in names)
    )
    return sorted(stale)


def fetch_metadata(name: str, version: str) -> dict[str, Any]:
    """Return the ``info`` mapping PyPI holds for one release, retrying a failed request."""
    request = urllib.request.Request(
        f"{_INDEX}/{name}/{version}/json", headers={"User-Agent": "licence-check"}
    )
    failure = ""
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                info: Any = json.load(response)["info"]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
            failure = f"{type(error).__name__}: {error}"
            time.sleep(2**attempt)
        else:
            if isinstance(info, dict):
                return info
            failure = "the response carried no info mapping"
    raise UnreachableError(f"{name}=={version}: {failure}")


def report(verdicts: list[Verdict], stale: list[str]) -> str:
    """Return the Markdown report: counts, then every package that is not plainly allowed."""
    counts = Counter(verdict.status for verdict in verdicts)
    declared = Counter(verdict.declared or "(nothing declared)" for verdict in verdicts)
    lines = [
        "## Dependency licences",
        "",
        f"{len(verdicts)} locked releases: "
        + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())),
        "",
        "| Declared | Releases |",
        "|---|---|",
        *(f"| `{text}` | {count} |" for text, count in declared.most_common()),
        "",
        "| Package | Declared | Status | Why |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {verdict.name} {verdict.version} | `{verdict.declared or '(nothing declared)'}` "
        f"| {verdict.status} | {verdict.detail} |"
        for verdict in verdicts
        if verdict.status != "allowed"
    )
    lines.extend(
        f"| {name} | | stale | reviewed, but no longer in the lockfile |" for name in stale
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Judge every release in a requirements export; report, and exit per the docstring."""
    parser = argparse.ArgumentParser(
        description="Check locked dependencies against a licence list."
    )
    parser.add_argument("export", help="a `uv export --format requirements-txt` file")
    parser.add_argument("--summary", help="append the Markdown report to this file")
    arguments = parser.parse_args(argv)

    releases = locked_releases(Path(arguments.export).read_text(encoding="utf-8"))
    if not releases:
        sys.stderr.write(f"check_licences: {arguments.export} pins nothing\n")
        return 2
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            metadata = list(pool.map(lambda release: fetch_metadata(*release), releases))
    except UnreachableError as error:
        sys.stderr.write(f"check_licences: could not ask PyPI about {error}\n")
        return 3

    verdicts = [
        judge(name, version, info) for (name, version), info in zip(releases, metadata, strict=True)
    ]
    stale = stale_reviews({name for name, _ in releases})
    document = report(verdicts, stale)
    if arguments.summary:
        with Path(arguments.summary).open("a", encoding="utf-8") as handle:
            handle.write(document)

    failures = [verdict for verdict in verdicts if not verdict.passed]
    for verdict in verdicts:
        if verdict.status == "reviewed":
            sys.stdout.write(
                f"reviewed: {verdict.name} {verdict.version} "
                f"declares {verdict.declared or 'nothing'!r}\n"
            )
    for verdict in failures:
        sys.stderr.write(
            f"check_licences: {verdict.status}: {verdict.name} {verdict.version} "
            f"declares {verdict.declared or 'nothing'!r} -- {verdict.detail}\n"
        )
    for name in stale:
        sys.stderr.write(
            f"check_licences: stale: the reviewed exception for {name!r} matches nothing in "
            "the lockfile; delete it\n"
        )
    counts = Counter(verdict.status for verdict in verdicts)
    sys.stdout.write(
        f"licences: {len(verdicts)} locked releases -- "
        + ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        + "\n"
    )
    return 1 if failures or stale else 0


if __name__ == "__main__":
    sys.exit(main())
