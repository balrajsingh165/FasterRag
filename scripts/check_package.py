"""Prove a built wheel installs and imports the way a `pip install` user gets it.

See docs/verification.md for why this cannot be a plain pytest module.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

# Every symbol docs/python-api.md names as SemVer-stable. A name that stops importing from a
# built wheel is a broken promise to an adopter, and nothing else in the suite would notice:
# the unit run puts src/ on sys.path, so it imports the checkout rather than the package.
PUBLIC_IMPORTS: Final = (
    "import fasterrag; assert fasterrag.__version__",
    "from fasterrag import FasterRag",
    "from fasterrag.sync import FasterRag",
    "from fasterrag.errors import FasterRagError, ErrorCode",
    "from fasterrag.parsing import parse_document",
    "from fasterrag.chunking import RecursiveChunker, SemanticChunker, LayoutChunker",
    "from fasterrag.retrieval import rrf_fuse",
    "from fasterrag.evals import evaluate",
)

# Imported lazily by design, so a core install must NOT pull them. Asserting their absence is
# what keeps the dependency promise honest: `pip install fasterrag` claims not to drag in a
# multi-gigabyte deep-learning stack, and only a core-only environment can prove it.
EXTRA_ONLY_MODULES: Final = ("torch", "transformers", "sentence_transformers")


def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing output, without raising on failure."""
    return subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)  # type: ignore[call-overload,no-any-return]


def _python_in(environment: Path) -> Path:
    """Return the interpreter inside a created virtual environment."""
    # CRITICAL: a conditional expression, not an if/return pair. mypy narrows sys.platform
    # per platform, so the second `return` of an if/return would be unreachable code on
    # Windows and `warn_unreachable` makes that an error — the trap registered as TASK-0263,
    # which kept CI red on Linux for a month while passing on every developer machine.
    leaf = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    return environment.joinpath(*leaf)


def _build(into: Path) -> list[str]:
    """Build wheel and sdist into a directory, returning problems."""
    built = _run([sys.executable, "-m", "build", "--outdir", str(into), str(REPOSITORY_ROOT)])
    if built.returncode != 0:
        return [f"the build failed:\n{built.stdout}\n{built.stderr}"]

    wheels = list(into.glob("*.whl"))
    sdists = list(into.glob("*.tar.gz"))
    problems = []
    if len(wheels) != 1:
        problems.append(f"expected exactly one wheel, got {[w.name for w in wheels]}")
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist, got {[s.name for s in sdists]}")
    return problems


def _check_imports(interpreter: Path) -> list[str]:
    """Import each documented public symbol in the clean environment."""
    problems = []
    for statement in PUBLIC_IMPORTS:
        imported = _run([str(interpreter), "-c", statement])
        if imported.returncode != 0:
            problems.append(f"documented import failed: {statement}\n{imported.stderr.strip()}")
    return problems


def _check_core_stays_small(interpreter: Path) -> list[str]:
    """Confirm a core install did not pull the optional heavy stack."""
    problems = []
    for module in EXTRA_ONLY_MODULES:
        found = _run([str(interpreter), "-c", f"import {module}"])
        if found.returncode == 0:
            problems.append(
                f"{module!r} imports from a core install; it belongs to an extra, and a user "
                "who asked for neither should not be paying for it"
            )
    return problems


def _check_cli(interpreter: Path) -> list[str]:
    """Confirm the console script is installed and runs."""
    executable = interpreter.parent / ("fasterrag.exe" if sys.platform == "win32" else "fasterrag")
    if not executable.exists():
        return [f"the fasterrag console script was not installed at {executable}"]

    helped = _run([str(executable), "--help"])
    if helped.returncode != 0:
        return [f"`fasterrag --help` failed:\n{helped.stderr.strip()}"]
    return []


def _check_packaged_config(interpreter: Path) -> list[str]:
    """Confirm the shipped config.yaml is byte-identical to the repository's."""
    located = _run(
        [
            str(interpreter),
            "-c",
            "import fasterrag, pathlib, sys;"
            "packaged = pathlib.Path(fasterrag.__file__).parent / 'data' / 'config.yaml';"
            "sys.stdout.write(str(packaged))",
        ]
    )
    if located.returncode != 0:
        return [f"could not locate the packaged config.yaml:\n{located.stderr.strip()}"]

    packaged = Path(located.stdout.strip())
    if not packaged.is_file():
        return [f"the wheel did not ship config.yaml (looked at {packaged})"]

    # CRITICAL: compared with newlines normalised. git may check the repository copy out as
    # CRLF on Windows while the wheel carries what the builder wrote, and a difference that
    # exists only in line endings would fail a check meant to catch drifted content.
    repository = (REPOSITORY_ROOT / "config.yaml").read_text(encoding="utf-8")
    if packaged.read_text(encoding="utf-8") != repository:
        return ["the packaged config.yaml has drifted from the repository's"]
    return []


def check(keep: bool = False) -> list[str]:
    """Build, install into a clean venv, and verify the result. Returns problems."""
    workspace = Path(tempfile.mkdtemp(prefix="fasterrag-package-"))
    try:
        distributions = workspace / "dist"
        problems = _build(distributions)
        if problems:
            return problems

        wheel = next(distributions.glob("*.whl"))
        environment = workspace / "venv"
        venv.create(environment, with_pip=True, clear=True)
        interpreter = _python_in(environment)

        installed = _run([str(interpreter), "-m", "pip", "install", "--quiet", str(wheel)])
        if installed.returncode != 0:
            return [f"the wheel did not install into a clean venv:\n{installed.stderr}"]

        problems.extend(_check_imports(interpreter))
        problems.extend(_check_core_stays_small(interpreter))
        problems.extend(_check_cli(interpreter))
        problems.extend(_check_packaged_config(interpreter))
        return problems
    finally:
        if not keep:
            shutil.rmtree(workspace, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Report problems on stderr; exit 0 when the package is sound."""
    parser = argparse.ArgumentParser(description="Verify the built package imports cleanly.")
    parser.add_argument("--keep", action="store_true", help="leave the build and venv in place")
    arguments = parser.parse_args(argv)

    problems = check(keep=arguments.keep)
    for problem in problems:
        sys.stderr.write(f"check_package: {problem}\n")
    if not problems:
        sys.stdout.write(
            "package ok: wheel builds, installs clean, and every documented import works\n"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
