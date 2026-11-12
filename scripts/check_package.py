"""Prove a built wheel installs and works the way a `pip install` user gets it.

Two environments. The first is the bare core install: every documented import works, the
console script runs, the packaged config.yaml is the repository's, nothing an extra provides
is present, and each extra that is missing is named when it is needed. The second adds only
the extras the local profile needs and runs it end to end: `config init --profile local`,
ingest a Markdown file, answer a `retrieve` call, with no network service and no API key
(TASK-0444). See docs/verification.md for why this cannot be a plain pytest module.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

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

EXTRA_ONLY_MODULES: Final = (
    "torch",
    "transformers",
    "sentence_transformers",
    "fastapi",
    "uvicorn",
    "qdrant_client",
    "pdfplumber",
    "docx",
    "bs4",
    "lxml",
)

NAMED_EXTRAS: Final = (
    (
        "a PDF document",
        "from fasterrag.errors import ParseError\n"
        "from fasterrag.parsing import parse_bytes\n"
        "try:\n"
        "    parse_bytes(b'%PDF-1.4', filename='report.pdf')\n"
        "except ParseError as exc:\n"
        "    print(exc.detail)\n",
        'pip install "fasterrag[pdf]"',
    ),
    (
        "the default vector database",
        "from fasterrag.adapters.vectordb.factory import create_vector_db_adapter\n"
        "from fasterrag.config.schema import Settings\n"
        "from fasterrag.errors import ConfigError\n"
        "try:\n"
        "    create_vector_db_adapter(Settings())\n"
        "except ConfigError as exc:\n"
        "    print(exc.detail)\n",
        'pip install "fasterrag[qdrant]"',
    ),
)

LOCAL_PROFILE_EXTRAS: Final = "chroma,huggingface"

LOCAL_PROFILE_ABSENT: Final = ("fastapi", "qdrant_client", "pdfplumber", "docx", "bs4")

# CRITICAL: PyPI's Linux torch wheel pulls the CUDA runtime, gigabytes the local profile never
# loads. The CPU index serves `+cpu` builds, which pip prefers over the same version from PyPI
# because a local version label sorts higher. `--torch-index ""` turns it off.
TORCH_CPU_INDEX: Final = "https://download.pytorch.org/whl/cpu"

LOCAL_PROFILE_SCRIPT: Final = """
import asyncio
import pathlib
import subprocess
import sys

import fasterrag
from fasterrag import FasterRag

assert "site-packages" in fasterrag.__file__, fasterrag.__file__
console = pathlib.Path(sys.argv[1])
subprocess.run([str(console), "config", "init", "--profile", "local"], check=True)
pathlib.Path("note.md").write_text(
    "# Lighthouses\\n\\nA Fresnel lens lets a lighthouse beam carry for many miles.\\n\\n"
    "## Bread\\n\\nSourdough rises because wild yeast and lactic acid bacteria ferment it.\\n",
    encoding="utf-8",
)
subprocess.run([str(console), "ingest", "note.md"], check=True)


async def retrieve() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        chunks = await rag.retrieve("What makes sourdough rise?", top_k=1)
    assert chunks, "retrieve returned no chunk"
    assert "yeast" in chunks[0].text, chunks[0].text
    assert chunks[0].dense_rank == 1, chunks[0]
    print(f"retrieved: {' '.join(chunks[0].text.split())[:90]!r}")


asyncio.run(retrieve())
"""

_SCRUBBED_VARIABLES: Final = frozenset({"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"})


def clean_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment with no import path override and no credential in it.

    Without this, a gate run as `PYTHONPATH=src` imports the checkout inside the clean venv.
    """
    source = dict(os.environ if environ is None else environ)
    return {
        name: value
        for name, value in source.items()
        if name.upper() not in _SCRUBBED_VARIABLES
        and not name.upper().endswith(("_API_KEY", "_DSN"))
        and not name.upper().startswith("FASTERRAG_")
    }


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command in the clean environment, capturing text, without raising on failure."""
    return subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=cwd, env=clean_environment()
    )


def _python_in(environment: Path) -> Path:
    """Return the interpreter inside a created virtual environment."""
    # CRITICAL: a conditional expression, not an if/return pair. mypy narrows sys.platform
    # per platform, so the second `return` of an if/return would be unreachable code on
    # Windows and `warn_unreachable` makes that an error — the trap registered as TASK-0263,
    # which kept CI red on Linux for a month while passing on every developer machine.
    leaf = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    return environment.joinpath(*leaf)


def _console_in(interpreter: Path) -> Path:
    """Return the ``fasterrag`` console script beside an environment's interpreter."""
    return interpreter.parent / ("fasterrag.exe" if sys.platform == "win32" else "fasterrag")


def _say(message: str) -> None:
    """Report a step on stdout, so a slow install is visibly progressing."""
    sys.stdout.write(f"check_package: {message}\n")
    sys.stdout.flush()


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


def _check_absent(interpreter: Path, modules: tuple[str, ...], install: str) -> list[str]:
    """Report each of ``modules`` that imports in ``interpreter``'s environment."""
    problems = []
    for module in modules:
        found = _run([str(interpreter), "-c", f"import {module}"])
        if found.returncode == 0:
            problems.append(
                f"{module!r} imports from {install}; it belongs to an extra, and a user "
                "who asked for neither should not be paying for it"
            )
    return problems


def _check_core_stays_small(interpreter: Path) -> list[str]:
    """Confirm a core install did not pull the optional heavy stack."""
    return _check_absent(interpreter, EXTRA_ONLY_MODULES, "a core install")


def _check_named_extras(interpreter: Path) -> list[str]:
    """Confirm that what a missing extra breaks says which extra to install."""
    problems = []
    for need, statement, hint in NAMED_EXTRAS:
        ran = _run([str(interpreter), "-c", statement])
        if ran.returncode != 0 or hint not in ran.stdout:
            problems.append(
                f"{need} without its extra did not name {hint!r}:\n"
                f"{ran.stdout.strip()}\n{ran.stderr.strip()}"
            )
    return problems


def _check_cli(interpreter: Path) -> list[str]:
    """Confirm the console script is installed and runs."""
    executable = _console_in(interpreter)
    if not executable.exists():
        return [f"the fasterrag console script was not installed at {executable}"]

    helped = _run([str(executable), "--help"])
    if helped.returncode != 0:
        return [f"`fasterrag --help` failed:\n{helped.stderr.strip()}"]
    return []


def _check_serve_names_its_extra(interpreter: Path, directory: Path) -> list[str]:
    """Confirm ``serve`` without the server extra exits 2 naming it, not with a traceback."""
    directory.mkdir(parents=True, exist_ok=True)
    console = str(_console_in(interpreter))
    _run([console, "config", "init", "--profile", "local"], cwd=directory)
    served = _run([console, "serve"], cwd=directory)
    if served.returncode != 2 or 'pip install "fasterrag[server]"' not in served.stderr:
        return [
            "`fasterrag serve` without the server extra did not exit 2 naming it "
            f"(exit {served.returncode}):\n{served.stdout.strip()}\n{served.stderr.strip()}"
        ]
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


def _check_local_profile(
    interpreter: Path, wheel: Path, directory: Path, torch_index: str
) -> list[str]:
    """Add the local profile's extras, then ingest a Markdown file and retrieve from it."""
    command = [str(interpreter), "-m", "pip", "install", "--quiet"]
    if torch_index:
        command += ["--extra-index-url", torch_index]
    started = time.monotonic()
    installed = _run([*command, f"{wheel}[{LOCAL_PROFILE_EXTRAS}]"])
    if installed.returncode != 0:
        return [f"the local profile's extras did not install:\n{installed.stderr}"]
    _say(f"installed [{LOCAL_PROFILE_EXTRAS}] in {time.monotonic() - started:.0f}s")

    problems = _check_absent(interpreter, LOCAL_PROFILE_ABSENT, "the local profile's install")

    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ran = _run(
        [str(interpreter), "-c", LOCAL_PROFILE_SCRIPT, str(_console_in(interpreter))],
        cwd=directory,
    )
    if ran.returncode != 0:
        return [
            *problems,
            f"the local profile did not ingest and retrieve:\n{ran.stdout}\n{ran.stderr}",
        ]
    for line in ran.stdout.splitlines():
        if line.startswith(("retrieved:", "indexed", "status")):
            _say(f"local profile: {line.strip()}")
    _say(f"local profile ran in {time.monotonic() - started:.0f}s with no service and no key")
    return problems


def check(
    keep: bool = False, *, local_profile: bool = True, torch_index: str = TORCH_CPU_INDEX
) -> list[str]:
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
        problems.extend(_check_named_extras(interpreter))
        problems.extend(_check_serve_names_its_extra(interpreter, workspace / "serve"))
        _say(f"core install checked: {len(problems)} problem(s)")
        if local_profile and not problems:
            problems.extend(
                _check_local_profile(interpreter, wheel, workspace / "local", torch_index)
            )
        return problems
    finally:
        if keep:
            _say(f"kept {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Report problems on stderr; exit 0 when the package is sound."""
    parser = argparse.ArgumentParser(description="Verify the built package installs and works.")
    parser.add_argument("--keep", action="store_true", help="leave the build and venv in place")
    parser.add_argument(
        "--skip-local-profile",
        action="store_true",
        help="check only the core install; skip installing the local profile's extras",
    )
    parser.add_argument(
        "--torch-index",
        default=TORCH_CPU_INDEX,
        help="extra index for the CPU-only torch wheel; an empty string disables it",
    )
    arguments = parser.parse_args(argv)

    problems = check(
        keep=arguments.keep,
        local_profile=not arguments.skip_local_profile,
        torch_index=arguments.torch_index,
    )
    for problem in problems:
        sys.stderr.write(f"check_package: {problem}\n")
    if not problems:
        scope = "" if arguments.skip_local_profile else "; the local profile ingests and retrieves"
        sys.stdout.write(
            "package ok: wheel builds, installs clean, every documented import works, "
            f"missing extras are named{scope}\n"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
