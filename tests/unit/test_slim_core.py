"""A core install carries no web server, no Qdrant client and no parser library (TASK-0444).

Every test here runs a child interpreter in which each package an extra provides is made
unimportable, which is what ``pip install fasterrag`` without extras looks like. The
in-process suite cannot show it, because the development environment installs every extra.
The wheel itself is proven the same way, for real, by ``scripts/check_package.py``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import fasterrag
from fasterrag.core.parsing import parse_bytes
from fasterrag.errors import ErrorCode, ParseError

SOURCE_TREE = Path(fasterrag.__file__).resolve().parents[1]

EXTRA_PACKAGES = (
    "fastapi",
    "starlette",
    "uvicorn",
    "qdrant_client",
    "grpc",
    "pdfplumber",
    "docx",
    "bs4",
    "lxml",
)


def without_extras(body: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a child interpreter where no extra's package can be imported."""
    blocker = f"import sys\nfor blocked in {EXTRA_PACKAGES!r}:\n    sys.modules[blocked] = None\n"
    return subprocess.run(
        [sys.executable, "-c", blocker + body],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SOURCE_TREE)},
        timeout=180,
        check=False,
    )


def test_the_library_cli_and_markdown_parsing_need_no_extra(tmp_path: Path) -> None:
    """The CLI imported the web framework for every command, and parsing every parser."""
    child = without_extras(
        "import fasterrag.cli.main, fasterrag.parsing, fasterrag.services.querying\n"
        "from fasterrag import FasterRag\n"
        "from fasterrag.parsing import parse_bytes\n"
        "document = parse_bytes(b'# Title\\n\\nBody text.\\n', filename='note.md')\n"
        "assert 'Body text.' in document.text, document.text\n"
        "loaded = [n for n, m in sys.modules.items() if m and n.startswith('fasterrag.api')]\n"
        "assert not loaded, loaded\n"
        "print('core ok')\n",
        tmp_path,
    )

    assert child.returncode == 0, child.stderr
    assert "core ok" in child.stdout


def test_the_cli_help_runs_without_the_web_server(tmp_path: Path) -> None:
    child = without_extras(
        "from fasterrag.cli.main import main\nraise SystemExit(main(['--help']))\n", tmp_path
    )

    assert child.returncode == 0, child.stderr
    assert "serve" in child.stdout


def test_serve_without_the_server_extra_names_it(tmp_path: Path) -> None:
    """It said nothing about an extra: the CLI could not even start without FastAPI."""
    (tmp_path / "config.yaml").write_text("security:\n  auth: false\n", encoding="utf-8")
    child = without_extras(
        "import os\n"
        "for name in ('OPENAI_API_KEY', 'QDRANT_API_KEY'):\n"
        "    os.environ[name] = 'placeholder'\n"
        "from fasterrag.cli.main import main\n"
        "raise SystemExit(main(['serve', '--config', 'config.yaml']))\n",
        tmp_path,
    )

    assert child.returncode == 2, child.stderr
    assert 'pip install "fasterrag[server]"' in child.stderr


def test_qdrant_without_its_extra_is_a_config_error_naming_it(tmp_path: Path) -> None:
    """The default config selects Qdrant, so a bare install must say which extra it needs."""
    child = without_extras(
        "from fasterrag.adapters.vectordb.factory import create_vector_db_adapter\n"
        "from fasterrag.config.schema import Settings\n"
        "from fasterrag.errors import ConfigError\n"
        "try:\n"
        "    create_vector_db_adapter(Settings())\n"
        "except ConfigError as exc:\n"
        "    print(exc.detail)\n",
        tmp_path,
    )

    assert child.returncode == 0, child.stderr
    assert 'pip install "fasterrag[qdrant]"' in child.stdout


@pytest.mark.parametrize(
    ("filename", "extra"),
    [("report.pdf", "pdf"), ("letter.docx", "docx"), ("page.html", "html")],
)
def test_a_format_whose_extra_is_missing_is_a_parse_error_naming_it(
    filename: str, extra: str, tmp_path: Path
) -> None:
    child = without_extras(
        "from fasterrag.errors import ParseError\n"
        "from fasterrag.parsing import parse_bytes\n"
        "try:\n"
        f"    parse_bytes(b'data', filename={filename!r})\n"
        "except ParseError as exc:\n"
        "    print(exc.code.value, exc.detail)\n",
        tmp_path,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.startswith("PARSE_FAILED")
    assert f'pip install "fasterrag[{extra}]"' in child.stdout


def test_html_without_lxml_names_the_extra_even_when_beautifulsoup_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bs4 alone imports fine and then fails at parse time asking for a tree builder."""
    monkeypatch.setitem(sys.modules, "lxml", None)

    with pytest.raises(ParseError, match=r"fasterrag\[html\]") as caught:
        parse_bytes(b"<p>hi</p>", filename="page.html")

    assert caught.value.code is ErrorCode.PARSE_FAILED
    assert "lxml" in caught.value.detail


def test_an_installed_extra_still_parses() -> None:
    """The lazy wrapper must hand the document to the real parser, not only refuse."""
    document = parse_bytes(b"<h1>Title</h1><p>Body text.</p>", filename="page.html")

    assert "Body text." in document.text
