"""The Python examples in README.md, quickstart.md, python-api.md and the site must run.

Nothing executed a document's code, which is how two examples came to call a facade method
that never existed (TASK-0435); the published site's pages (``userdocs/*.md``) then sat
outside the test that closed it (TASK-0499, TASK-0552). Every fenced ``python`` block in
:data:`DOCUMENTS` is run here, as a script (``__name__ == "__main__"``), from a directory
holding the files the examples name and a ``config.yaml`` that selects in-memory providers
by entry point (``tests/support/doc_examples.py``). No fasterRag code is replaced: the
facade, the loader, the pipeline and its parse workers are the real ones.

**Illustrative blocks.** A block that cannot run as written — it uses names it never
defines, or a file the reader is meant to supply — carries the marker
``<!-- doc-example: illustrative -->`` as the last non-blank line before its fence. The
comment renders as nothing on GitHub and in MkDocs. The marker is the only way out, and a
block carrying it must still compile, so a syntax error is caught either way; an illustrative
fragment may use a top-level ``await``, the form a reader pastes into an asyncio REPL.

**Indented fences.** The site's tabbed blocks (``=== "Python"``) indent their fence. Such a
block is extracted and dedented, so the example a site reader copies is the one that runs.
"""

from __future__ import annotations

import ast
import asyncio
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.facade import FasterRag
from tests.support import doc_examples
from tests.support.plugins import install_distribution

ROOT: Final = Path(__file__).resolve().parents[3]
REPOSITORY_DOCUMENTS: Final = ("README.md", "docs/quickstart.md", "docs/python-api.md")
SITE_DOCUMENTS: Final = tuple(
    path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "userdocs").glob("*.md"))
)
DOCUMENTS: Final = (*REPOSITORY_DOCUMENTS, *SITE_DOCUMENTS)
MARKER: Final = "<!-- doc-example: illustrative -->"
PLUGIN: Final = "tests.support"

_FENCE_OPEN: Final = re.compile(r"^(?P<indent>[ \t]*)```python\s*$")

AWAITING_MARKER: Final[dict[tuple[str, str], str]] = {}
"""Unmarked blocks known not to run, keyed by document and the start of the block's first line.

Each is an expected failure until its document gains the marker. An entry that no longer names
an unmarked block fails ``test_every_awaiting_marker_entry_is_still_needed``.
"""

CONFIG: Final = """\
vector_db:
  provider: shared
  api_key_env: null
embeddings:
  provider: acme
  model: acme-embed
llm:
  provider: acme
  model: acme-chat
  api_key_env: null
retrieval:
  rerank: false
"""

CORPUS: Final = {
    "notes.md": "# Notes\n\nX is the codename of the warranty project; it ships in May.\n",
    "my-docs/vendor-agreement.md": (
        "# Vendor agreement\n\nEither party may terminate with thirty days' notice.\n"
    ),
    "my-docs/payment.md": "# Payment terms\n\nInvoices are payable within 45 days.\n",
    "docs/handbook.md": "# Handbook\n\nRetries back off exponentially, at most three.\n",
    "docs/spec.md": "# Spec\n\nThe spec says retries use jittered exponential backoff.\n",
}


@dataclass(frozen=True)
class Block:
    """One fenced ``python`` block of a document."""

    document: str
    line: int
    code: str
    illustrative: bool

    @property
    def name(self) -> str:
        """Return ``document:line``, the id a failure is reported under."""
        return f"{self.document}:{self.line}"

    @property
    def key(self) -> tuple[str, str]:
        """Return the document and the start of the first line, the :data:`AWAITING_MARKER` key."""
        return self.document, self.code.lstrip().splitlines()[0][:28]

    @property
    def awaiting_marker(self) -> str | None:
        """Return why this unmarked block is known not to run, or ``None``."""
        return None if self.illustrative else AWAITING_MARKER.get(self.key)


def extract(document: str, text: str) -> list[Block]:
    """Return every ``python`` block of ``text``, with whether it is marked illustrative."""
    lines = text.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        fence = _FENCE_OPEN.match(lines[index])
        if fence is None:
            index += 1
            continue
        indent = fence.group("indent")
        before = next((line.strip() for line in reversed(lines[:index]) if line.strip()), "")
        end = next(i for i in range(index + 1, len(lines)) if lines[i].startswith(f"{indent}```"))
        code = "\n".join(line.removeprefix(indent) for line in lines[index + 1 : end]) + "\n"
        blocks.append(Block(document, index + 1, code, before == MARKER))
        index = end + 1
    return blocks


def all_blocks() -> list[Block]:
    """Return the blocks of every document this file covers."""
    return [
        block
        for document in DOCUMENTS
        for block in extract(document, (ROOT / document).read_text(encoding="utf-8"))
    ]


BLOCKS: Final = all_blocks()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A deployment directory with the examples' files, the providers and an indexed corpus."""
    install_distribution(
        tmp_path,
        monkeypatch,
        {
            "fasterrag.vectordb": {"shared": f"{PLUGIN}.doc_examples:SharedVectorDB"},
            "fasterrag.embeddings": {"acme": f"{PLUGIN}.acme_plugin:AcmeEmbedder"},
            "fasterrag.llm": {"acme": f"{PLUGIN}.acme_plugin:AcmeLLM"},
        },
    )
    root = tmp_path / "deployment"
    for relative, content in CORPUS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "config.yaml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv("FASTERRAG_API_KEY", "unit-test-control-plane-key")
    doc_examples.reset()
    asyncio.run(_seed(root))
    yield root
    doc_examples.reset()


async def _seed(root: Path) -> None:
    """Index the notes, as the quickstart's ``fasterrag ingest ./notes.md`` step does."""
    async with FasterRag.from_config(root / "config.yaml") as rag:
        job = await rag.ingest([str(root / "notes.md")])
    assert job.status == "completed"


def _exercise(namespace: dict[str, Any]) -> None:
    """Drive what a block only defines: an ASGI app is served one request."""
    app = namespace.get("app")
    if isinstance(app, FastAPI):
        with TestClient(app) as client:
            response = client.get("/answer", params={"q": "What do my notes say about X?"})
        assert response.status_code == 200, response.text
        assert response.json()["answer"]


def test_the_documents_have_examples_to_run() -> None:
    """An extractor that silently found nothing would make every test below vacuous."""
    runnable = {block.document for block in BLOCKS if not block.illustrative}
    assert set(REPOSITORY_DOCUMENTS) <= runnable
    assert {"userdocs/index.md", "userdocs/python.md", "userdocs/quickstart.md"} <= runnable


def test_every_python_fence_yields_a_block() -> None:
    """The site's tabbed blocks are indented, and a column-0 extractor silently missed them."""
    for document in DOCUMENTS:
        lines = (ROOT / document).read_text(encoding="utf-8").splitlines()
        fences = sum(1 for line in lines if line.strip() == "```python")
        assert fences == sum(1 for block in BLOCKS if block.document == document), document


def test_every_awaiting_marker_entry_is_still_needed() -> None:
    """Once a document carries the marker, its entry must go, or the table hides nothing."""
    assert {block.key for block in BLOCKS if block.awaiting_marker} == set(AWAITING_MARKER)


def _params(blocks: list[Block]) -> list[Any]:
    """Wrap each block as a parameter, expecting failure from one still awaiting its marker."""
    return [
        pytest.param(
            block,
            id=block.name,
            marks=[pytest.mark.xfail(reason=block.awaiting_marker, strict=True)]
            if block.awaiting_marker
            else [],
        )
        for block in blocks
    ]


@pytest.mark.parametrize("block", _params(BLOCKS))
def test_every_python_block_compiles(block: Block) -> None:
    """Even an illustrative block must be valid Python."""
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT if block.illustrative else 0
    compile(block.code, block.name, "exec", flags=flags)


@pytest.mark.parametrize("block", _params([block for block in BLOCKS if not block.illustrative]))
def test_every_runnable_python_block_runs(block: Block, workspace: Path) -> None:
    """The defect: a documented example could call a method that did not exist (TASK-0435)."""
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(block.code, block.name, "exec"), namespace)
    _exercise(namespace)


def test_the_marker_must_sit_directly_before_a_fence() -> None:
    """Pins the extractor: a marker separated from its fence by prose marks nothing."""
    text = f"{MARKER}\n\n```python\nx = 1\n```\n{MARKER}\nprose\n```python\ny = 2\n```\n"
    assert [block.illustrative for block in extract("doc.md", text)] == [True, False]


def test_an_indented_fence_is_extracted_and_dedented() -> None:
    """Pins the extractor on the site's tab syntax, where the fence sits four spaces in."""
    text = '=== "Python"\n\n    ```python\n    if True:\n        x = 1\n    ```\n'
    [block] = extract("doc.md", text)
    assert block.code == "if True:\n    x = 1\n"
