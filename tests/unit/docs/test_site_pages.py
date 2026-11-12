"""The published documentation site stays small, user-facing and reachable.

The site at docs-fasterrag.fortiqo.xyz is built from ``userdocs/``, not from ``docs/``: the
maintainer asked for a short site carrying only what a developer using fasterRag needs, while
``docs/`` stays the engineering record. These tests hold the three things that drift first: a
page added to ``userdocs/`` but left out of the navigation (still published and indexed, just
unreachable), an engineering task id leaking onto the site, and a ``vercel.json`` redirect
pointing at a page that no longer exists.
"""

import json
import re
from pathlib import Path
from typing import Any, Final

import yaml

ROOT: Final = Path(__file__).resolve().parents[3]
SITE: Final = ROOT / "userdocs"
MKDOCS: Final = ROOT / "mkdocs.yml"
VERCEL: Final = ROOT / "vercel.json"
TASK_ID: Final = re.compile(r"\bTASK-\d{4}\b")
PYTHON_TAG: Final = re.compile(r"!!python/name:\S+")
UNPUBLISHABLE: Final = frozenset({"todo.md", "blockers.md", "task-briefs.md", "code-notes.md"})


def _config() -> dict[str, Any]:
    """Return mkdocs.yml as data, reading its ``!!python/name:`` tags as nulls."""
    text = PYTHON_TAG.sub("null", MKDOCS.read_text(encoding="utf-8"))
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def _nav_pages(entries: Any) -> list[str]:
    """Return every page file the navigation names, depth first."""
    pages: list[str] = []
    if isinstance(entries, str):
        return [entries]
    if isinstance(entries, list):
        for entry in entries:
            pages += _nav_pages(entry)
    if isinstance(entries, dict):
        for value in entries.values():
            pages += _nav_pages(value)
    return [page for page in pages if not page.startswith(("http://", "https://"))]


def _site_pages() -> list[str]:
    """Return every markdown page under the site source, relative to it."""
    return sorted(path.relative_to(SITE).as_posix() for path in SITE.rglob("*.md"))


def _page_url(page: str) -> str:
    """Return the URL mkdocs serves a page at, with directory URLs on."""
    stem = page.removesuffix(".md")
    return "/" if stem == "index" else f"/{stem.removesuffix('/index')}/"


def test_the_site_is_built_from_userdocs() -> None:
    """Building from docs/ would publish the task ledger and every engineering document."""
    assert _config()["docs_dir"] == "userdocs"
    assert len(_site_pages()) >= 5


def test_every_site_page_is_in_the_navigation() -> None:
    """A page outside the nav is still published and indexed, only unreachable."""
    nav = _nav_pages(_config()["nav"])
    assert sorted(nav) == _site_pages()


def test_no_unpublishable_document_is_in_the_site_source() -> None:
    """todo.md and blockers.md describe open security findings and must never be published."""
    names = {Path(page).name for page in _site_pages()}
    assert not names & UNPUBLISHABLE


def test_no_site_page_mentions_a_task_id() -> None:
    """Task ids are engineering bookkeeping; the site is for people using fasterRag."""
    leaks = [
        f"{page}: {match.group(0)}"
        for page in _site_pages()
        for match in TASK_ID.finditer((SITE / page).read_text(encoding="utf-8"))
    ]
    assert leaks == []


def test_every_site_page_links_to_its_full_reference_or_is_the_home_page() -> None:
    """Each short page points at the GitHub document that carries the detail."""
    missing = [
        page
        for page in _site_pages()
        if page != "index.md"
        and "github.com/balrajsingh165/fasterrag/blob/main/docs/"
        not in (SITE / page).read_text(encoding="utf-8")
    ]
    assert missing == []


def test_redirects_land_on_live_pages_and_never_shadow_one() -> None:
    """A bookmarked URL from the old site must reach a page, and no live page may be redirected."""
    live = {_page_url(page) for page in _site_pages()}
    redirects = json.loads(VERCEL.read_text(encoding="utf-8"))["redirects"]
    assert redirects
    for redirect in redirects:
        destination, source = redirect["destination"], redirect["source"]
        if destination.startswith("/"):
            assert destination.split("#")[0] in live, redirect
        else:
            assert destination.startswith("https://github.com/balrajsingh165/fasterrag/"), redirect
        assert source.rstrip("/") + "/" not in live, redirect
