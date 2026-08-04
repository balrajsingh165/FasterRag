"""Directory expansion. Passing a directory used to produce one unreadable document."""

from pathlib import Path

from fasterrag.cli.sources import expand_sources


def corpus(root: Path) -> Path:
    (root / "a.md").write_text("first", encoding="utf-8")
    (root / "b.md").write_text("second", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("third", encoding="utf-8")
    return root


def names(paths: list[str]) -> list[str]:
    return sorted(Path(path).name for path in paths)


def test_a_directory_becomes_its_files(tmp_path: Path) -> None:
    assert names(expand_sources([str(corpus(tmp_path))])) == ["a.md", "b.md"]


def test_recursive_descends(tmp_path: Path) -> None:
    expanded = expand_sources([str(corpus(tmp_path))], recursive=True)

    assert names(expanded) == ["a.md", "b.md", "c.md"]


def test_without_recursive_a_subdirectory_is_left_alone(tmp_path: Path) -> None:
    """Otherwise --recursive would mean nothing, which is what it meant before."""
    expanded = expand_sources([str(corpus(tmp_path))])

    assert "c.md" not in names(expanded)


def test_a_file_passes_through(tmp_path: Path) -> None:
    target = tmp_path / "one.md"
    target.write_text("body", encoding="utf-8")

    assert expand_sources([str(target)]) == [str(target)]


def test_a_url_passes_through() -> None:
    """Expansion must not touch anything the filesystem does not own."""
    assert expand_sources(["https://example.com/policy.pdf"]) == ["https://example.com/policy.pdf"]


def test_a_missing_path_passes_through_to_be_reported(tmp_path: Path) -> None:
    """The pipeline reports it as unreadable with a reason code; swallowing it here hides it."""
    missing = str(tmp_path / "absent.md")

    assert expand_sources([missing]) == [missing]


def test_hidden_files_are_skipped(tmp_path: Path) -> None:
    """A directory argument asks for a corpus, and .env is emphatically not part of one."""
    root = corpus(tmp_path)
    (root / ".env").write_text("KEY=value", encoding="utf-8")

    assert ".env" not in names(expand_sources([str(root)], recursive=True))


def test_junk_directories_are_skipped(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    junk = root / "node_modules"
    junk.mkdir()
    (junk / "index.js").write_text("noise", encoding="utf-8")

    assert "index.js" not in names(expand_sources([str(root)], recursive=True))


def test_the_order_is_deterministic(tmp_path: Path) -> None:
    """Document ids and job order derive from this list; an arbitrary order re-ids a corpus."""
    root = corpus(tmp_path)

    assert expand_sources([str(root)], recursive=True) == expand_sources(
        [str(root)], recursive=True
    )


def test_several_sources_keep_their_given_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        (directory / "doc.md").write_text("body", encoding="utf-8")

    expanded = expand_sources([str(second), str(first)])

    assert expanded[0].startswith(str(second))


def test_an_empty_directory_expands_to_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    assert expand_sources([str(empty)]) == []
