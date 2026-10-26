import datetime as dt
from pathlib import Path

import pytest
from stamp_release import Tree, check, pep440, stamp

DATE = dt.date(2026, 10, 1)


def build(tmp_path: Path, *, version: str = "0.1.0.dev0", banners: bool = True) -> Tree:
    """A minimal tree in the development state: the shape the real checkout is in today."""
    (tmp_path / "src" / "fasterrag").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "fasterrag" / "__init__.py").write_text(
        f'"""pkg."""\n\n__version__ = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "docs" / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- a thing\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nclassifiers = [\n    "Development Status :: 2 - Pre-Alpha",\n]\n',
        encoding="utf-8",
    )
    if banners:
        (tmp_path / "README.md").write_text("> Status: not yet on PyPI.\n", encoding="utf-8")
        (tmp_path / "docs" / "python-api.md").write_text(
            "| Not yet published |\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "how-to-use.md").write_text(
            "(Until the first release lands, install from source.)\n", encoding="utf-8"
        )
    return Tree(tmp_path)


def remove_banners(tree: Tree) -> None:
    (tree.root / "README.md").write_text("> Status: on PyPI.\n", encoding="utf-8")
    (tree.root / "docs" / "python-api.md").write_text("| Published |\n", encoding="utf-8")
    (tree.root / "docs" / "how-to-use.md").write_text("Install it.\n", encoding="utf-8")


def test_this_repository_is_coherent() -> None:
    """The gate runs against the real checkout in CI; it must be green here first."""
    assert check() == []


@pytest.mark.parametrize(
    ("semver", "expected"),
    [
        ("0.1.0-beta.1", "0.1.0b1"),
        ("0.1.0-alpha.2", "0.1.0a2"),
        ("1.0.0-rc.1", "1.0.0rc1"),
        ("1.0.0", "1.0.0"),
    ],
)
def test_the_release_form_converts_to_pep440(semver: str, expected: str) -> None:
    """The tag and CHANGELOG use SemVer; hatch and pip use PEP 440. Both must name one thing."""
    assert pep440(semver) == expected


@pytest.mark.parametrize("bad", ["0.1.0-beta1", "0.1.0b1", "v0.1.0", "0.1", "0.1.0-dev.1"])
def test_anything_outside_the_release_procedure_shape_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="not a release version"):
        pep440(bad)


def test_a_development_tree_is_coherent(tmp_path: Path) -> None:
    assert check(build(tmp_path)) == []


def test_a_finished_stamp_is_coherent(tmp_path: Path) -> None:
    tree = build(tmp_path)

    assert stamp("0.1.0-beta.1", DATE, tree) == "0.1.0b1"
    remove_banners(tree)

    assert check(tree) == []
    assert '__version__ = "0.1.0b1"' in tree.init.read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [0.1.0-beta.1] — 2026-10-01\n\n### Added" in (
        tree.changelog.read_text(encoding="utf-8")
    )
    assert "Development Status :: 4 - Beta" in tree.pyproject.read_text(encoding="utf-8")


def test_the_stamp_keeps_an_empty_unreleased_section_above_the_release(tmp_path: Path) -> None:
    """Keep a Changelog: the next change after a release has somewhere to go."""
    tree = build(tmp_path)
    stamp("0.1.0-beta.1", DATE, tree)

    lines = tree.changelog.read_text(encoding="utf-8").splitlines()

    assert lines.index("## [Unreleased]") < lines.index("## [0.1.0-beta.1] — 2026-10-01")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_the_stamp_preserves_each_files_line_ending(tmp_path: Path, newline: str) -> None:
    """A Windows checkout holds CRLF under autocrlf; the stamp must not rewrite every line."""
    tree = build(tmp_path)
    for path in (tree.init, tree.changelog, tree.pyproject):
        text = path.read_text(encoding="utf-8")
        with path.open("w", encoding="utf-8", newline=newline) as handle:
            handle.write(text)

    stamp("0.1.0-beta.1", DATE, tree)

    for path in (tree.init, tree.changelog, tree.pyproject):
        raw = path.read_bytes()
        assert raw.count(newline.encode()) == raw.count(b"\n"), path.name
    heading = f"## [Unreleased]{newline}{newline}## [0.1.0-beta.1] — 2026-10-01{newline}{newline}"
    assert heading.encode() in tree.changelog.read_bytes()


def test_a_stamp_that_left_the_banners_is_reported_by_file(tmp_path: Path) -> None:
    """The prose is edited by hand; the gate names each document still promising nothing shipped."""
    tree = build(tmp_path)
    stamp("0.1.0-beta.1", DATE, tree)

    drift = check(tree)

    assert len(drift) == 3
    assert any("README.md" in entry and "not yet on PyPI" in entry for entry in drift)
    assert any("python-api.md" in entry for entry in drift)
    assert any("how-to-use.md" in entry for entry in drift)


def test_a_version_bumped_without_the_changelog_is_reported(tmp_path: Path) -> None:
    tree = build(tmp_path, version="0.1.0b1")
    remove_banners(tree)

    drift = check(tree)

    assert any("no release heading" in entry for entry in drift)
    assert any("classifier" in entry for entry in drift)


def test_a_changelog_heading_that_disagrees_with_the_version_is_reported(tmp_path: Path) -> None:
    tree = build(tmp_path)
    stamp("0.1.0-beta.2", DATE, tree)
    remove_banners(tree)
    init = tree.init.read_text(encoding="utf-8").replace("0.1.0b2", "0.1.0b1")
    tree.init.write_text(init, encoding="utf-8")

    drift = check(tree)

    assert any("0.1.0b1" in entry and "[0.1.0-beta.2]" in entry for entry in drift)


def test_an_undated_heading_is_reported(tmp_path: Path) -> None:
    """A release heading without a date is a release nobody can place in time."""
    tree = build(tmp_path)
    stamp("0.1.0-beta.1", DATE, tree)
    remove_banners(tree)
    changelog = tree.changelog.read_text(encoding="utf-8").replace(" — 2026-10-01", "")
    tree.changelog.write_text(changelog, encoding="utf-8")

    assert any("no release date" in entry for entry in check(tree))


def test_a_release_heading_on_a_dev_tree_is_reported(tmp_path: Path) -> None:
    """Stamping the CHANGELOG before the version is the half-done state in the other direction."""
    tree = build(tmp_path)
    changelog = tree.changelog.read_text(encoding="utf-8").replace(
        "## [Unreleased]", "## [Unreleased]\n\n## [0.1.0-beta.1] — 2026-10-01"
    )
    tree.changelog.write_text(changelog, encoding="utf-8")

    assert any("already has a release heading" in entry for entry in check(tree))


def test_a_banner_deleted_early_is_reported(tmp_path: Path) -> None:
    """The banners say the package is unpublished; while it is, they must stay."""
    tree = build(tmp_path)
    (tree.root / "README.md").write_text("> Status: on PyPI.\n", encoding="utf-8")

    drift = check(tree)

    assert any("banner is missing" in entry and "README.md" in entry for entry in drift)


def test_a_missing_unreleased_section_is_reported_in_either_state(tmp_path: Path) -> None:
    tree = build(tmp_path)
    tree.changelog.write_text("# Changelog\n\n### Added\n\n- a thing\n", encoding="utf-8")

    assert any("no [Unreleased]" in entry for entry in check(tree))
    with pytest.raises(ValueError, match="no \\[Unreleased\\]"):
        stamp("0.1.0-beta.1", DATE, tree)
