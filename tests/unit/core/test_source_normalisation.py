import pytest

from fasterrag.core.identity import document_id, normalise_source


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (r"d:\docs\a.md", r"D:\docs\a.md"),
        (r"D:\docs\a.md", "D:/docs/a.md"),
        (r"C:\Docs\A.MD", r"c:\docs\a.md"),
    ],
)
def test_two_spellings_of_one_windows_file_are_one_document(first: str, second: str) -> None:
    """Hashing the raw string left a corpus quietly holding several copies of one file."""
    assert document_id(first) == document_id(second)


def test_different_windows_files_stay_different() -> None:
    assert document_id(r"D:\docs\a.md") != document_id(r"D:\docs\b.md")


def test_url_case_is_preserved() -> None:
    """URL paths are case-sensitive by specification; folding merges different resources."""
    assert document_id("https://x.test/A") != document_id("https://x.test/a")


def test_a_url_is_returned_untouched() -> None:
    assert normalise_source("https://Example.test/Spec.PDF") == "https://Example.test/Spec.PDF"


def test_posix_paths_keep_their_case() -> None:
    """Linux filesystems are case-sensitive; two names there are two files."""
    assert document_id("/srv/docs/A.md") != document_id("/srv/docs/a.md")


def test_separators_are_unified_on_posix_paths() -> None:
    assert normalise_source("/srv/docs/a.md") == "/srv/docs/a.md"


def test_the_tenant_still_separates_documents() -> None:
    assert document_id(r"D:\a.md", "tenant-a") != document_id(r"D:\a.md", "tenant-b")
