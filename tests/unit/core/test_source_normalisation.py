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


def test_a_non_ascii_letter_before_colon_slash_is_no_drive() -> None:
    """`isalpha` is Unicode-wide, so `É:/` and `é:/` folded to one POSIX document (TASK-0304)."""
    assert normalise_source("É:/Docs/A.PDF") == "É:/Docs/A.PDF"
    assert document_id("É:/Docs/A.PDF") != document_id("é:/docs/a.pdf")


@pytest.mark.parametrize(
    ("source", "tenant", "pinned"),
    [
        (r"C:\Docs\A.MD", None, "d_000b4849727b6537"),
        ("d:/docs/a.md", None, "d_04fe2bf8d5017587"),
        ("Z:/x", None, "d_e6471c7a15e56a5d"),
        ("/srv/docs/A.md", None, "d_e46d1eed0ffec501"),
        ("docs/a.md", None, "d_56e17dddd1743075"),
        ("https://x.test/A", None, "d_d92e842d7b84c04c"),
        (r"D:\a.md", "tenant-a", "d_e4e86df494d185da"),
    ],
)
def test_identity_version_three_ids_are_pinned(
    source: str, tenant: str | None, pinned: str
) -> None:
    """Recorded under identity version 3; a change here needs an `IDENTITY_VERSION` bump."""
    assert document_id(source, tenant) == pinned
