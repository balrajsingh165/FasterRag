from pathlib import Path

from check_openapi_drift import check, documented_routes, served_routes


def write(tmp_path: Path, body: str) -> Path:
    reference = tmp_path / "api-reference.md"
    reference.write_text(body, encoding="utf-8")
    return reference


def test_this_repository_has_no_drift() -> None:
    """The gate runs against the real API in CI; it must be green here first."""
    assert check() == []


def test_the_served_set_is_read_from_the_real_application() -> None:
    routes = served_routes()

    assert "GET /healthz" in routes
    assert "POST /v1/query" in routes


def test_a_route_the_reference_omits_is_reported(tmp_path: Path) -> None:
    drift = check(write(tmp_path, "| `GET /healthz` | liveness |\n"))

    assert any("served but undocumented" in entry for entry in drift)


def test_a_documented_route_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    drift = check(write(tmp_path, "| `POST /v1/invented` | nothing serves this |\n"))

    assert any("POST /v1/invented" in entry for entry in drift)


def test_an_endpoint_marked_unbuilt_is_not_drift(tmp_path: Path) -> None:
    """A specification legitimately describes endpoints ahead of their implementation."""
    reference = write(
        tmp_path, "| `POST /v1/later` | **Not yet implemented.** Coming in a later slice. |\n"
    )

    assert not any("POST /v1/later" in entry for entry in check(reference))


def test_a_stale_unbuilt_marker_is_reported(tmp_path: Path) -> None:
    """The reverse drift: it got built and the note was left behind."""
    reference = write(tmp_path, "| `GET /healthz` | **Not yet implemented.** |\n")

    drift = check(reference)

    assert any("marked 'not yet implemented' but served" in entry for entry in drift)


def test_parameter_names_are_not_treated_as_differences(tmp_path: Path) -> None:
    """Both documents name parameters independently and both spellings are correct."""
    shipped, _ = documented_routes(write(tmp_path, "| `GET /v1/ingest/{anything}` | x |\n"))

    assert "GET /v1/ingest/{}" in shipped


def test_the_schema_endpoint_is_not_required_to_document_itself() -> None:
    """Documenting /openapi.json inside the document it describes is circular."""
    assert not any("openapi.json" in entry for entry in check())
