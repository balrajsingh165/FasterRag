from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api import admin
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.services.archive_import import ArchiveReader, open_archive
from tests.unit.api.conftest import StubVectorDB


class IterableStub(StubVectorDB):
    """A stub that also serves points, so an export has something to write."""

    def __init__(self, points: list[Any] | None = None) -> None:
        super().__init__()
        self._points = points or []
        self.written: list[Any] = []
        self.created: list[Any] = []

    async def create_collection(self, spec: Any) -> None:
        self.created.append(spec)

    async def upsert(self, points: list[Any]) -> Any:
        from fasterrag.adapters.vectordb.base import UpsertResult

        self.written.extend(points)
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> Any:
        for point in self._points:
            yield point


def point(chunk_id: str) -> Any:
    from fasterrag.adapters.vectordb.base import Point

    return Point(
        point_id=chunk_id,
        collection="docs",
        vector=[0.1, 0.2],
        payload={
            "document_id": "d_1",
            "source_uri": "a.md",
            "content_hash": "h",
            "text": "body",
            "span": {"start": 0, "end": 4},
        },
    )


def client(points: list[Any] | None = None) -> TestClient:
    app = create_app(Settings.model_validate({}))
    app.state.vector_db = IterableStub(points)
    return TestClient(app)


def test_export_writes_an_archive(tmp_path: Path) -> None:
    out = tmp_path / "backup"

    with client([point("c_1")]) as api:
        response = api.post("/v1/admin/export", json={"out": str(out)})

    assert response.status_code == 200
    assert response.json()["chunks"] == 1
    assert out.with_suffix(".fragx").is_file()


def test_export_reports_the_counts(tmp_path: Path) -> None:
    with client([point("c_1"), point("c_2")]) as api:
        body = api.post(
            "/v1/admin/export", json={"out": str(tmp_path / "b"), "include_vectors": True}
        ).json()

    assert body["chunks"] == 2
    assert body["vectors"] == 2


def test_exporting_an_empty_collection_is_refused(tmp_path: Path) -> None:
    """An empty archive imports cleanly and silently produces an empty collection."""
    with client([]) as api:
        response = api.post("/v1/admin/export", json={"out": str(tmp_path / "b")})

    assert response.status_code == 404


def test_a_misspelled_field_is_rejected(tmp_path: Path) -> None:
    with client([point("c_1")]) as api:
        response = api.post(
            "/v1/admin/export", json={"out": str(tmp_path / "b"), "include_vector": True}
        )

    assert response.status_code == 422


def test_importing_a_missing_archive_is_a_usage_error(tmp_path: Path) -> None:
    with client() as api:
        response = api.post("/v1/admin/import", json={"archive": str(tmp_path / "absent.fragx")})

    assert response.status_code == 422


def test_the_round_trip_runs_over_rest(tmp_path: Path) -> None:
    """Export then import through the API, with no CLI or service call in between."""
    out = tmp_path / "roundtrip"

    with client([point("c_1"), point("c_2")]) as api:
        exported = api.post(
            "/v1/admin/export", json={"out": str(out), "include_vectors": True}
        ).json()
        response = api.post(
            "/v1/admin/import",
            json={"archive": str(out.with_suffix(".fragx")), "target_collection": "restored"},
        )
        imported = response.json()

    assert response.status_code == 200, imported
    assert exported["chunks"] == imported["chunks"] == 2
    assert imported["collection"] == "restored"
    assert imported["reembed"] is False


@pytest.fixture
def staged(monkeypatch: pytest.MonkeyPatch) -> list[ArchiveReader]:
    """Keep every reader the endpoint opened, so its staging directory can be checked."""
    readers: list[ArchiveReader] = []

    def spy(path: Path) -> ArchiveReader:
        reader = open_archive(path)
        readers.append(reader)
        return reader

    monkeypatch.setattr(admin, "open_archive", spy)
    return readers


def test_a_successful_import_releases_the_staging_directory(
    tmp_path: Path, staged: list[ArchiveReader]
) -> None:
    """Relying on the finalizer left 61 abandoned directories behind (TASK-0301)."""
    out = tmp_path / "released"

    with client([point("c_1")]) as api:
        api.post("/v1/admin/export", json={"out": str(out), "include_vectors": True})
        response = api.post("/v1/admin/import", json={"archive": str(out.with_suffix(".fragx"))})

    assert response.status_code == 200, response.json()
    assert [reader.staging.exists() for reader in staged] == [False]


def test_a_refused_import_releases_the_staging_directory(
    tmp_path: Path, staged: list[ArchiveReader]
) -> None:
    """A vectorless archive asked for as a vector copy is refused after it has been staged."""
    out = tmp_path / "novectors"

    with client([point("c_1")]) as api:
        api.post("/v1/admin/export", json={"out": str(out)})
        response = api.post("/v1/admin/import", json={"archive": str(out.with_suffix(".fragx"))})

    assert response.status_code == 409, response.json()
    assert [reader.staging.exists() for reader in staged] == [False]


def confined(roots: list[Path], points: list[Any] | None = None) -> TestClient:
    """A deployment that has named the directories its archives live in."""
    app = create_app(
        Settings.model_validate(
            {"security": {"allowed_archive_roots": [str(root) for root in roots]}}
        )
    )
    app.state.vector_db = IterableStub(points)
    return TestClient(app)


def test_exporting_outside_the_allowed_roots_is_refused(tmp_path: Path) -> None:
    """The destination arrives in the request body, so it is a write primitive."""
    allowed = tmp_path / "archives"
    allowed.mkdir()

    with confined([allowed], [point("c_1")]) as api:
        response = api.post("/v1/admin/export", json={"out": str(tmp_path / "config")})

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"
    assert not (tmp_path / "config.fragx").exists()


def test_exporting_inside_an_allowed_root_still_works(tmp_path: Path) -> None:
    """Control case: confinement must not break the feature it protects."""
    allowed = tmp_path / "archives"
    allowed.mkdir()

    with confined([allowed], [point("c_1")]) as api:
        response = api.post("/v1/admin/export", json={"out": str(allowed / "backup")})

    assert response.status_code == 200
    assert (allowed / "backup.fragx").is_file()


def test_importing_from_outside_the_allowed_roots_is_refused(tmp_path: Path) -> None:
    """An archive anywhere on disk is readable without this."""
    allowed = tmp_path / "archives"
    allowed.mkdir()
    outside = tmp_path / "elsewhere.fragx"
    outside.write_bytes(b"")

    with confined([allowed]) as api:
        response = api.post("/v1/admin/import", json={"archive": str(outside)})

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"
    assert "allowed_archive_roots" in response.json()["detail"]


SECRET = "acme-key"


def tenant_client(
    monkeypatch: pytest.MonkeyPatch, root: Path, points: list[Any] | None = None
) -> TestClient:
    """A multi-tenant deployment whose one key may act for `acme` only."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET}:admin:acme")
    app = create_app(
        Settings.model_validate(
            {
                "security": {
                    "auth": True,
                    "multi_tenancy": True,
                    "allowed_archive_roots": [str(root)],
                }
            }
        )
    )
    app.state.vector_db = IterableStub(points)
    return TestClient(app)


def tenant_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SECRET}", "X-Tenant-ID": "acme"}


def test_a_tenant_export_records_the_name_the_tenant_gave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-0340: the backend name is `acme__docs`, and the archive must not say so.

    `collection.name` is specified as the source collection name, and a tenant never
    chose the prefix; recording it makes import scope an already-scoped name.
    """
    out = tmp_path / "tenant"

    with tenant_client(monkeypatch, tmp_path, [point("c_1")]) as api:
        response = api.post(
            "/v1/admin/export",
            json={"out": str(out), "collection": "docs", "include_vectors": True},
            headers=tenant_headers(),
        )

    assert response.status_code == 200, response.json()
    assert response.json()["collection"] == "docs"
    with open_archive(out.with_suffix(".fragx")) as reader:
        assert reader.collection == "docs"
        assert reader.manifest["tenant"] == "acme"


def test_a_tenant_round_trip_lands_in_the_tenants_own_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export and import as a tenant, with no `target_collection` to paper over the name.

    Before TASK-0340 this wrote `acme__acme__docs` and answered with `acme__docs`: a
    backend-prefixed name in a 200 body, which is the TASK-0324 class of leak.
    """
    out = tmp_path / "roundtrip"

    with tenant_client(monkeypatch, tmp_path, [point("c_1"), point("c_2")]) as api:
        api.post(
            "/v1/admin/export",
            json={"out": str(out), "collection": "docs", "include_vectors": True},
            headers=tenant_headers(),
        )
        response = api.post(
            "/v1/admin/import",
            json={"archive": str(out.with_suffix(".fragx"))},
            headers=tenant_headers(),
        )
        app = cast(FastAPI, api.app)
        created = [spec.name for spec in app.state.vector_db.created]

    body = response.json()
    assert response.status_code == 200, body
    assert created == ["acme__docs"]
    assert body["collection"] == "docs"
    assert body["chunks"] == 2
