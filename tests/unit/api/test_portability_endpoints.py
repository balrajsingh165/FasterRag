from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
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
