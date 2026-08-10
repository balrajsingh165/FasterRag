"""`POST /v1/estimate` and the switch that is supposed to turn it off.

`cost.estimator` was declared, documented as enabling this endpoint, and read by nothing
(TASK-0200): the estimate ran whatever the setting said. The CLI half of the same defect is
covered in `tests/unit/cli`, and both surfaces are asserted because a gate one control plane
honours and the other ignores is the shape the original bug had.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode
from tests.unit.api.conftest import StubVectorDB


def client(**overrides: object) -> TestClient:
    application = create_app(Settings.model_validate(dict(overrides)))
    application.state.vector_db = StubVectorDB()
    return TestClient(application, raise_server_exceptions=False)


def corpus(root: Path) -> list[str]:
    document = root / "note.txt"
    document.write_text("Either party may terminate on thirty days notice.", encoding="utf-8")
    return [str(document)]


def test_an_enabled_estimator_reports_the_corpus(tmp_path: Path) -> None:
    response = client().post("/v1/estimate", json={"sources": corpus(tmp_path)})

    assert response.status_code == 200
    assert response.json()["documents"] == 1
    assert response.json()["tokens"] > 0


def test_a_disabled_estimator_is_refused_rather_than_answered(tmp_path: Path) -> None:
    """A disabled endpoint answering 200 with zeroes would read as a corpus that is free."""
    response = client(cost={"estimator": False}).post(
        "/v1/estimate", json={"sources": corpus(tmp_path)}
    )

    assert response.status_code == 422
    assert response.json()["code"] == ErrorCode.VALIDATION_FAILED.value


def test_the_refusal_names_the_setting_that_caused_it(tmp_path: Path) -> None:
    """Without the name, an operator cannot tell which switch produced the 422."""
    response = client(cost={"estimator": False}).post(
        "/v1/estimate", json={"sources": corpus(tmp_path)}
    )

    assert "cost.estimator" in response.json()["detail"]


def test_a_disabled_estimator_parses_nothing(tmp_path: Path) -> None:
    """The refusal comes before the work, so an absent path is still refused for the switch."""
    response = client(cost={"estimator": False}).post(
        "/v1/estimate", json={"sources": [str(tmp_path / "absent.txt")]}
    )

    assert response.status_code == 422
    assert "cost.estimator" in response.json()["detail"]
