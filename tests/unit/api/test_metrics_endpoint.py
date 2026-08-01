from fastapi.testclient import TestClient

from fasterrag.api.metrics import METRICS_MEDIA_TYPE
from fasterrag.observability import metrics


def series(text: str, name: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(name) and "{" in line]


def test_the_scrape_endpoint_serves_the_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert METRICS_MEDIA_TYPE.startswith("text/plain")


def test_every_catalogue_metric_is_present_before_any_traffic(client: TestClient) -> None:
    body = client.get("/metrics").text

    for name in metrics.REGISTRY.names:
        assert f"# TYPE {name} " in body


def test_a_request_is_counted(client: TestClient) -> None:
    before = metrics.REQUESTS.value(endpoint="/healthz", method="GET", status="200", tenant="none")

    client.get("/healthz")

    after = metrics.REQUESTS.value(endpoint="/healthz", method="GET", status="200", tenant="none")
    assert after == before + 1


def test_the_endpoint_label_is_the_route_template_not_the_concrete_path(
    client: TestClient,
) -> None:
    client.get("/v1/ingest/job_whatever_id")

    body = client.get("/metrics").text

    assert "job_whatever_id" not in body
    assert "/v1/ingest/{job_id}" in body


def test_request_duration_is_observed(client: TestClient) -> None:
    before = metrics.REQUEST_DURATION.count(endpoint="/healthz")

    client.get("/healthz")

    assert metrics.REQUEST_DURATION.count(endpoint="/healthz") == before + 1


def test_a_problem_response_increments_the_error_counter(client: TestClient) -> None:
    labels = {"endpoint": "/v1/ingest/{job_id}", "code": "NOT_FOUND", "tenant": "none"}
    before = metrics.ERRORS.value(**labels)

    client.get("/v1/ingest/job_absent")

    assert metrics.ERRORS.value(**labels) == before + 1


def test_the_error_label_is_the_template_so_ids_cannot_explode_cardinality(
    client: TestClient,
) -> None:
    client.get("/v1/ingest/job_one")
    client.get("/v1/ingest/job_two")

    body = client.get("/metrics").text

    assert "job_one" not in body
    assert "job_two" not in body
    assert 'fasterrag_errors_total{code="NOT_FOUND",endpoint="/v1/ingest/{job_id}"' in body


def test_a_failed_request_is_still_counted_with_its_status(client: TestClient) -> None:
    before = metrics.REQUESTS.value(
        endpoint="/v1/ingest/{job_id}", method="GET", status="404", tenant="none"
    )

    client.get("/v1/ingest/job_absent")

    after = metrics.REQUESTS.value(
        endpoint="/v1/ingest/{job_id}", method="GET", status="404", tenant="none"
    )
    assert after == before + 1


def test_the_scrape_itself_is_not_authenticated(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 200


def test_the_scrape_is_absent_from_the_openapi_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/metrics" not in schema["paths"]
