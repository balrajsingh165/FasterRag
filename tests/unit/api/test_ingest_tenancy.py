from pathlib import Path

import pytest

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.journal import create_journal


def journal(tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_journal(Settings.model_validate({}), root=tmp_path)


def test_a_job_records_its_tenant(tmp_path: Path) -> None:
    store = journal(tmp_path)

    record = store.create_job("docs", [{"type": "path", "value": "a.md"}], tenant="acme")

    assert record.tenant == "acme"


def test_a_tenant_can_read_its_own_job(tmp_path: Path) -> None:
    store = journal(tmp_path)
    created = store.create_job("docs", [{"type": "path", "value": "a.md"}], tenant="acme")

    assert store.load_job(created.job_id, tenant="acme").job_id == created.job_id


def test_another_tenants_job_is_reported_absent_not_forbidden(tmp_path: Path) -> None:
    """A job record carries the source paths of the corpus it ingested.

    Job ids sort chronologically, so a caller holding one of their own can guess at
    neighbours — a distinct "forbidden" would confirm which guesses are real.
    """
    store = journal(tmp_path)
    created = store.create_job("docs", [{"type": "path", "value": "secret.md"}], tenant="acme")

    with pytest.raises(IngestionError) as caught:
        store.load_job(created.job_id, tenant="globex")

    assert caught.value.code is ErrorCode.NOT_FOUND
    assert "unknown ingest job" in caught.value.detail


def test_the_refusal_matches_an_unknown_id_exactly(tmp_path: Path) -> None:
    """Two different messages would be enough to tell a real id from a guess."""
    store = journal(tmp_path)
    created = store.create_job("docs", [{"type": "path", "value": "a.md"}], tenant="acme")

    with pytest.raises(IngestionError) as other_tenant:
        store.load_job(created.job_id, tenant="globex")
    with pytest.raises(IngestionError) as unknown:
        store.load_job("job_does_not_exist", tenant="globex")

    assert other_tenant.value.code is unknown.value.code


def test_an_operator_reads_any_job(tmp_path: Path) -> None:
    """Passing no tenant is the single-operator deployment, not a tenant with no name."""
    store = journal(tmp_path)
    created = store.create_job("docs", [{"type": "path", "value": "a.md"}], tenant="acme")

    assert store.load_job(created.job_id).job_id == created.job_id


def test_an_untenanted_job_is_hidden_from_a_tenant(tmp_path: Path) -> None:
    store = journal(tmp_path)
    created = store.create_job("docs", [{"type": "path", "value": "a.md"}])

    with pytest.raises(IngestionError):
        store.load_job(created.job_id, tenant="acme")
