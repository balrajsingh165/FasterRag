"""`doctor --fix` against a real Docker daemon.

The unit tier proves the engine's decisions against a fake daemon; only this proves the
repairs mean to Docker what the fake assumes. Two things in particular cannot be faked
honestly: whether the created volume is one `docker volume inspect` agrees exists, and
whether the management label fasterRag writes at `docker run` is the one it reads back at
`docker inspect` — the guard that stops `--fix` starting somebody else's container is
worth nothing if those two strings ever disagree.

Everything here works on throwaway names of its own. It never touches
`fasterrag_qdrant_storage` or the managed container, because a test of a diagnostic that
could delete an operator's index would be worse than the defect it guards against.
"""

from collections.abc import AsyncIterator

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.doctor import FixOutcome, apply_fixes, run_doctor
from fasterrag.services.provisioning import (
    CONTAINER_LABEL,
    container_state,
    docker_available,
    run_docker,
    start_container,
    volume_exists,
)

pytestmark = pytest.mark.integration

PROBE_VOLUME = "fasterrag_test_doctor_volume"
PROBE_CONTAINER = "fasterrag-test-doctor-start"
UNLABELLED_CONTAINER = "fasterrag-test-doctor-foreign"


def probe_settings() -> Settings:
    """Return docker-mode settings pointed at the throwaway volume."""
    return Settings.model_validate(
        {"vector_db": {"mode": "docker", "docker": {"volume": PROBE_VOLUME}}}
    )


def attempt_status(outcome: FixOutcome, check: str) -> str:
    return next(item.status for item in outcome.attempts if item.check == check)


@pytest.fixture
async def absent_volume() -> AsyncIterator[None]:
    """Guarantee the throwaway volume does not exist, and remove it again afterwards."""
    if not await docker_available():
        pytest.skip("docker daemon is not available")

    await run_docker(["volume", "rm", "--force", PROBE_VOLUME])
    yield
    await run_docker(["volume", "rm", "--force", PROBE_VOLUME])


@pytest.fixture
async def stopped_containers() -> AsyncIterator[None]:
    """Create one labelled and one unlabelled container, both never started.

    ``docker create`` leaves them stopped without ever running, and neither publishes a
    port, so this cannot collide with anything already on the machine.
    """
    if not await docker_available():
        pytest.skip("docker daemon is not available")

    image = Settings().vector_db.docker.image
    for name, labels in (
        (PROBE_CONTAINER, ["--label", CONTAINER_LABEL]),
        (UNLABELLED_CONTAINER, []),
    ):
        await run_docker(["rm", "--force", name])
        created = await run_docker(["create", "--name", name, *labels, image])
        if not created.ok:
            pytest.skip(f"could not create {name}: {created.stderr}")

    yield

    for name in (PROBE_CONTAINER, UNLABELLED_CONTAINER):
        await run_docker(["rm", "--force", name])


@pytest.mark.usefixtures("absent_volume")
async def test_the_missing_volume_check_reads_the_real_daemon(qdrant: Settings) -> None:
    """A volume Docker has never heard of must fail the check, not pass it."""
    assert await volume_exists(PROBE_VOLUME) is False

    check = next(
        item
        for item in (await run_doctor(probe_settings())).checks
        if item.name == "vector_db_volume"
    )

    assert check.passed is False
    assert check.fix


@pytest.mark.usefixtures("absent_volume")
async def test_fix_creates_a_volume_docker_agrees_exists(qdrant: Settings) -> None:
    outcome = await apply_fixes(probe_settings())

    assert await volume_exists(PROBE_VOLUME) is True
    assert attempt_status(outcome, "vector_db_volume") == "fixed"
    assert "vector_db_volume" in outcome.repaired
    assert "vector_db_volume" not in {check.name for check in outcome.still_failing}


@pytest.mark.usefixtures("absent_volume")
async def test_fixing_twice_changes_nothing_the_second_time(qdrant: Settings) -> None:
    """Converging, not reinstalling: a second run must find nothing left to repair."""
    await apply_fixes(probe_settings())
    outcome = await apply_fixes(probe_settings())

    assert await volume_exists(PROBE_VOLUME) is True
    assert outcome.repaired == []
    assert "vector_db_volume" not in {item.check for item in outcome.attempts}


@pytest.mark.usefixtures("stopped_containers")
async def test_the_label_written_at_run_is_the_one_read_back() -> None:
    """The guard that refuses a foreign container depends on these two strings agreeing."""
    ours = await container_state(PROBE_CONTAINER)
    theirs = await container_state(UNLABELLED_CONTAINER)

    assert (ours.exists, ours.running, ours.managed) == (True, False, True)
    assert (theirs.exists, theirs.running, theirs.managed) == (True, False, False)


@pytest.mark.usefixtures("stopped_containers")
async def test_starting_a_stopped_container_really_starts_it() -> None:
    await start_container(PROBE_CONTAINER)

    assert (await container_state(PROBE_CONTAINER)).running is True
