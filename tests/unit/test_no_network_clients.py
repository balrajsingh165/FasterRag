"""Proves the unit tier's network guard refuses a real vector-db client.

TASK-0353. The guard lives in ``tests/unit/conftest.py`` as an autouse fixture, which makes
it invisible: nothing fails if a future edit deletes it, and the defect it catches showed up
as a stray warning on stderr in 1 of 6 runs rather than as a failure. These cases are the
only thing that can tell the difference between a guard that works and one that is gone.

Nothing here opens a socket — the guard raises inside the constructor, before the client
builds a connection or starts its background version-check thread.
"""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient


def test_building_a_remote_async_client_fails_the_test_that_did_it() -> None:
    with pytest.raises(AssertionError, match="constructed a real AsyncQdrantRemote"):
        AsyncQdrantClient(host="localhost", port=6333)


def test_building_a_remote_sync_client_fails_the_test_that_did_it() -> None:
    with pytest.raises(AssertionError, match="constructed a real QdrantRemote"):
        QdrantClient(host="localhost", port=6333)


def test_the_refusal_names_the_test_and_the_way_out() -> None:
    """A guard that only says 'no' sends the next person to suppress the warning instead."""
    with pytest.raises(AssertionError) as caught:
        AsyncQdrantClient(host="localhost", port=6333)

    message = str(caught.value)
    assert "test_the_refusal_names_the_test_and_the_way_out" in message
    assert "tests/integration/" in message
    assert "TASK-0353" in message


def test_a_local_client_is_left_alone() -> None:
    """In-memory mode speaks to no server, so the guard must not stand in its way."""
    client = AsyncQdrantClient(location=":memory:")

    assert client is not None
