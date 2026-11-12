"""In-memory providers that the documents' code examples run against (TASK-0499).

The examples open ``FasterRag.from_config("config.yaml")`` several times — the quickstart
retrieves from an index a separate ``fasterrag ingest`` built — so the store here is shared
by every instance in the process, and :func:`reset` empties it between tests. Everything
else is the acme plugin, reached by name through an entry point like any installed package.
"""

from __future__ import annotations

from typing import ClassVar

from fasterrag.adapters.vectordb.base import CollectionSpec, Point
from fasterrag.config.schema import Settings
from tests.support.acme_plugin import AcmeVectorDB

__all__ = ["SharedVectorDB", "reset"]


class SharedVectorDB(AcmeVectorDB):
    """The acme store, with one set of collections for every instance in the process."""

    specs_shared: ClassVar[dict[str, CollectionSpec]] = {}
    points_shared: ClassVar[dict[str, dict[str, Point]]] = {}
    aliases_shared: ClassVar[dict[str, str]] = {}
    snapshots_shared: ClassVar[dict[str, dict[str, dict[str, Point]]]] = {}

    def __init__(self, settings: Settings) -> None:
        """Point this instance at the process-wide collections."""
        super().__init__(settings)
        self.specs = self.specs_shared
        self.points = self.points_shared
        self.aliases = self.aliases_shared
        self.snapshots = self.snapshots_shared


def reset() -> None:
    """Forget every collection, so no test reads another's index."""
    SharedVectorDB.specs_shared.clear()
    SharedVectorDB.points_shared.clear()
    SharedVectorDB.aliases_shared.clear()
    SharedVectorDB.snapshots_shared.clear()
