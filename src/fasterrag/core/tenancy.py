"""The payload field that names a chunk's owning tenant.

One constant, in ``core`` because both sides of the isolation filter need it: the indexer in
``workers/`` writes it and retrieval in ``services/`` filters on it. While it lived in
``services/tenancy.py`` the worker reached up a layer to read it (TASK-0456).
``services/tenancy.py`` re-exports it beside the collection-name scoping it belongs with.
"""

from __future__ import annotations

from typing import Final

__all__ = ["TENANT_PAYLOAD_KEY"]

# CRITICAL: the one spelling of the payload field that carries a chunk's owning tenant.
# `workers.indexer` writes it and `services.querying` filters on it; the two named the same
# string independently until TASK-0339, and a writer and a reader that disagree about a
# field name make an isolation filter that silently matches nothing.
TENANT_PAYLOAD_KEY: Final = "tenant"
