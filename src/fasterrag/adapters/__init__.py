"""Vendor isolation boundary.

Every third-party client lives behind an adapter interface, and no vendor type ever
escapes this package: core code imports ``VectorDBAdapter``, never ``qdrant_client``.
That is what makes ``vector_db.provider`` a one-line config change rather than a
refactor (``docs/adr/ADR-0002``).
"""
