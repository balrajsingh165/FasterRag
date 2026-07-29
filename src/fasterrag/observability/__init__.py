"""Observability: correlation ids, structured logging, spans, metrics, and the dashboard.

Everything in this package is a one-directional consumer of pipeline events. Nothing in
the pipeline depends on it, and the dashboard it will eventually serve is read-only —
observability never controls the RAG (``docs/adr/ADR-0005``).
"""
