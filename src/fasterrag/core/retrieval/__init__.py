"""Retrieval: the dense and sparse legs, fusion, and filter push-down.

The sparse leg's term encoding lives here so every backend sees identical terms; inverse
document frequency and storage are delegated to the backend that holds the corpus
(``docs/adr/ADR-0007``).
"""

from fasterrag.core.retrieval.bm25 import (
    SparseVector,
    encode_document,
    encode_query,
    tokenize,
)

__all__ = ["SparseVector", "encode_document", "encode_query", "tokenize"]
