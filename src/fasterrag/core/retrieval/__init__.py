"""Retrieval: the dense and sparse legs, fusion, and filter push-down.

The sparse leg's term encoding lives here so every backend sees identical terms; inverse
document frequency and storage are delegated to the backend that holds the corpus
(``docs/adr/ADR-0007``). Fusion also lives here rather than in a backend, so the configured
``retrieval.rrf_k`` is the constant that actually applies.
"""

from fasterrag.core.retrieval.bm25 import (
    SparseVector,
    encode_document,
    encode_query,
    tokenize,
)
from fasterrag.core.retrieval.fusion import DEFAULT_RRF_K, FusedResult, Ranking, rrf_fuse
from fasterrag.core.retrieval.models import DENSE_LEG, SPARSE_LEG, ScoredChunk

__all__ = [
    "DEFAULT_RRF_K",
    "DENSE_LEG",
    "SPARSE_LEG",
    "FusedResult",
    "Ranking",
    "ScoredChunk",
    "SparseVector",
    "encode_document",
    "encode_query",
    "rrf_fuse",
    "tokenize",
]
