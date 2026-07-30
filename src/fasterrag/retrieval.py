"""Public retrieval surface: ``from fasterrag.retrieval import rrf_fuse``.

The documented standalone components (``docs/python-api.md``). Fusion is usable on its own
over any two rankings, so an application can adopt fasterRag's hybrid combination without
adopting its pipeline.
"""

from fasterrag.core.retrieval.bm25 import encode_document, encode_query, tokenize
from fasterrag.core.retrieval.fusion import DEFAULT_RRF_K, FusedResult, Ranking, rrf_fuse
from fasterrag.core.retrieval.models import ScoredChunk

__all__ = [
    "DEFAULT_RRF_K",
    "FusedResult",
    "Ranking",
    "ScoredChunk",
    "encode_document",
    "encode_query",
    "rrf_fuse",
    "tokenize",
]
