"""Public reranking surface: ``from fasterrag.rerank import CrossEncoderReranker``.

The documented standalone component (``docs/python-api.md``). An application can rerank its
own candidates without adopting the rest of the pipeline.
"""

from fasterrag.core.rerank import CrossEncoderReranker, Reranker

__all__ = ["CrossEncoderReranker", "Reranker"]
