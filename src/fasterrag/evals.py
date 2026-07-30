"""Public evaluation surface: ``from fasterrag.evals import evaluate``.

The documented standalone component (``docs/python-api.md``). An application can score its
own retriever against its own golden set without adopting the rest of the framework.
"""

from fasterrag.core.evals import (
    EvalReport,
    GoldenRecord,
    QueryScore,
    Retriever,
    evaluate,
    load_golden_set,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    write_golden_set,
)

__all__ = [
    "EvalReport",
    "GoldenRecord",
    "QueryScore",
    "Retriever",
    "evaluate",
    "load_golden_set",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "write_golden_set",
]
