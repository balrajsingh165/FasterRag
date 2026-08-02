"""Retrieval evaluation: golden sets, metrics, and the harness.

Measured quality is what stops "the retrieval got better" from being an opinion. Everything
here is pure computation over a golden set and a retriever, so it runs without a model, a
network, or a container.
"""

from fasterrag.core.evals.generator import (
    ADVERSARIAL_FRACTION,
    CandidateChunk,
    generate_golden_set,
    stratified_sample,
)
from fasterrag.core.evals.golden import GoldenRecord, load_golden_set, write_golden_set
from fasterrag.core.evals.harness import EvalReport, QueryScore, Retriever, evaluate
from fasterrag.core.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "ADVERSARIAL_FRACTION",
    "CandidateChunk",
    "EvalReport",
    "GoldenRecord",
    "QueryScore",
    "Retriever",
    "evaluate",
    "generate_golden_set",
    "load_golden_set",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "stratified_sample",
    "write_golden_set",
]
