"""Measure what cross-encoder reranking buys and what it costs — input to TASK-0175.

TASK-0175 is a maintainer decision about the shipped ``retrieval.rerank`` default, and it is
blocked on a number nobody has: reranking is *expected* to be the strongest quality lever in
the stack, and that expectation has never been measured (TASK-0084). This script produces
the four numbers the decision turns on, per candidate configuration — the retrieval quality
the eval harness already defines, the model's on-disk size, its cold load time, and its
per-query rerank latency.

**It decides nothing.** It changes no default, recommends none, and writes to no
configuration file. It prints a table and, optionally, JSON.

**It reuses this repository's own notion of retrieval quality** rather than inventing a
second one: the committed golden sets under ``tests/eval/datasets/``, their document-level
ground truth, and the recall@k / MRR / nDCG@k of ``fasterrag.core.evals.evaluate``. A metric
invented here would not be comparable with the committed baselines, with the regression
gate, or with anything the nightly job records — which would make it nearly worthless for
deciding anything.

**What is substituted, and why the comparison still holds.** The vector database is replaced
by an exact brute-force store in this process (:class:`InMemoryVectorDB`), so the
measurement runs without a container. Everything else is the shipped pipeline: the real
parser and chunker (``workers.cpu_pool.parse_and_chunk``), the real embedding adapter, the
real BM25 encoder, the real RRF fusion, the real ``CrossEncoderReranker``, assembled by the
real ``api.dependencies.build_retrieval``. The store searches *exactly* rather than
approximately, so at these corpus sizes it is if anything a better retriever than an HNSW
index; and because every configuration is scored against the same store, the differences
between configurations are unaffected by the substitution. The absolute numbers are not a
reproduction of a Qdrant deployment and must not be read as one.

**The limits are printed with the numbers rather than left to the reader.** The committed
golden sets are tens of queries each, and ``docs/performance.md`` §Methodology requires an
isolated, documented machine before any number is citable. Every quality delta is therefore
reported as a *paired* per-query mean with its own standard error, so a difference smaller
than the noise shows up as one instead of being read as a result.

Usage::

    python scripts/measure_rerank.py
    python scripts/measure_rerank.py --dataset tests/eval/datasets/policies --json out.json

Weights are never downloaded unless ``--allow-download`` is passed: the shipped default is
2.2 GB, and a measurement script that silently pulls it is the very surprise TASK-0174
fixed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from fasterrag.adapters.embeddings.tiering import TieringRouter, create_embedding_router, matches
from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    SparseVector,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.api.dependencies import build_retrieval
from fasterrag.config.schema import Settings
from fasterrag.core.evals import EvalReport, evaluate
from fasterrag.core.rerank import Reranker
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.services.benchmark import REPETITIONS, commit_hash, fingerprint, percentile
from fasterrag.services.evaluation import EvalDataset, load_dataset
from fasterrag.workers.cpu_pool import CpuWorkerPool, parse_and_chunk
from fasterrag.workers.indexer import Indexer
from fasterrag.workers.queues import ChunkPayload, EmbeddedBatch

__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_MODELS",
    "EMBEDDING_MODEL",
    "METRICS",
    "OFF",
    "UNLOADED_DIRECTORIES",
    "Configuration",
    "Footprint",
    "InMemoryVectorDB",
    "Latency",
    "Measurement",
    "PairedDelta",
    "comparison_table",
    "configurations",
    "delta_table",
    "human_bytes",
    "loadable_files",
    "main",
    "measure_dataset",
    "paired_delta",
    "parse_args",
    "per_query",
]

EMBEDDING_MODEL: Final = "BAAI/bge-small-en-v1.5"
SHIPPED_MODEL: Final = "BAAI/bge-reranker-v2-m3"
SMALL_MODEL: Final = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MODELS: Final[tuple[str, ...]] = (SMALL_MODEL, SHIPPED_MODEL)

OFF: Final = "rerank off"
COLLECTION: Final = "rerank_measurement"

DEFAULT_DATASETS: Final[tuple[str, ...]] = (
    "tests/eval/datasets/handbook",
    "tests/eval/datasets/policies",
)

METRICS: Final[tuple[str, ...]] = ("recall", "reciprocal_rank", "ndcg")
METRIC_LABELS: Final[dict[str, str]] = {
    "recall": "recall@k",
    "reciprocal_rank": "MRR",
    "ndcg": "nDCG@k",
}

# CRITICAL: two standard errors, not one. A one-sigma band calls roughly a third of pure
# noise a result, and the whole point of this script is to be honest about whether a golden
# set of a few dozen queries can resolve the difference at all.
NOISE_SIGMAS: Final = 2.0
MINIMUM_PAIRS: Final = 2

# What a sentence-transformers cross-encoder load never reads, and must therefore not be
# charged to the model's size. Both halves are real: `cross-encoder/ms-marco-MiniLM-L-6-v2`
# ships five ONNX builds and a Flax copy of the same 90 MB of weights, and
# `BAAI/bge-reranker-v2-m3` ships a quarter of a megabyte of README screenshots. Counting
# either would report a model as several times larger than the pipeline makes it.
UNLOADED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {"onnx", "openvino", "coreml", "tflite", "rust_model", "assets", "imgs", "images"}
)
DUPLICATE_WEIGHTS: Final[frozenset[str]] = frozenset(
    {"pytorch_model.bin", "tf_model.h5", "flax_model.msgpack", "rust_model.ot"}
)
SAFETENSORS: Final = "model.safetensors"
WEIGHT_SUFFIXES: Final[tuple[str, ...]] = (".safetensors", ".bin", ".pt", ".ckpt")
IGNORED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".md", ".png", ".jpg", ".jpeg", ".gif", ".svg"}
)
# CRITICAL: matched by name, not by suffix. ``Path(".gitattributes").suffix`` is the empty
# string, so a dotfile listed among the suffixes above is silently never excluded.
IGNORED_NAMES: Final[frozenset[str]] = frozenset({".gitattributes", ".gitignore"})

_BYTES_PER_UNIT: Final = 1024.0
_UNITS: Final[tuple[str, ...]] = ("B", "KB", "MB", "GB", "TB")


@dataclass(frozen=True, slots=True)
class Configuration:
    """One candidate setting of the ``retrieval.rerank`` default."""

    name: str
    rerank: bool
    model: str = ""


def configurations(models: Sequence[str]) -> list[Configuration]:
    """Return the configurations to measure, reranking-off first as the baseline."""
    ordered = [Configuration(name=OFF, rerank=False)]
    ordered.extend(
        Configuration(name=model, rerank=True, model=model)
        for model in dict.fromkeys(models)
        if model
    )
    return ordered


@dataclass(frozen=True, slots=True)
class Latency:
    """Per-call latencies, in milliseconds."""

    samples: tuple[float, ...] = ()
    candidates: int = 0

    @property
    def p50(self) -> float:
        """Return the median call, in milliseconds."""
        return percentile(self.samples, 0.50)

    @property
    def p95(self) -> float:
        """Return the 95th-percentile call, in milliseconds."""
        return percentile(self.samples, 0.95)

    @property
    def mean(self) -> float:
        """Return the mean call, in milliseconds."""
        return math.fsum(self.samples) / len(self.samples) if self.samples else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "calls": len(self.samples),
            "candidates": self.candidates,
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "mean_ms": round(self.mean, 2),
        }


@dataclass(frozen=True, slots=True)
class Footprint:
    """What one cross-encoder costs on disk and how long it takes to load."""

    model: str
    counted_bytes: int = 0
    weight_bytes: int = 0
    files: tuple[str, ...] = ()
    load_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "model": self.model,
            "counted_bytes": self.counted_bytes,
            "counted": human_bytes(self.counted_bytes),
            "weight_bytes": self.weight_bytes,
            "weights": human_bytes(self.weight_bytes),
            "files": list(self.files),
            "load_seconds": round(self.load_seconds, 2),
        }


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """How far a configuration moved one metric, and how noisy that move is."""

    metric: str
    mean: float = 0.0
    standard_error: float = 0.0
    n: int = 0

    @property
    def resolvable(self) -> bool:
        """Return whether the move is larger than the noise in the measurement itself."""
        return self.n >= MINIMUM_PAIRS and abs(self.mean) > NOISE_SIGMAS * self.standard_error

    def describe(self) -> str:
        """Return the ``+0.0000 +/- 0.0000`` form the table prints."""
        return f"{self.mean:+.4f} +/- {self.standard_error:.4f}"

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "metric": self.metric,
            "mean": round(self.mean, 6),
            "standard_error": round(self.standard_error, 6),
            "queries": self.n,
            "larger_than_noise": self.resolvable,
        }


@dataclass(frozen=True, slots=True)
class Measurement:
    """Everything one configuration produced against one dataset."""

    dataset: str
    configuration: Configuration
    report: EvalReport
    footprint: Footprint | None = None
    rerank_latency: Latency | None = None
    probe_latency: Latency | None = None
    shortlist: float = 0.0
    indexed: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "dataset": self.dataset,
            "indexed_chunks": self.indexed,
            "configuration": self.configuration.name,
            "model": self.configuration.model,
            "quality": self.report.as_dict(),
            "footprint": self.footprint.as_dict() if self.footprint else None,
            "rerank_latency": self.rerank_latency.as_dict() if self.rerank_latency else None,
            "probe_latency": self.probe_latency.as_dict() if self.probe_latency else None,
            "mean_shortlist": round(self.shortlist, 1),
        }


def human_bytes(count: int) -> str:
    """Return a byte count in the largest unit that keeps it above one."""
    size = float(count)
    for unit in _UNITS:
        if abs(size) < _BYTES_PER_UNIT or unit == _UNITS[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= _BYTES_PER_UNIT
    return f"{size:.2f} {_UNITS[-1]}"


def loadable_files(paths: Iterable[str]) -> list[str]:
    """Return the snapshot files a sentence-transformers cross-encoder load actually reads."""
    kept = [path for path in paths if _is_loadable(path)]
    if SAFETENSORS in kept:
        kept = [path for path in kept if path not in DUPLICATE_WEIGHTS]
    return sorted(kept)


def _is_loadable(path: str) -> bool:
    """Return whether one snapshot-relative path belongs to the loaded model."""
    parsed = Path(path)
    if any(part in UNLOADED_DIRECTORIES for part in parsed.parts[:-1]):
        return False
    if parsed.name in IGNORED_NAMES:
        return False
    return parsed.suffix not in IGNORED_SUFFIXES


def footprint(model: str, snapshot: Path) -> Footprint:
    """Measure what a cached model occupies on disk, counting only what is loaded."""
    relative = [
        path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()
    ]
    counted = loadable_files(relative)
    sizes = {name: (snapshot / name).stat().st_size for name in counted}
    weights = [size for name, size in sizes.items() if Path(name).suffix in WEIGHT_SUFFIXES]

    return Footprint(
        model=model,
        counted_bytes=sum(sizes.values()),
        weight_bytes=max(weights, default=0),
        files=tuple(counted),
    )


def per_query(report: EvalReport, metric: str) -> dict[str, float]:
    """Return every scored query's value of one metric, keyed by query id."""
    if metric not in METRICS:
        raise ValueError(f"{metric!r} is not one of the harness metrics: {', '.join(METRICS)}")

    return {
        score.id: float(getattr(score, metric))
        for score in report.per_query
        if not score.adversarial
    }


def paired_delta(
    metric: str, baseline: Mapping[str, float], variant: Mapping[str, float]
) -> PairedDelta:
    """Return the mean per-query change in a metric and the standard error of that change.

    Paired because both configurations score the same queries: differencing per query
    removes the between-query variance that otherwise swamps a few dozen samples.
    """
    shared = sorted(set(baseline) & set(variant))
    differences = [variant[identifier] - baseline[identifier] for identifier in shared]
    if not differences:
        return PairedDelta(metric=metric)

    mean = math.fsum(differences) / len(differences)
    error = (
        statistics.stdev(differences) / math.sqrt(len(differences))
        if len(differences) >= MINIMUM_PAIRS
        else 0.0
    )
    return PairedDelta(metric=metric, mean=mean, standard_error=error, n=len(differences))


def _row(cells: Sequence[str]) -> str:
    """Return one markdown table row."""
    return "| " + " | ".join(cells) + " |"


def comparison_table(measurements: Sequence[Measurement]) -> str:
    """Render the configuration-by-number table the decision turns on."""
    header = (
        "Dataset",
        "Configuration",
        "recall@k",
        "MRR",
        "nDCG@k",
        "On disk",
        "Cold load",
        "Rerank p50",
        "Rerank p95",
    )
    lines = [_row(header), _row(["---"] * len(header))]

    for measurement in measurements:
        report = measurement.report
        disk = measurement.footprint
        latency = measurement.rerank_latency
        lines.append(
            _row(
                (
                    measurement.dataset,
                    measurement.configuration.name,
                    f"{report.recall_at_k:.4f}",
                    f"{report.mrr:.4f}",
                    f"{report.ndcg_at_k:.4f}",
                    human_bytes(disk.counted_bytes) if disk else "n/a",
                    f"{disk.load_seconds:.1f} s" if disk else "n/a",
                    f"{latency.p50:.1f} ms" if latency else "n/a",
                    f"{latency.p95:.1f} ms" if latency else "n/a",
                )
            )
        )

    return "\n".join(lines)


def delta_table(measurements: Sequence[Measurement]) -> str:
    """Render each configuration's paired quality delta against reranking off."""
    header = ("Dataset", "Configuration", "Metric", "Delta vs rerank off", "Larger than noise?")
    lines = [_row(header), _row(["---"] * len(header))]

    for dataset in dict.fromkeys(measurement.dataset for measurement in measurements):
        group = [measurement for measurement in measurements if measurement.dataset == dataset]
        baselines = [item for item in group if item.configuration.name == OFF]
        if not baselines:
            continue

        for measurement in group:
            if measurement.configuration.name == OFF:
                continue
            for metric in METRICS:
                delta = paired_delta(
                    metric,
                    per_query(baselines[0].report, metric),
                    per_query(measurement.report, metric),
                )
                lines.append(
                    _row(
                        (
                            dataset,
                            measurement.configuration.name,
                            METRIC_LABELS[metric],
                            delta.describe(),
                            "yes" if delta.resolvable else "no",
                        )
                    )
                )

    return "\n".join(lines)


class InMemoryVectorDB(VectorDBAdapter):
    """Exact brute-force dense and BM25 search over points held in this process.

    A stand-in for the backend so the measurement needs no container. Only the operations
    retrieval uses are implemented; the rest raise rather than pretend.
    """

    def __init__(self, settings: Settings) -> None:
        """Build an empty store."""
        super().__init__(settings)
        self._points: dict[str, list[Point]] = {}

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Register an empty collection."""
        self._points.setdefault(spec.name, [])

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Append points, replacing any that share a point id."""
        for point in points:
            held = self._points.setdefault(point.collection, [])
            self._points[point.collection] = [
                existing for existing in held if existing.point_id != point.point_id
            ]
            self._points[point.collection].append(point)
        return UpsertResult(upserted=len(points))

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points by cosine similarity or by BM25 with IDF."""
        held = [
            point
            for point in self._points.get(query.collection, [])
            if query.filters is None or matches(query.filters, point.payload)
        ]

        if query.vector is not None:
            scored = [(point, _cosine(query.vector, point.vector)) for point in held]
        else:
            scored = _bm25_scores(query.sparse, held)

        ranked = sorted(
            (item for item in scored if item[1] > 0.0),
            key=lambda item: (-item[1], item[0].point_id),
        )
        return [
            ScoredPoint(point_id=point.point_id, score=score, payload=point.payload)
            for point, score in ranked[: query.limit]
        ]

    async def health(self) -> HealthStatus:
        """Report the store as healthy; it is this process's own memory."""
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        """Release nothing; there is nothing to release."""
        return

    async def list_collections(self) -> list[CollectionInfo]:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def drop_collection(self, name: str) -> bool:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def snapshot(self, collection: str) -> str:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def list_snapshots(self, collection: str) -> list[str]:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def set_alias(self, alias: str, collection: str) -> None:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def alias_target(self, alias: str) -> str | None:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def delete_alias(self, alias: str) -> bool:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def update(self, updates: list[PointUpdate]) -> None:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")

    async def delete(self, selector: PointSelector) -> None:
        """Unsupported in the measurement store."""
        raise NotImplementedError("the measurement store only indexes and searches")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two dense vectors."""
    numerator = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(math.fsum(a * a for a in left)) * math.sqrt(math.fsum(b * b for b in right))
    return numerator / norms if norms else 0.0


def _bm25_scores(query: SparseVector | None, held: Sequence[Point]) -> list[tuple[Point, float]]:
    """Score a sparse query against held points, supplying the IDF a backend would.

    The encoder leaves IDF to the store on purpose (``core/retrieval/bm25``), so a store
    that omitted it would run a different BM25 from the one the pipeline is built around.
    """
    if query is None:
        return []

    total = len(held)
    frequencies: dict[int, int] = {}
    for point in held:
        for index in set(point.sparse.indices if point.sparse else ()):
            frequencies[index] = frequencies.get(index, 0) + 1

    scored: list[tuple[Point, float]] = []
    for point in held:
        sparse = point.sparse
        weights = dict(zip(sparse.indices, sparse.values, strict=True)) if sparse else {}
        score = math.fsum(
            value * weights[index] * _idf(frequencies.get(index, 0), total)
            for index, value in zip(query.indices, query.values, strict=True)
            if index in weights
        )
        scored.append((point, score))
    return scored


def _idf(documents: int, total: int) -> float:
    """Return the Robertson-Sparck-Jones inverse document frequency of one term."""
    if total <= 0:
        return 0.0
    return math.log(1 + (total - documents + 0.5) / (documents + 0.5))


class TimingReranker:
    """Wraps a reranker and records how long each call took and how wide it was."""

    def __init__(self, inner: Reranker) -> None:
        """Wrap ``inner``, which must satisfy the reranker protocol."""
        self.inner = inner
        self.samples: list[float] = []
        self.widths: list[int] = []

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Rerank through the wrapped reranker, timing the call."""
        started = time.perf_counter()
        reordered = await self.inner.rerank(query, chunks)
        self.samples.append((time.perf_counter() - started) * 1000)
        self.widths.append(len(chunks))
        return list(reordered)

    @property
    def latency(self) -> Latency:
        """Return the recorded latencies and the mean shortlist width."""
        width = round(math.fsum(self.widths) / len(self.widths)) if self.widths else 0
        return Latency(samples=tuple(self.samples), candidates=width)


def build_settings(
    configuration: Configuration, embedding_model: str, chunk_size: int | None = None
) -> Settings:
    """Return the settings one configuration is measured under."""
    retrieval: dict[str, Any] = {"rerank": configuration.rerank}
    if configuration.model:
        retrieval["reranker_model"] = configuration.model

    payload: dict[str, Any] = {
        "embeddings": {"provider": "huggingface", "model": embedding_model},
        "retrieval": retrieval,
        "vector_db": {"collection": {"default_name": COLLECTION}},
    }
    if chunk_size is not None:
        payload["chunking"] = {"chunk_size": chunk_size, "overlap": min(64, chunk_size // 4)}

    return Settings.model_validate(payload)


def _chunks(dataset: EvalDataset, settings: Settings) -> list[ChunkPayload]:
    """Parse and chunk the dataset's corpus exactly as ingestion would."""
    sources = [str(dataset.corpus / path.name) for path in sorted(dataset.corpus.glob("*.md"))]
    payloads: list[ChunkPayload] = []
    for task in CpuWorkerPool.tasks_for(sources):
        payloads.extend(parse_and_chunk(task, settings).chunks)
    return payloads


async def _index(
    dataset: EvalDataset, settings: Settings, router: TieringRouter
) -> tuple[InMemoryVectorDB, int]:
    """Chunk, embed, and index the corpus once, for every configuration to share."""
    payloads = _chunks(dataset, settings)
    embedded = await router.default.embed_documents([payload.text for payload in payloads])

    store = InMemoryVectorDB(settings)
    indexer = Indexer(settings, store, collection=COLLECTION)
    await indexer.write(
        EmbeddedBatch(
            chunks=payloads,
            vectors=[list(vector) for vector in embedded.vectors],
            model=embedded.model,
            model_version=embedded.model_version,
        )
    )
    return store, len(payloads)


def _snapshot(model: str, *, allow_download: bool) -> Path:
    """Locate a model's cached snapshot, fetching it beforehand only when permitted.

    Resolved through one known file rather than ``snapshot_download``: a repository
    carrying ONNX builds the pipeline never loads is *incomplete* by that function's
    definition and raises, although everything needed is on disk.
    """
    from huggingface_hub import try_to_load_from_cache

    from fasterrag.core.rerank import load_cross_encoder

    cached = try_to_load_from_cache(model, "config.json")
    if not isinstance(cached, str):
        if not allow_download:
            raise SystemExit(
                f"{model} is not in the local hugging-face cache, and fetching it could take "
                "gigabytes; the shipped default is 2.2 GB. Re-run with --allow-download."
            )
        _write(f"  fetching {model} before timing anything\n")
        load_cross_encoder(model)
        cached = try_to_load_from_cache(model, "config.json")

    if not isinstance(cached, str):
        raise SystemExit(f"{model} is still not in the hugging-face cache after fetching it")
    return Path(cached).parent


async def _warm(settings: Settings, *, allow_download: bool) -> tuple[Reranker, Footprint]:
    """Load a cross-encoder, timing the load, and measure what it occupies on disk."""
    from fasterrag.core.rerank import CrossEncoderReranker

    model = settings.retrieval.reranker_model
    disk = footprint(model, _snapshot(model, allow_download=allow_download))

    reranker = CrossEncoderReranker(settings)
    probe = [ScoredChunk(chunk_id="c_warm", text="a passage used only to force the load")]
    started = time.perf_counter()
    await asyncio.to_thread(reranker.score, "a warm-up query", probe)
    elapsed = time.perf_counter() - started

    return reranker, replace(disk, load_seconds=elapsed)


async def _probe(
    reranker: Reranker,
    payloads: Sequence[ChunkPayload],
    query: str,
    *,
    width: int,
    repetitions: int,
) -> Latency:
    """Time reranking at a fixed shortlist width, the one the shipped default retrieves."""
    pool = [
        ScoredChunk(chunk_id=f"c_{index}", text=payloads[index % len(payloads)].text)
        for index in range(width)
    ]

    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        await reranker.rerank(query, pool)
        samples.append((time.perf_counter() - started) * 1000)
    return Latency(samples=tuple(samples), candidates=width)


async def measure_dataset(
    root: Path,
    specs: Sequence[Configuration],
    *,
    embedding_model: str,
    repetitions: int,
    probe_width: int,
    allow_download: bool,
    chunk_size: int | None = None,
) -> list[Measurement]:
    """Measure every configuration against one dataset, sharing one indexed corpus."""
    dataset = load_dataset(root)
    base = build_settings(Configuration(name=OFF, rerank=False), embedding_model, chunk_size)
    router = create_embedding_router(base)

    try:
        store, indexed = await _index(dataset, base, router)
        label = f"{root.name}/{base.chunking.chunk_size}t"
        _write(
            f"{label}: {indexed} chunks indexed, {dataset.k} retrieved per query, "
            f"{len(dataset.records)} golden records\n"
        )

        measurements: list[Measurement] = []
        for spec in specs:
            measurements.append(
                await _measure_one(
                    dataset,
                    spec,
                    store,
                    router,
                    label=label,
                    indexed=indexed,
                    embedding_model=embedding_model,
                    repetitions=repetitions,
                    probe_width=probe_width,
                    allow_download=allow_download,
                    chunk_size=chunk_size,
                )
            )
        return measurements
    finally:
        await router.close()


async def _measure_one(
    dataset: EvalDataset,
    spec: Configuration,
    store: InMemoryVectorDB,
    router: TieringRouter,
    *,
    label: str,
    indexed: int,
    embedding_model: str,
    repetitions: int,
    probe_width: int,
    allow_download: bool,
    chunk_size: int | None,
) -> Measurement:
    """Score one configuration, timing its reranker if it has one."""
    settings = build_settings(spec, embedding_model, chunk_size)
    disk: Footprint | None = None
    probe: Latency | None = None

    retrieval = build_retrieval(settings, store, router)
    timer: TimingReranker | None = None
    if spec.rerank:
        reranker, disk = await _warm(settings, allow_download=allow_download)
        timer = TimingReranker(reranker)
        retrieval.reranker = timer

    _write(f"  {spec.name}: scoring {repetitions} repetitions\n")
    reports = [
        await evaluate(dataset.resolved(), retrieval, k=dataset.k, collection=COLLECTION)
        for _ in range(repetitions)
    ]
    _check_stability(spec, reports)

    if timer is not None and spec.rerank:
        payloads = _chunks(dataset, settings)
        query = next(record.query for record in dataset.records if not record.adversarial)
        probe = await _probe(
            timer.inner, payloads, query, width=probe_width, repetitions=repetitions
        )

    return Measurement(
        dataset=label,
        configuration=spec,
        report=reports[0],
        footprint=disk,
        rerank_latency=timer.latency if timer else None,
        probe_latency=probe,
        shortlist=timer.latency.candidates if timer else 0.0,
        indexed=indexed,
    )


def _check_stability(spec: Configuration, reports: Sequence[EvalReport]) -> None:
    """Report whether repeated runs of a deterministic pipeline really agreed."""
    distinct = {(report.recall_at_k, report.mrr, report.ndcg_at_k) for report in reports}
    if len(distinct) > 1:
        _write(f"  {spec.name}: WARNING - repetitions disagreed: {sorted(distinct)}\n")


def _write(text: str) -> None:
    """Write progress to stderr so stdout carries only the report."""
    sys.stderr.write(text)
    sys.stderr.flush()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="measure_rerank.py",
        description=(
            "Measure retrieval quality, model size, cold load time, and per-query rerank "
            "latency for each candidate rerank default. Input to TASK-0175; decides nothing."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="an eval dataset directory; repeatable (default: both committed datasets)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="a cross-encoder to measure; repeatable (default: the small and shipped models)",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"the embedding model both legs share (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
        help=f"how many times to score each configuration (default: {REPETITIONS})",
    )
    parser.add_argument(
        "--probe-candidates",
        type=int,
        default=100,
        help="shortlist width for the standalone latency probe (default: retrieval.rerank_top_n)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit fetching weights that are not already cached (the default is 2.2 GB)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "override chunking.chunk_size, to retrieve over a corpus the default 768-token "
            "chunker collapses to one chunk per document (default: the shipped value)"
        ),
    )
    parser.add_argument("--json", dest="json_path", default=None, help="also write JSON here")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.probe_candidates < 1:
        parser.error("--probe-candidates must be at least 1")
    return args


def _document(measurements: Sequence[Measurement]) -> dict[str, Any]:
    """Return the machine-readable report, hardware and commit attached."""
    return {
        "hardware": fingerprint().as_dict(),
        "commit": commit_hash(),
        "citable": False,
        "citable_reason": (
            "docs/performance.md §Methodology requires an isolated documented machine; "
            "this ran on a developer workstation with co-tenant load"
        ),
        "measurements": [measurement.as_dict() for measurement in measurements],
    }


async def run(args: argparse.Namespace) -> int:
    """Measure every configuration against every dataset and print the report."""
    specs = configurations(args.models or list(DEFAULT_MODELS))
    roots = [Path(name) for name in (args.datasets or list(DEFAULT_DATASETS))]

    measurements: list[Measurement] = []
    for root in roots:
        measurements.extend(
            await measure_dataset(
                root,
                specs,
                embedding_model=args.embedding_model,
                repetitions=args.repetitions,
                probe_width=args.probe_candidates,
                allow_download=args.allow_download,
                chunk_size=args.chunk_size,
            )
        )

    sys.stdout.write(comparison_table(measurements) + "\n\n")
    sys.stdout.write(delta_table(measurements) + "\n\n")
    for measurement in measurements:
        if measurement.probe_latency is not None:
            probe = measurement.probe_latency
            sys.stdout.write(
                f"probe {measurement.dataset} / {measurement.configuration.name}: "
                f"{probe.candidates} candidates, p50 {probe.p50:.1f} ms, "
                f"p95 {probe.p95:.1f} ms\n"
            )

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(_document(measurements), indent=2) + "\n", encoding="utf-8"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement."""
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
