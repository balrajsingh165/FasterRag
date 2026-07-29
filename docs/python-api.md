# python-api.md — Python Package (Library Surface)

fasterRag is released as an installable Python package so applications can **import and use the framework in-process**, without running the HTTP service. The library is the third programmatic control surface alongside the [REST API](api-reference.md) and the [CLI](cli-reference.md) — all three call the identical service layer, so behavior, config, and errors are the same everywhere. **There is still no GUI control surface**; the dashboard remains observability-only.

> All code below is the **documented intended public interface** for the beta — illustrative snippets, not shipped code. The build tasks live in [todo.md](todo.md).

## Installation

```bash
pip install fasterrag                 # core (local HuggingFace embeddings, Qdrant adapter)
pip install "fasterrag[all]"          # every optional adapter/provider
pip install "fasterrag[openai]"       # extras per provider: openai, anthropic, cohere, ollama
pip install "fasterrag[milvus]"       # extras per vector DB: milvus, weaviate, pinecone, pgvector, chroma
pip install "fasterrag[rerank]"       # cross-encoder reranker models
```

- Requires **Python 3.12+**.
- Versioned by **SemVer 2.0.0**; the public API defined in this document is the compatibility contract — breaking it requires a major version bump and a CHANGELOG entry.
- Wheels published to PyPI at every tagged release; dependencies pinned and hash-locked in the repo.

## Quickstart

```python
import asyncio
from fasterrag import FasterRag

async def main() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        job = await rag.ingest(["./docs/", "https://example.com/spec.pdf"])
        await job.wait()

        result = await rag.query("What does the spec say about retries?")
        print(result.answer)
        for c in result.citations:
            print(c.source, c.page, c.span)

asyncio.run(main())
```

The same `config.yaml` + `.env` contract applies ([config-reference.md](config-reference.md)): config drives all behavior, secrets come only from the environment, validation fails fast at `from_config()` with a `ConfigError` naming the offending key.

## Public API surface (beta contract)

Everything importable from the top-level `fasterrag` namespace is public and stable; anything under `fasterrag._internal` or not exported in `__all__` is private and may change without notice.

### `FasterRag`

| Member | Signature (intended) | Behavior |
|---|---|---|
| `FasterRag.from_config` | `(path: str \| Path = "config.yaml") -> FasterRag` | Load + validate config, build adapters via factories. No I/O to backends yet. |
| `FasterRag.from_settings` | `(settings: Settings) -> FasterRag` | Construct from an in-memory validated `Settings` object (for embedding in apps that manage their own config). |
| `__aenter__` / `__aexit__` | async context manager | Startup: connect adapters, run health checks, start in-process worker pools sized per `workers.*`. Shutdown: drain queues, flush journal, close connections. |
| `ingest` | `(sources: list[str \| Source], *, collection: str \| None = None, metadata: dict \| None = None, priority_class: str \| None = None) -> IngestJob` | Async accept, identical semantics to `POST /v1/ingest` (journaled, deduplicated, DLQ on failure). |
| `query` | `(text: str, *, collection: str \| None = None, top_k: int \| None = None, filters: dict \| None = None) -> QueryResult` | Full pipeline: hybrid retrieve → RRF → rerank → assemble → generate. |
| `query_stream` | `(...) -> AsyncIterator[QueryEvent]` | Streaming variant; yields `meta`, `token`, `citations`, `usage`, `done` events mirroring the SSE contract. |
| `retrieve` | `(text: str, *, top_k: int \| None = None, filters: dict \| None = None, rerank: bool \| None = None) -> list[ScoredChunk]` | Retrieval only — no generation. For apps that bring their own LLM step. |
| `collections` | `CollectionsAPI` | `list()`, `create()`, `delete()`, `reindex()` (blue/green, eval-gated), `rollback()`, `verify_lock()`. |
| `estimate` | `(sources: list[str \| Source]) -> Estimate` | D9 preflight: token counts, projected embedding cost/time per provider. |
| `doctor` | `() -> DoctorReport` | D10 diagnostics; each check has `passed: bool` and `fix: str`. |
| `replay` | `(trace_id: str, config_overrides: dict) -> ReplayDiff` | D8 side-by-side replay diff. |
| `export_archive` / `import_archive` | `(path, *, include_vectors=False)` / `(path, *, reembed=False)` | D11 portability. |

### Result models (frozen Pydantic models)

- `QueryResult`: `answer: str | None`, `citations: list[Citation]`, `usage: Usage`, `timings_ms: Timings`, `degraded: bool`, `mode: str`, `faithfulness: float | None`, `trace_id: str`, `insufficient_evidence: bool` (D5 — `answer is None` when refused).
- `Citation`: `chunk_id`, `source`, `page`, `span (start, end)`, `score`.
- `ScoredChunk`: `chunk_id`, `text`, `metadata`, `dense_rank`, `bm25_rank`, `rrf_score`, `rerank_score | None`.
- `IngestJob`: `job_id`, `status`, `counts`, `await job.wait(timeout=None)`, `async for doc in job.documents(status="dead_lettered")`, `await job.retry_dlq()`.

### Exceptions

The full typed taxonomy from [reliability.md](reliability.md) is importable:

```python
from fasterrag.errors import (
    FasterRagError, ConfigError, IngestionError, ParseError, ChunkError,
    EmbedError, RetrievalError, GenerationError, ProviderError, CacheError,
    ProvisioningError,
)
```

Every exception carries `code` (identical to the API's problem `code`), `trace_id`, and `retryable`. Library and API surface the same error identities — a wrapper service can pass `code` straight through.

### Sync facade

For scripts and notebooks that are not async:

```python
from fasterrag.sync import FasterRag

with FasterRag.from_config("config.yaml") as rag:
    rag.ingest(["./docs/"]).wait()
    print(rag.query("...").answer)
```

The sync facade wraps the async engine in a managed event loop; it is a thin adapter, not a second implementation.

## Standalone components (use the pieces without the pipeline)

Every pipeline stage is importable and usable on its own, so applications can adopt fasterRag piecemeal — just the chunkers, just hybrid fusion, just the evaluator — before (or instead of) adopting the full engine:

```python
from fasterrag.parsing import parse_document            # -> ParsedDocument (reading order, tables, headings, metadata)
from fasterrag.chunking import RecursiveChunker, SemanticChunker, LayoutChunker
from fasterrag.retrieval import rrf_fuse                # Reciprocal Rank Fusion, k=60 default
from fasterrag.rerank import CrossEncoderReranker
from fasterrag.evals import evaluate                    # recall@k, MRR, nDCG, faithfulness

chunks = RecursiveChunker(chunk_size=768, overlap=64).split(parse_document("spec.pdf"))
fused = rrf_fuse(dense_ranking, bm25_ranking, k=60)
report = evaluate(golden_set, retriever=my_retriever)
```

Each standalone component obeys the same contracts (typed errors, docstrings, tested invariants) as the full pipeline.

## Embedding fasterRag inside an existing FastAPI/ASGI app

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fasterrag import FasterRag

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with FasterRag.from_config("config.yaml") as rag:
        app.state.rag = rag
        yield

app = FastAPI(lifespan=lifespan)

@app.get("/answer")
async def answer(q: str):
    result = await app.state.rag.query(q)
    return {"answer": result.answer, "citations": [c.model_dump() for c in result.citations]}
```

Deployment note: in-process worker pools are suitable for small/medium loads; for heavy ingestion run dedicated `fasterrag worker` processes against the same config and queue backend ([deployment.md](deployment.md)).

## Extending the framework (plugin contract)

Third parties add providers by implementing the adapter base classes and registering them via Python entry points — no fork required:

```toml
[project.entry-points."fasterrag.vectordb"]
mydb = "my_pkg.adapter:MyDBAdapter"
```

- `fasterrag.vectordb` → subclass `VectorDBAdapter` (`create_collection`, `upsert`, `search`, `update`, `delete`, `health`).
- `fasterrag.embeddings` → subclass `EmbeddingAdapter`.
- `fasterrag.llm` → subclass `LLMAdapter`.

A registered provider becomes selectable in `config.yaml` (`vector_db.provider: mydb`) and MUST pass the shared **adapter contract test suite** ([testing-strategy.md](testing-strategy.md)) to be considered conformant.

## Stability guarantees

| Surface | Guarantee |
|---|---|
| `fasterrag` top-level exports, `fasterrag.errors`, `fasterrag.sync` | SemVer-stable public API. |
| Adapter base classes + entry-point group names | SemVer-stable extension contract. |
| `fasterrag._internal.*` | No guarantees. |
| Result-model fields | Additive-only within a major version. |
