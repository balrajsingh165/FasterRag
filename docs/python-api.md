# python-api.md — Python Package (Library Surface)

fasterRag is released as an installable Python package so applications can **import and use the framework in-process**, without running the HTTP service. The library is the third programmatic control surface alongside the [REST API](api-reference.md) and the [CLI](cli-reference.md) — all three call the identical service layer, so behavior, config, and errors are the same everywhere. **There is still no GUI control surface**; the dashboard remains observability-only.

> **Per-surface status against the implemented package** (build tasks in [todo.md](todo.md)):
>
> | Surface in this document | Status |
> |---|---|
> | Standalone components — `fasterrag.parsing`, `.chunking`, `.retrieval`, `.rerank`, `.evals` | **Shipped** and importable today |
> | Typed error taxonomy — `fasterrag.errors` (same `code`s as the API) | **Shipped** |
> | `FasterRag` facade — `from_config`, `from_settings`, `ingest`, `query`, `query_stream`, `retrieve`, `estimate`, `index_lock` | **Shipped**. Verified end to end against live Qdrant and OpenAI |
> | `FasterRag.collections`, `.doctor`, `.replay`, `.export_archive` / `.import_archive` | **Not yet implemented** — the CLI and REST surfaces cover these today |
> | `fasterrag.sync` blocking facade | **Not yet implemented** (follows the async facade) |
> | Entry-point plugin groups (`fasterrag.vectordb` / `.embeddings` / `.llm`) | **Not yet implemented** (TASK-0163) — today the factories resolve built-ins only |
> | PyPI wheels (`pip install fasterrag`) | **Not yet published** (TASK-0087) — install from source: `pip install -e ".[all]"` |
>
> Sections describing an unshipped surface are the design contract the remaining work must satisfy, kept here so each one is built to a reviewed spec rather than improvised.

## Installation

```bash
pip install fasterrag                 # core: Qdrant adapter, all document parsers, chunking
pip install "fasterrag[all]"          # every optional adapter/provider
pip install "fasterrag[huggingface]"  # local sentence-transformers embeddings
pip install "fasterrag[openai]"       # extras per provider: openai, anthropic, cohere, ollama
pip install "fasterrag[milvus]"       # extras per vector DB: milvus, weaviate, pinecone, pgvector, chroma
pip install "fasterrag[rerank]"       # cross-encoder reranker models
pip install "fasterrag[ocr]"          # OCR for scanned PDFs (also needs the tesseract binary)
```

Parsers for PDF, HTML, Markdown, DOCX, text, CSV, and JSON are part of the core install, as is the chunking pipeline.

**Every embedding provider is an extra, including the local default.** `huggingface` is separate because sentence-transformers pulls a deep-learning runtime measured in gigabytes, and a deployment that embeds through a hosted provider should not have to install it. `fasterrag[huggingface]` is the fully-local starting point; selecting a provider whose extra is missing fails at startup with a `ConfigError` naming the exact install command, never with an import traceback.

OCR is optional for a second reason: it additionally requires the `tesseract` executable on the host. Without it a scanned page is flagged `low_text_yield` rather than silently indexed as empty.

- Requires **Python 3.12+**.
- Versioned by **SemVer 2.0.0**; the public API defined in this document is the compatibility contract — breaking it requires a major version bump and a CHANGELOG entry.
- Wheels published to PyPI at every tagged release; dependencies pinned and hash-locked in the repo.

## Quickstart

```python
import asyncio
from fasterrag import FasterRag

async def main() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        job = await rag.ingest(["./docs/handbook.md", "./docs/spec.md"])
        print(job.status, job.counts["indexed"])

        result = await rag.query("What does the spec say about retries?")
        print(result.answer)
        for citation in result.citations:
            print(citation.source, citation.page, citation.span)

# CRITICAL: the __main__ guard is required, not stylistic. Parsing runs in a
# ProcessPoolExecutor, and on Windows and macOS child processes are spawned rather than
# forked — each one re-imports this module. Calling asyncio.run() at import level makes
# every worker re-run the whole script instead of parsing, and the pool dies. Without the
# guard fasterRag raises CHUNK_FAILED naming this exact cause.
if __name__ == "__main__":
    asyncio.run(main())
```

`ingest` awaits completion and returns the settled job, rather than returning a job id the way `POST /v1/ingest` does. A library caller already has the one thing an HTTP client lacks: somewhere to wait.

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
