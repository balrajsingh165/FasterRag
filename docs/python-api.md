# python-api.md — Python Package (Library Surface)

fasterRag is released as an installable Python package so applications can **import and use the framework in-process**, without running the HTTP service. The library is the third programmatic control surface alongside the [REST API](api-reference.md) and the [CLI](cli-reference.md) — all three call the identical service layer, so behavior, config, and errors are the same everywhere. **There is still no GUI control surface**; the dashboard remains observability-only.

> **Per-surface status against the implemented package** (build tasks in [todo.md](todo.md)):
>
> | Surface in this document | Status |
> |---|---|
> | Standalone components — `fasterrag.parsing`, `.chunking`, `.retrieval`, `.rerank`, `.evals` | **Shipped** and importable today |
> | Typed error taxonomy — `fasterrag.errors` (same `code`s as the API) | **Shipped** |
> | `FasterRag` facade — `from_config`, `from_settings`, `ingest`, `query`, `query_stream`, `retrieve`, `estimate`, `index_lock` | **Shipped**. Verified end to end against live Qdrant and OpenAI — a one-off manual run, not a committed test, since a unit run must not depend on a credential or spend money (the standing gap is TASK-0205) |
> | `FasterRag.doctor`, `.collections`, `.create_collection`, `.drop_collection`, `.documents`, `.delete_document`, `.replay` | **Shipped** on both facades |
> | `FasterRag.vector_db`, `.embeddings`, `.cache`, `.journal` | **Shipped** on the async facade only. They hand back the live adapter for callers that need the backend directly; the sync facade has no counterparts, because an async adapter driven from the blocking facade's managed loop is not a thing a caller can hold |
> | `FasterRag.export_archive` / `.import_archive` | **Not yet wrapped on the facade** (TASK-0079). The archive itself **is** built — `services/archive.py`, `services/archive_import.py`, the `export`/`import` CLI commands, and `POST /v1/admin/export` / `/import` all ship, and a Qdrant round trip is verified against a live backend (TASK-0214 ✅). Only the facade methods are missing |
> | `fasterrag.sync` blocking facade | **Shipped**. Verified end to end against live Qdrant and OpenAI — a one-off manual run, not a committed test, since a unit run must not depend on a credential or spend money (the standing gap is TASK-0205) |
> | Entry-point plugin groups (`fasterrag.vectordb` / `.embeddings` / `.llm`) | **Shipped**. All three factories resolve registered plugins; a built-in name always wins, so an installed package cannot silently take over a configured provider |
> | PyPI wheels (`pip install fasterrag`) | **Not yet published** (TASK-0087) — install from source: `pip install -e ".[all]"` |
>
> Sections describing an unshipped surface are the design contract the remaining work must satisfy, kept here so each one is built to a reviewed spec rather than improvised.

## Installation

```bash
pip install fasterrag                         # core: the library, the CLI, chunking, BM25 + fusion,
                                              #   Markdown/text/CSV/JSON parsing
pip install "fasterrag[chroma,huggingface]"   # the local profile: no service, no key (see below)
pip install "fasterrag[server]"               # `fasterrag serve`: FastAPI + uvicorn
pip install "fasterrag[qdrant]"               # the Qdrant adapter (the default config.yaml uses it)
pip install "fasterrag[pdf,docx,html]"        # the PDF, DOCX and HTML parsers
pip install "fasterrag[all]"                  # every optional adapter/provider
pip install "fasterrag[huggingface]"          # local sentence-transformers embeddings
pip install "fasterrag[openai]"               # extras per provider: openai, anthropic, cohere, ollama
pip install "fasterrag[milvus]"               # extras per vector DB: qdrant, milvus, weaviate, pgvector, chroma
pip install "fasterrag[rerank]"               # cross-encoder reranker models
pip install "fasterrag[ocr]"                  # OCR for scanned PDFs (also needs the tesseract binary)
```

**The core install is the library, not the web service** (TASK-0444). It holds configuration, chunking, fusion, the facade, the CLI, and the Markdown, plain-text, CSV and JSON parsers. The web server (`server`), the Qdrant client (`qdrant`) and the PDF, DOCX and HTML parsers (`pdf`, `docx`, `html`) are extras, each imported only where it is used: `fasterrag serve` without `server` exits `2` naming `pip install "fasterrag[server]"`; selecting `vector_db.provider: qdrant` without `qdrant` is a `ConfigError` naming `pip install "fasterrag[qdrant]"`; and a PDF, DOCX or HTML document without its extra is dead-lettered with `PARSE_FAILED` and a detail naming the extra, while the rest of the job is ingested. The canonical `config.yaml` selects Qdrant, so using it as written needs `fasterrag[qdrant]`; `fasterrag config init --profile local` writes a configuration that needs only `fasterrag[chroma,huggingface]`. The container image installs `server`, `qdrant`, `pdf`, `docx` and `html` ([deployment.md](deployment.md)).

**Every embedding provider is an extra, including the local default.** `huggingface` is separate because sentence-transformers pulls a deep-learning runtime measured in gigabytes, and a deployment that embeds through a hosted provider should not have to install it. `fasterrag[huggingface]` is the fully-local starting point; selecting a provider whose extra is missing fails at startup with a `ConfigError` naming the exact install command, never with an import traceback.

OCR is optional for a second reason: it additionally requires the `tesseract` executable on the host. Without it a scanned page is flagged `low_text_yield` rather than silently indexed as empty.

- Requires **Python 3.12+**.
- Versioned by **SemVer 2.0.0**; the public API defined in this document is the compatibility contract — breaking it requires a major version bump and a CHANGELOG entry.
- Dependencies pinned and hash-locked in the repo (`uv.lock`, gated by `uv lock --check` in CI). Publishing wheels to PyPI at every tagged release is the **intended** release process; **nothing has been published yet** (TASK-0087) and there has been no tagged release.

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

**`.env` is read only when a caller asks for it (TASK-0341, 2026-11-09).** `FasterRag.from_config()` asks, so the walkthrough above is unchanged. The lower-level `load_settings()` no longer does: it used to copy the dotenv file into the real `os.environ` permanently, process-wide, as a side effect of being called, which is how a `.env` key reached a billed test run. A caller that loads settings itself and then builds components — `load_settings(...)` followed by `FasterRag.from_settings(...)` — must now pass `load_settings(..., export_env=True)`, because adapters resolve credentials through `os.environ`. Forgetting it is loud, not silent: the load fails with a `ConfigError` naming the variable and saying that the dotenv file was not exported.

## Public API surface (beta contract)

Everything a module's `__all__` exports is public and stable; anything else is private and may change without notice, whatever its name suggests. The top-level namespace exports exactly `FasterRag` and `__version__`.

**Every signature in the table below is checked against the shipped facade** by `tests/unit/docs/test_python_api_parity.py` (TASK-0368), which resolves each name with `importlib` and compares it with `inspect.signature` rather than comparing text with text. Drift fails in both directions: a row for a member that does not exist sends a caller into an `AttributeError`, and a public member with no row is stable for nobody.

### `FasterRag`

The async facade. Every **method** here has a blocking counterpart of the same name and parameters on the [sync facade](#sync-facade); the four adapter properties do not, because an async adapter driven from the blocking facade's managed loop is not a thing a caller can hold.

| Member | Signature | Behavior |
|---|---|---|
| `from_config` | `(path: str \| Path = "config.yaml") -> FasterRag` | Classmethod. Load + validate config and build adapters via factories. No I/O to backends yet, and `.env` is exported for the adapters that read credentials from the environment. |
| `from_settings` | `(settings: Settings) -> FasterRag` | Classmethod. Construct from an in-memory validated `Settings` object, for apps that manage their own config. |
| `__aenter__` / `__aexit__` | async context manager | Startup: connect adapters, run health checks, start in-process worker pools sized per `workers.*`. Shutdown: close each backend independently, so one refusing to shut down cleanly does not strand the others. |
| `settings` | attribute | The validated `Settings` this instance was built from. |
| `ingest` | `(sources: Sequence[str], *, collection: str \| None = None, metadata: dict[str, Any] \| None = None, tenant: str \| None = None, idempotency_key: str \| None = None, recursive: bool = False) -> JobRecord` | Awaits completion and returns the settled job, rather than returning a job id the way `POST /v1/ingest` does. Journaled, deduplicated, DLQ on failure. A directory in `sources` is replaced by the visible files inside it, exactly as `fasterrag ingest` does (`recursive=True` descends); `POST /v1/ingest` deliberately does **not** expand directories, so a served deployment is never handed a server-side directory walk. A document is recorded as indexed only after every one of its chunks is written to the vector database. |
| `query` | `(text: str, *, collection: str \| None = None, top_k: int \| None = None, filters: Mapping[str, Any] \| None = None, tenant: str \| None = None) -> Answer` | Full pipeline: hybrid retrieve → RRF → rerank → assemble → generate. With `tenant`, the query runs on that tenant's behalf, as a REST request authenticated as that tenant does: retrieval is limited to its chunks when `security.tenant_scoped_retrieval` is on (and refused with `TENANT_FORBIDDEN` without one), and the token budget and semantic cache are kept per tenant. Each call builds its own generation service, so concurrent queries for different tenants share no state; they share only the instance's one LLM client. |
| `query_stream` | `(text: str, *, collection: str \| None = None, top_k: int \| None = None, filters: Mapping[str, Any] \| None = None, tenant: str \| None = None) -> AsyncIterator[QueryEvent]` | Streaming variant, mirroring the SSE contract event for event. `tenant` as for `query`. |
| `retrieve` | `(text: str, *, collection: str \| None = None, top_k: int \| None = None, filters: Mapping[str, Any] \| None = None, tenant: str \| None = None) -> list[ScoredChunk]` | Retrieval only — no generation, for apps that bring their own LLM step. Reranking follows `retrieval.rerank.*` rather than a per-call flag. `tenant` as for `query`. |
| `collections` | `() -> list[CollectionInfo]` | What the configured vector DB holds, as the adapter reports it. |
| `create_collection` | `(name: str, *, distance: str \| None = None, shard_number: int = 1, replication_factor: int = 1) -> None` | Create a collection sized to the configured embedding dimensions. |
| `drop_collection` | `(name: str) -> bool` | Delete a collection; `False` when there was none to delete. |
| `documents` | `(collection: str \| None = None, *, offset: int = 0, limit: int = 100) -> DocumentListing` | One page of the documents indexed in a collection (the configured default when `collection` is omitted), ordered by `document_id`; follow `DocumentListing.next_offset` until it is `None`. `limit` is 1 to 1000, and anything else, or a negative `offset`, raises `VALIDATION_FAILED`. The listing is the journal's record of what was indexed — the record `delete_document` tombstones, so a deleted document is not listed — not a scan of the vector database: documents indexed with `ingestion.dedup: false` or by an archive import are not listed. A collection the vector database does not hold raises `NOT_FOUND`. The same listing `GET /v1/collections/{name}/documents` and `fasterrag documents list` return (TASK-0469). |
| `delete_document` | `(document_id: str, *, collection: str \| None = None, tenant: str \| None = None) -> DocumentDeletion` | Delete one document from a collection (the configured default when `collection` is omitted): every chunk carrying that `document_id`, its dedup record — so the same source can be ingested again — its `index.lock` entry, and this instance's semantic cache. **Idempotent**: a document nothing holds is not an error, and `DocumentDeletion.known` says whether the journal held a live record of it; the number of chunks removed is not reported, because no adapter can count what a filter matched. A collection that does not exist raises `NOT_FOUND`. With `tenant`, only chunks whose `tenant` payload tag is that id are removed. The `document_id` is `DocumentRecord.document_id` from `await rag.journal.documents(job_id)`. |
| `estimate` | `(sources: Sequence[str], *, all_providers: bool = False) -> Estimate` | D9 preflight, and the one call on this table that is not a coroutine: it parses and counts locally and reaches no provider. Raises `FasterRagError` (`VALIDATION_FAILED`) when `cost.estimator` is false — the library obeys the same switch as the CLI and the API. |
| `doctor` | `() -> DoctorReport` | D10 diagnostics; each check carries its own `passed` flag and `fix` string. |
| `replay` | `(trace_id: str, candidate: Settings \| None = None) -> ReplayResult` | D8 side-by-side replay of a journaled trace, optionally against a candidate configuration. |
| `index_lock` | `(collection: str \| None = None) -> IndexLock \| None` | The identity an index was built under, or `None` when the collection has never been written. |
| `vector_db` | property | The live `VectorDBAdapter`. Every property and method here raises `USAGE_ERROR` outside `async with`, naming the context manager, rather than failing later somewhere unrelated. (It was `CONFIG_INVALID`, a 500-class code for a caller's mistake, until TASK-0461.) |
| `embeddings` | property | The live `TieringRouter` over the configured embedding providers. |
| `cache` | property | The live `SemanticCache`. |
| `journal` | property | The live `Journal`, for reading job and trace records directly. Its operations are **coroutines** — `await journal.load_job(job_id)` — each running the matching `*_blocking` twin in a worker thread so journal I/O never holds the caller's event loop (TASK-0399). The twins are public and are what a synchronous caller or a test fixture uses. |

#### Documented as deliberately absent

Specified, built everywhere else, and deliberately not wrapped on the facade yet. These are **not** rows in the table above, because a row is a promise a caller may call today. The parity gate reads this table as its own third state: each entry must still be missing from both facades and must still cite the task that closes it, so the entry cannot outlive the gap it records.

| Absent member | Closed by | Why it is absent |
|---|---|---|
| `export_archive` | TASK-0079 | D11 portability is built — `services/archive.py`, the `export` CLI command and `POST /v1/admin/export` all ship, and a Qdrant round trip is verified against a live backend (TASK-0214 ✅). Only the facade wrapper is missing. |
| `import_archive` | TASK-0079 | As above, against `services/archive_import.py`, the `import` CLI command and `POST /v1/admin/import`. |

### Result types

Dataclasses rather than Pydantic models: they leave the service layer already validated, and a second validation pass on every answer buys nothing. Each list below is the type's complete field set, computed properties included; the parity gate fails on a field this document invents and on one it omits.

Every one of them is frozen except `JobRecord`, which a running job is updated in place as it progresses — so a caller holding one is holding a view that can move under it, and should copy what it needs.

- `Answer` (`fasterrag.services.generation`) — `answer`, `citations`, `mode`, `trace_id`, `prompt_tokens`, `completion_tokens`, `timings_ms`, `faithfulness`, `threshold`, `best_candidates`, `cache`, `truncated`, `estimated_cost_usd`, `evidence_dropped`, and the computed `degraded`, `insufficient_evidence`.
- `Citation` (`fasterrag.services.generation`) — `chunk_id`, `source`, `page`, `span`, `score`.
- `QueryEvent` (`fasterrag.services.generation`) — `type`, `data`.
- `ScoredChunk` (`fasterrag.retrieval`) — `chunk_id`, `text`, `payload`, `dense_rank`, `dense_score`, `bm25_rank`, `bm25_score`, `rrf_score`, `rerank_score`, `final_rank`, and the computed `document_id`, `source`, `found_by_both_legs`.
- `JobRecord` (`fasterrag.services.journal`) — `job_id`, `collection`, `status`, `counts`, `sources`, `checkpoint`, `started_at`, `finished_at`, `idempotency_key`, `tenant`, `reason_code`.
- `Estimate` (`fasterrag.services.estimation`) — `documents`, `unreadable`, `chunks`, `tokens`, `bytes_read`, `providers`, `parse_seconds`, `projected_seconds`, `projection_note`, `enrichment`.
- `DoctorReport` (`fasterrag.services.doctor`) — `checks`, and the computed `passed`, `failures`.
- `ReplayResult` (`fasterrag.services.replay`) — `trace_id`, `query`, `config_changes`, `retrieval`, `original_answer`, `replayed_answer`, `original_citations`, `replayed_citations`, and the computed `answer_changed`, `deterministic`.
- `CollectionInfo` (`fasterrag.adapters.vectordb.base`) — `name`, `vectors`, `dimensions`, `distance`, `sparse`.
- `DocumentDeletion` (`fasterrag.services.documents`) — `collection`, `document_id`, `known`, `deleted_at`.
- `DocumentListing` (`fasterrag.services.documents`) — `collection`, `documents`, `total`, `offset`, `limit`, and the computed `next_offset`.
- `ListedDocument` (`fasterrag.services.documents`) — `document_id`, `content_hash`.
- `IndexLock` (`fasterrag.services.lockfile`) — `collection`, `config_hash`, `embedding_model`, `embedding_model_version`, `dimensions`, `chunker_strategy`, `chunker_version`, `chunk_size`, `overlap`, `contextual_enrichment`, `identity_version`, `document_hashes`, `built_at`, `built_by`, `lock_version`.

D5's refusal is `Answer.insufficient_evidence`, and the answer text is `None` when it fires.

### Exceptions

The full typed taxonomy from [reliability.md](reliability.md) is importable:

```python
from fasterrag.errors import (
    FasterRagError, ConfigError, IngestionError, ParseError, ChunkError,
    EmbedError, RetrievalError, GenerationError, ProviderError, CacheError,
    ProvisioningError,
)
from fasterrag.errors import ErrorCode, ProblemSpec, problem_spec, PROBLEM_SPECS, PROBLEM_TYPE_BASE
```

Every exception carries `code` (identical to the API's problem `code`), `trace_id`, and `retryable`. Library and API surface the same error identities — a wrapper service can pass `code` straight through. The second line is the problem table itself: `ErrorCode` is the enum of stable codes, `problem_spec` and `PROBLEM_SPECS` map a code to its RFC 9457 `ProblemSpec` (HTTP status, title, retryability), and `PROBLEM_TYPE_BASE` is the URI prefix those `type` fields are built on. A wrapper service that renders its own problem documents reads them from here rather than restating them.

### Sync facade

For scripts and notebooks that are not async:

```python
from fasterrag.sync import FasterRag

with FasterRag.from_config("config.yaml") as rag:
    rag.ingest(["./docs/handbook.md"])
    print(rag.query("What does the spec say about retries?").answer)

    for event in rag.query_stream("Summarise the handbook"):
        if event.type == "token":
            print(event.data["text"], end="")
```

The sync facade wraps the async engine in a managed event loop; it is a thin adapter, not a second implementation. Three consequences worth knowing:

- **It owns its event loop**, created on `__enter__` and closed on `__exit__`. `asyncio.run` per call would tear down the connection pools between calls, and borrowing an ambient loop would make behavior depend on what the caller happened to be running.
- **It works inside a running event loop — a Jupyter notebook, or code under `asyncio.run`** (TASK-0497). The thread already running that loop cannot also drive this facade's loop, and blocking it on a future its own loop must resolve deadlocks, so the first call made under a running loop moves the facade's private loop onto a dedicated daemon thread (`fasterrag-sync-loop`) and every call is handed to it; the caller blocks while its own loop stays where it was. `__exit__` stops that thread and joins it. The one call still refused is a call made *from that dedicated thread* — from code the facade itself is running — which raises `USAGE_ERROR` and names the async facade, because it would wait on itself forever. In async code, `from fasterrag import FasterRag` with `async with` remains the natural fit: the blocking facade blocks the caller's loop for the duration of each call.
- **`query_stream` stays incremental.** Each event is pulled individually rather than the answer being buffered and replayed, so the first token arrives long before the last. Buffering would make the return type honest and the feature pointless. (A one-off developer-machine run on 2026-08-03 saw the first token well ahead of a 51-event response; that is an illustration, not a ledger entry — no hardware, dataset, or repetitions, so no time-to-first-token figure is published here.)

The same `__main__` guard requirement applies here as in the async quickstart: ingestion parses in worker processes.

## Standalone components (use the pieces without the pipeline)

Every pipeline stage is importable and usable on its own, so applications can adopt fasterRag piecemeal — just the chunkers, just hybrid fusion, just the evaluator — before (or instead of) adopting the full engine:

```python
from fasterrag.parsing import parse_document            # -> ParsedDocument (reading order, tables, headings, metadata)
from fasterrag.chunking import RecursiveChunker, SemanticChunker, LayoutChunker
from fasterrag.retrieval import rrf_fuse                # Reciprocal Rank Fusion, k=60 default
from fasterrag.rerank import CrossEncoderReranker
from fasterrag.evals import evaluate                    # recall@k, MRR, nDCG

chunks = RecursiveChunker(chunk_size=768, overlap=64).split(parse_document("spec.pdf"))
fused = rrf_fuse(dense_ranking, bm25_ranking, k=60)

async def score() -> None:
    report = await evaluate(golden_set, retriever=my_retriever)
```

Parsing, chunking and fusion are synchronous; `evaluate` runs the retriever it is given, so it is a coroutine and must be awaited. Calling it without `await` hands back a coroutine object rather than a report, which is why the parity gate checks the `await` in this very snippet against `inspect.iscoroutinefunction`.

Each standalone component obeys the same contracts (typed errors, docstrings, tested invariants) as the full pipeline. A standalone parser needs no `config.yaml`: `parse_document`/`parse_bytes` apply the shipped defaults and accept `options=ParsingOptions(...)` to override the [`parsing`](config-reference.md#parsing) thresholds — the OCR trigger and DPI, the PDF heading rules, and CSV row grouping. Inside the pipeline the same object is built from configuration by `create_parsing_options(settings)`.

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
    return {"answer": result.answer, "citations": [c.as_dict() for c in result.citations]}
```

Deployment note: ingestion runs in worker pools inside the process that holds the facade, sized by `workers.*`. That is the only ingestion path: a dedicated worker process fed by a queue backend, which would take heavy ingestion off this process, is not built (TASK-0130), and there is no command to start one ([deployment.md](deployment.md)).

## Extending the framework (plugin contract)

Third parties add providers by implementing the adapter base classes and registering them via Python entry points — no fork required:

> **Status (2026-11-12): vector databases, embedders and LLMs are selectable by name; chunkers, parsers and rerankers are not.** A name registered under `fasterrag.vectordb`, `fasterrag.embeddings` or `fasterrag.llm` validates as `vector_db.provider`, `embeddings.provider` (and a tier rule's `provider`) or `llm.provider`, and the matching factory builds it. Validation reads the installed distributions' entry-point metadata and imports nothing, so a plugin's module is loaded only when its adapter is built. A name that is neither built in nor registered still stops the load, and the error lists the built-in names and the plugins that are installed. `tests/unit/test_plugin_selection.py` proves the whole path: it installs a fake distribution, loads a `config.yaml` naming it through the real loader, ingests a document and answers a query. Chunkers, parsers and rerankers have no entry-point group yet (TASK-0513).

```toml
[project.entry-points."fasterrag.vectordb"]
mydb = "my_pkg.adapter:MyDBAdapter"
```

Each line below names the entry-point group, the base class and every abstract member a conformant plugin must implement. The lists are compared with the base class's own abstract-method set, because a plugin author implements what this document says and a name missing here is a plugin that will not instantiate.

- `fasterrag.vectordb` → subclass `VectorDBAdapter` (`from fasterrag.adapters.vectordb.base import VectorDBAdapter`): `alias_target`, `close`, `create_collection`, `delete`, `delete_alias`, `delete_snapshot`, `drop_collection`, `health`, `iterate_points`, `list_collections`, `list_snapshots`, `restore_snapshot`, `search`, `set_alias`, `snapshot`, `update`, `upsert`
- `fasterrag.embeddings` → subclass `EmbeddingAdapter` (`from fasterrag.adapters.embeddings.base import EmbeddingAdapter`): `close`, `dimensions`, `embed_documents`, `embed_query`, `health`, `model`, `model_version`
- `fasterrag.llm` → subclass `LLMAdapter` (`from fasterrag.adapters.llm.base import LLMAdapter`): `close`, `complete`, `health`, `stream`

A registered provider becomes selectable in `config.yaml` (`vector_db.provider: mydb`) and MUST pass the shared **adapter contract test suite** ([testing-strategy.md](testing-strategy.md)) to be considered conformant.

**Credentials.** A plugin reads its key through the name in its section's `api_key_env`, exactly as a built-in does, and never inherits a built-in's default: for a plugin provider `llm.api_key_env` and `vector_db.api_key_env` must be written in `config.yaml`, as a variable name or as `null`, or the load stops ([config-reference.md](config-reference.md), cross-field rule 3). Configuration is validated without loading the plugin, so it cannot know whether a key is needed — a plugin that needs one says so when it is built, by calling `require_key` (`fasterrag.adapters.embeddings.base`) or `require_llm_key` (`fasterrag.adapters.llm.base`) with its `api_key_env`, which raises a `ConfigError` naming the missing variable.

**Refusals that say what to do.** An embedding plugin that classifies its own failures can raise two subclasses of `EmbedError` from `fasterrag.adapters.embeddings.base` to get the treatment a built-in gets: `EmbedTooLargeError(detail)` when the provider refused a request for its size, which makes the embedding pool halve the batch rather than retry it, and `EmbedThrottledError(detail, retry_after_seconds=...)` for a retryable refusal that named a pause, which the pool waits out up to `reliability.retries.backoff_max_ms`. `classify_provider_failure` produces both from a vendor exception's `status_code`, message and response headers (TASK-0478).

**The `provider` attribute.** `EmbeddingAdapter` and `LLMAdapter` declare a class attribute `provider` and give it no default; the pipeline reads it to label metrics, attribute cost and name circuit breakers. Set it on the subclass to the name the entry point registers — it is not an abstract member, so a plugin without it instantiates and then fails at its first embedding or completion.

```yaml
# config.yaml — the only change needed to select an installed plugin
vector_db:
  provider: mydb
  api_key_env: MYDB_API_KEY   # or null; never left to the Qdrant default
```

## Stability guarantees

| Surface | Guarantee |
|---|---|
| `fasterrag` top-level exports, `fasterrag.errors`, `fasterrag.sync` | SemVer-stable public API. |
| Adapter base classes + entry-point group names | SemVer-stable extension contract. |
| Standalone components — `fasterrag.parsing`, `.chunking`, `.retrieval`, `.rerank`, `.evals` | Shipped and importable, but **not yet** under the SemVer promise: they are listed separately because promising them means documenting every name they export, which is open work, not a decision already taken. |
| Anything a module's `__all__` does not export | No guarantees. |
| Result-type fields | Additive-only within a major version. |
