# data-model.md — Canonical Entity Schemas


> **Document ids are versioned.** `IDENTITY_VERSION` in `core/identity.py` records which scheme minted the ids in a collection, and `index.lock` stores it. Changing what `document_id` returns for the same source renames every document at once — an existing collection then scores recall 0.0, which reads as a broken retriever rather than a renamed corpus. The lockfile comparison reports it as drift instead, naming `identity_version`. A lockfile written before the field existed defaults to scheme **1**, never to the current value: defaulting to "current" would declare a stale index up to date and hide exactly the mismatch the field exists to surface. Scheme 2 (2026-08-03) normalises local paths so two spellings of one Windows file are a single id.

The single source of truth for every entity fasterRag persists or returns. The REST API ([api-reference.md](api-reference.md)), the Python package ([python-api.md](python-api.md)), and the portability archive ([archive-format.md](archive-format.md)) are all **projections of these entities** — where a field appears in more than one place, it means the same thing and carries the same type. Inventing a field that isn't here, or reusing a name with a different meaning, is a bug ([todo.md](todo.md)).

Conventions: types in Python/Pydantic notation · `str|null` = optional · all timestamps ISO 8601 UTC · all hashes SHA-256 hex · IDs are opaque strings with a typed prefix.

## ID scheme

| Prefix | Entity | Derivation |
|---|---|---|
| `d_` | Document | deterministic from `source_uri` + tenant |
| `c_` | Chunk | **deterministic** from `document_id` + chunk index + chunker config hash — this determinism is what makes indexing idempotent and replay-safe (D3) |
| `job_` | Ingest job | random, time-ordered |
| `col_` | Collection | random (the human-facing key is `name`) |
| *(none)* | Trace | a Trace's id **is** the propagated `trace_id`: a bare 32-hex lowercase OpenTelemetry trace id, identical in logs, spans, and problem responses — deliberately unprefixed for OTel compatibility (resolved AUDIT-0006, 2026-08-02) |
| `memver_`-style suffixes | — | not used; fasterRag entities use the prefixes above only |

## Document

The source unit as ingested. One document produces many chunks.

| Field | Type | Notes |
|---|---|---|
| `document_id` | str | `d_…` |
| `source_uri` | str | Path, URL, or inline reference — the stable identity of the source |
| `content_hash` | str | Hash of the raw bytes; drives deduplication (D3) and lockfile drift detection (D1) |
| `version` | int | Increments when the same `source_uri` is re-ingested with different content |
| `mime_type` | str | Detected at parse time |
| `size_bytes` | int | Raw size; bounded by `ingestion.max_document_mb` |
| `parsed_at` | str | Timestamp |
| `parser` | str | Parser used (e.g. `pdf-layout`, `ocr`) |
| `parse_flags` | list[str] | Quality signals, e.g. `low_text_yield`, `tables_detected`, `ocr_applied` |
| `metadata` | object | User metadata from ingest merged with extracted document metadata (title, author, dates) |
| `tenant` | str\|null | Set when `security.multi_tenancy: true` |
| `status` | str | `indexed` \| `deduplicated` \| `dead_lettered` \| `pending` |

## Chunk

The indexed retrieval unit. Everything retrieval and citation depends on lives here.

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | str | `c_…`, deterministic (see ID scheme) |
| `document_id` | str | Owning document |
| `text` | str | The text that is embedded and BM25-indexed. When contextual enrichment is on, this is `context_prefix + original_text` |
| `original_text` | str | The text without the enrichment prefix — what citations resolve against |
| `context_prefix` | str\|null | Generated document-level context (~50–100 tokens) when `chunking.contextual_enrichment: true` |
| `span` | `{start: int, end: int}` | Character offsets into the **parsed document**; monotonic and in-bounds (a chunker property-test invariant) |
| `page` | int\|null | 1-based page for paginated sources |
| `section` | str\|null | Heading path, e.g. `"3. Termination > 3.2 Notice"` |
| `token_count` | int | Tokens in `text` |
| `chunk_index` | int | 0-based position within the document |
| `embedding_model` | str | Model that embedded this chunk |
| `embedding_model_version` | str | Version — the drift-detection anchor (D1) |
| `chunker_strategy` | str | `fixed` \| `recursive` \| `semantic` \| `layout` \| `late` |
| `metadata` | object | Inherited document metadata + chunk-level additions; the filterable surface |
| `tenant` | str\|null | Mirrors the document |

## Collection

| Field | Type | Notes |
|---|---|---|
| `collection_id` / `name` | str | `col_…` / human-facing unique name |
| `distance` | str | `cosine` \| `dot` \| `euclid` |
| `dimensions` | int | Vector size; must match the embedding model |
| `shard_number` / `replication_factor` | int | Passed through the adapter |
| `vectors_count` / `documents_count` | int | Live counts |
| `alias_target` | str\|null | The physical collection this name currently points at — the blue/green swap target (D2) |
| `lock` | IndexLock | Embedded summary; see below |
| `drift` | `{detected: bool, fields: list[str]}` | Live comparison of config vs lock (D1) |
| `created_at` / `updated_at` | str | Timestamps |

## IndexLock (`index.lock`)

The reproducibility contract of an index (D1). Written atomically (write-temp-then-rename).

| Field | Type | Notes |
|---|---|---|
| `lock_version` | str | SemVer of the lockfile format |
| `collection` | str | Collection name |
| `config_hash` | str | Hash of the retrieval-affecting config subset (chunking, embeddings, retrieval) — **not** the whole file, so unrelated edits don't false-positive |
| `embedding_model` / `embedding_model_version` / `dimensions` | str / str / int | What produced the vectors |
| `chunker_strategy` / `chunker_version` / `chunk_size` / `overlap` | str / str / int / int | How chunks were produced |
| `contextual_enrichment` | bool | Whether chunks carry a context prefix |
| `document_hashes` | map[str, str] | `document_id → content_hash` for every indexed document |
| `built_at` / `built_by` | str | Timestamp / fasterRag version |

Drift = any mismatch between this file and live config/corpus. Reported, never silent.

## IngestJob

| Field | Type | Notes |
|---|---|---|
| `job_id` | str | `job_…` |
| `collection` | str | Target |
| `status` | str | `queued` \| `running` \| `completed` \| `failed` \| `partial` |
| `counts` | object | `total`, `parsed`, `chunked`, `embedded`, `indexed`, `deduplicated`, `dead_lettered` |
| `sources` | list[Source] | `{type: path\|url\|inline, value: str}` |
| `checkpoint` | `{last_document_index: int, written_at: str}` | Journal checkpoint — where a crash resumes from (D3) |
| `started_at` / `finished_at` | str\|null | Timestamps |
| `idempotency_key` | str\|null | Replay of the same key returns this job rather than creating a new one |
| `tenant` | str\|null | |

### DLQEntry

| Field | Type | Notes |
|---|---|---|
| `document_id` / `source` | str | What failed |
| `job_id` | str | Owning job |
| `reason_code` | str | Machine-readable, drawn from the error-code table (e.g. `PARSE_FAILED`, `EMBED_PROVIDER_ERROR`) |
| `detail` | str | Human-readable cause (never contains secret values) |
| `attempts` | int | Retries consumed of `ingestion.dlq.max_retries` |
| `first_failed_at` / `last_failed_at` | str | Timestamps |
| `trace_id` | str | Correlates to logs and spans |

## Query-side entities

### ScoredChunk

What retrieval returns internally and what `--show-chunks` / `include_chunks` exposes.

| Field | Type | Notes |
|---|---|---|
| `chunk` | Chunk | The retrieved chunk |
| `dense_rank` / `dense_score` | int\|null / float\|null | Dense leg position and raw score (`null` if the leg didn't return it) |
| `bm25_rank` / `bm25_score` | int\|null / float\|null | Sparse leg equivalents |
| `rrf_score` | float | Fused score, `Σ 1/(rrf_k + rank)` across legs |
| `rerank_score` | float\|null | Cross-encoder score; `null` when reranking is off or degraded to `hybrid_only` |
| `final_rank` | int | Position after truncation to `top_k` |

### Citation

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | str | Resolves to a real chunk |
| `source` | str | `source_uri` of the owning document |
| `page` | int\|null | |
| `span` | `{start: int, end: int}` | Offsets into `original_text` — span-level, not document-level (D5) |
| `score` | float | The chunk's final relevance score |

### QueryResult

| Field | Type | Notes |
|---|---|---|
| `answer` | str\|null | `null` when grounded-or-refuse declined (D5) |
| `citations` | list[Citation] | Mandatory and non-empty whenever `answer` is non-null and `generation.citations: true` |
| `chunks` | list[ScoredChunk]\|null | Present only when explicitly requested |
| `usage` | `{prompt_tokens, completion_tokens, estimated_cost_usd}` | `estimated_cost_usd` is a **list-price estimate, nullable by design** (TASK-0260 ✅): a model with no recorded rate reports `null` rather than `0.0`, because a zero would read as "this query was free" and that is a fabricated bill. A local provider reports a genuine `0.0`. Also counted into `fasterrag_cost_usd_total`, but emitted per response too — a caller holding one answer cannot read a process-wide counter. |
| `timings_ms` | object | Per stage: `embed`, `retrieve`, `fuse`, `rerank`, `assemble`, `generate`, `total` |
| `degraded` | bool | True whenever `mode != "full"` |
| `mode` | str | `full` \| `hybrid_only` \| `cache_only` \| `extractive` (D4) |
| `faithfulness` | float\|null | 0–1 grounding score |
| `insufficient_evidence` | bool | True when the answer was withheld below threshold |
| `cache` | `{semantic_hit: bool, similarity: float\|null}` | |
| `trace_id` | str | |

## Trace

Persisted for every query when `traces.store: true`; the substrate for replay (D8) and the dashboard's LLM I/O history.

| Field | Type | Notes |
|---|---|---|
| `trace_id` | str | bare 32-hex lowercase OTel trace id; identical to the value in logs, spans, and problem responses |
| `query` | str | Original query text |
| `filters` | object\|null | Metadata filters applied |
| `config_snapshot` | object | The retrieval-affecting config subset at execution time — what makes replay meaningful |
| `retrieved` | list[ScoredChunk] | Full candidate set with all leg scores, pre- and post-rerank ranks |
| `prompt` | str | Exact assembled prompt sent to the LLM |
| `response` | str | Exact response received |
| `result` | QueryResult | The returned result, minus `chunks` |
| `spans` | list[Span] | `retrieval`, `reranker`, `context-assembly`, `generation` — each `{name, start_ms, end_ms, attributes}` |
| `created_at` | str | Retained for `traces.retention_days` |

## Invariants (asserted by tests)

1. `chunk_id` is a pure function of `(document_id, chunk_index, chunker config hash)` — the same input always yields the same id, which is what makes upserts idempotent.
2. Every `Citation.chunk_id` resolves to a chunk that was in the query's retrieved set — citations can never reference something not retrieved.
3. `Chunk.span` offsets are monotonic across `chunk_index` and in-bounds for the parsed document (property-tested for every chunker).
4. `QueryResult.degraded == (mode != "full")` — the flag and the mode can never disagree.
5. `IndexLock.document_hashes` covers exactly the documents with `status: indexed` in that collection.
6. Nothing in any entity ever holds a secret value; provider keys appear only as env-var **names** in config.
