# flow.md — End-to-End Flows

All flows are documented intentions for the beta build. Stages marked **∥** are executed by parallel workers.

## 1. Ingestion flow (source → parse → chunk → embed → index)

```mermaid
flowchart LR
    SRC["Sources: files, dirs, URLs"] --> API["POST /v1/ingest (async accept, 202 + job_id)"]
    API --> Q1[("ingest queue")]
    Q1 --> P["∥ Parse (CPU pool): reading order, tables, headings, OCR, metadata"]
    P --> C["∥ Chunk (CPU pool): strategy from config.yaml"]
    C --> CE["∥ Contextual enrichment (optional): prepend ~50-100 token doc context"]
    CE --> Q2[("chunk queue (bounded)")]
    Q2 --> E["∥ Embed (GPU pool): stateful workers, batched"]
    E --> IDX["∥ Index: dense upsert + BM25 + metadata (dedup, versioning)"]
    IDX --> VDB[("Vector DB via adapter")]
    IDX --> BM25[("BM25 index")]
    IDX --> JOB["Job status: completed"]
```

Stage sequence: accept → enqueue → parse **∥** → chunk **∥** → enrich **∥** → embed **∥** → index **∥** → report. The API returns immediately after enqueue; all heavy stages run in worker pools.

## 2. Chunking/embedding/indexing worker hand-off (CPU pool → GPU pool streaming)

```mermaid
flowchart LR
    subgraph CPU["CPU worker pool (N workers) - CPU-bound"]
        W1["worker 1: load/parse/chunk"]
        W2["worker 2: load/parse/chunk"]
        WN["worker N: load/parse/chunk"]
    end
    subgraph GPU["Embedding worker pool (M workers) - GPU-bound, stateful"]
        G1["worker 1: model in memory (loaded once)"]
        G2["worker M: model in memory (loaded once)"]
    end
    W1 --> BQ[("bounded chunk queue - backpressure")]
    W2 --> BQ
    WN --> BQ
    BQ -->|"stream batches of chunk texts"| G1
    BQ -->|"stream batches of chunk texts"| G2
    G1 --> UP["batch upsert"]
    G2 --> UP
    UP --> VDB[("Vector DB")]
```

Key properties:

- **Streaming hand-off**: chunks flow to embedding workers as soon as they exist — expensive embedding workers never idle waiting for parsing to finish.
- **Stateful workers**: each embedding worker loads the model into memory once and reuses it across all batches (no reloading).
- **Fault isolation**: a failed embedding batch is retried from the queue; it never forces re-parsing (decoupled stages = fault tolerance).
- **Backpressure**: the bounded queue (`workers.queue_depth`) throttles CPU workers if embedding falls behind, preventing memory blow-up and provider rate-limit storms.

## 3. Query flow (query → hybrid retrieve → fuse → rerank → assemble → generate → stream)

```mermaid
flowchart TB
    Q["POST /v1/query"] --> QE["Embed query (embedding cache first)"]
    QE --> PAR{{"parallel retrieval legs"}}
    PAR --> D["∥ Dense ANN search (top rerank_top_n)"]
    PAR --> S["∥ Sparse BM25 search (top rerank_top_n)"]
    D --> F["Fuse: Reciprocal Rank Fusion, k=60"]
    S --> F
    F --> R["Cross-encoder rerank (top 100-1000 candidates, ~100-300 ms)"]
    R --> K["Truncate to top_k"]
    K --> CA["Context assembly: token budget, dedup, citations"]
    CA --> G["LLM generate (batched provider calls)"]
    G --> ST["Stream tokens via SSE (time-to-first-token)"]
    ST --> RESP["Response: text stream + citations + usage"]
```

Metadata filters (if provided) are pushed down into both retrieval legs before fusion.

## 4. Cache flow (semantic cache lookup → hit/miss → pipeline)

```mermaid
flowchart TB
    Q["Incoming query"] --> QE["Embed query (embedding cache)"]
    QE --> L["Semantic cache lookup: cosine similarity vs cached query embeddings"]
    L -->|"similarity >= threshold (~0.92-0.97)"| HIT["HIT: return cached response, mark cache_hit metric"]
    L -->|"below threshold"| MISS["MISS: run full query pipeline"]
    MISS --> PIPE["retrieve -> fuse -> rerank -> assemble -> generate"]
    PIPE --> STORE["Store (query embedding, response, TTL) in semantic cache"]
    STORE --> OUT["Return response"]
    CORPUS["Corpus change event (ingest/delete/reindex)"] -.->|"event-driven invalidation"| INV["Invalidate affected cache entries"]
    TTL["TTL expiry"] -.-> INV
    INV -.-> L
```

Both hit and miss increment the cache hit/miss metrics exposed to the dashboard and Grafana.

## 5. Auto-provisioning flow (config toggle → running URL, no code changes)

```mermaid
flowchart TB
    T["User flips toggle in config.yaml (e.g. observability.langfuse: true)"] --> RD["Read + validate config on startup or 'fasterrag provision'"]
    RD --> CHK["Check current state: already provisioned? healthy?"]
    CHK -->|"already running"| URL
    CHK -->|"not present"| INST["Auto-install: pull images, generate compose + secrets"]
    INST --> CFG["Configure: env vars, volumes, ports, headless bootstrap (e.g. LANGFUSE_INIT_*)"]
    CFG --> START["Start containers"]
    START --> HC["Health check until ready"]
    HC --> URL["Return running URL (e.g. http://host:3000)"]
    URL --> NOTE["NO application code changes at any point - pure config-driven provisioning"]
```

The same flow applies to `grafana: true` (provisioning-as-code datasources/dashboards) and `vector_db.mode: docker` (system-managed Qdrant). Provisioning is idempotent and highly optimized: repeated runs converge to the desired state without reinstalling.

## Where parallel workers act (summary)

| Stage | Executor | Parallelism |
|---|---|---|
| Ingestion accept | FastAPI (async) | Concurrent requests, non-blocking |
| Parse / chunk / enrich | CPU worker pool | `workers.cpu_pool_size` processes |
| Embedding | GPU/embedding pool | `workers.embedding_pool_size` stateful workers × batches |
| Indexing | Indexer workers | Batch upserts, concurrent per collection |
| Retrieval legs | Query service | Dense + BM25 run concurrently per query |
| Generation | LLM adapter | Batched calls; streamed output |
