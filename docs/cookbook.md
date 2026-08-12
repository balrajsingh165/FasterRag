# cookbook.md — Configuration Recipes

Ready-made `config.yaml` profiles for common situations. Each recipe shows **only the keys that differ from the defaults** ([config-reference.md](config-reference.md)) plus why each change earns its place. Recipes compose — read the trade-off note before stacking them.

Every recipe assumes the workflow in [quickstart.md](quickstart.md): validate → doctor → provision → estimate → ingest → query.

> Latency and cost characterizations below are **directional design rationale**, not measurements. Numbers only become claims via the [benchmark ledger](benchmarks.md) — measure on your own corpus and hardware.

---

## R1 — Fully local / air-gapped

Nothing leaves the machine: local embeddings, local generation, local vector DB. The privacy and compliance default.

```yaml
vector_db:
  provider: qdrant
  mode: docker
embeddings:
  provider: huggingface
  model: BAAI/bge-small-en-v1.5
llm:
  provider: ollama
  model: llama3.1
  base_url: http://localhost:11434
observability:
  otel: false
  langfuse: false          # provisioning pulls images — keep off on air-gapped hosts
```

**Why:** the only components that ever egress are hosted embedding/LLM providers; replacing both with local ones removes egress entirely. **Trade-off:** local generation quality and speed depend on your hardware; retrieval quality is unaffected.

## R2 — Maximum retrieval accuracy

When being right matters more than being fast or cheap.

```yaml
chunking:
  strategy: layout               # respect headings/tables/reading order
  chunk_size: 512                # smaller chunks = more precise matches
  overlap: 128
  contextual_enrichment: true    # −49% failed retrievals; −67% with reranking (references.md R1)
  context_tokens: 100
retrieval:
  hybrid: true
  rerank: true
  rerank_top_n: 300              # give the cross-encoder far more candidates
  top_k: 15
generation:
  grounded_or_refuse: true       # refuse rather than guess
  faithfulness_threshold: 0.75
eval:
  regression_gate: true          # never silently regress from here
```

**Why:** this stacks every quality lever — layout-aware chunking, enrichment, hybrid retrieval, a deep rerank pool, and grounding enforcement. **Trade-off:** enrichment costs an LLM call per chunk at ingest ([prompts.md](prompts.md) P2); `rerank_top_n: 300` materially increases per-query latency. Run `fasterrag estimate` before ingesting a large corpus with enrichment on.

## R3 — Cost-optimized at scale

Millions of documents, finite budget.

```yaml
chunking:
  strategy: recursive
  chunk_size: 1024               # fewer chunks = fewer embeddings and less storage
  overlap: 64
  contextual_enrichment: false   # a large ingest-time cost (one LLM call per chunk, unmeasured); enable per-collection if needed
embeddings:
  provider: huggingface          # local embeddings = zero per-token cost
  model: BAAI/bge-small-en-v1.5
  batch_size: 128
  cache:
    enabled: true
  tiering:
    enabled: true
    rules:
      - match: {priority_class: archive}
        provider: huggingface
        model: BAAI/bge-small-en-v1.5
      - match: {priority_class: critical}
        provider: openai
        model: text-embedding-3-large
cache:
  semantic: true                 # repeat questions never reach the LLM
  ttl: 86400
retrieval:
  rerank_top_n: 50
```

> ⚠️ **Token budgets now enforce, and the per-tenant one is per replica.** `cost.per_query_token_budget` and `cost.per_tenant_token_budget` are enforced before the provider is called (TASK-0242 ✅); a refused query gets `BUDGET_EXCEEDED` (402) rather than a degraded answer. The per-query budget is exact and identical on every replica. **The per-tenant budget is counted in-process**, like the per-key rate limiter (TASK-0216), so N replicas grant N times the configured budget — the error runs in the permissive direction, so size it for your replica count or run one. Budgets are checked against the prompt plus `llm.max_tokens`, the ceiling a call can reach, and a tenant is charged what the query actually cost. `fasterrag estimate` remains the preflight half.

**Why:** local embeddings remove the dominant variable cost; tiering spends only where precision pays; the semantic cache eliminates repeat generation. **Trade-off:** larger chunks and a shallower rerank pool cost some precision — pair with R7 to prove the loss is acceptable.

## R4 — Low latency / high throughput

Interactive UX, where time-to-first-token dominates perceived speed.

```yaml
llm:
  streaming: true                # TTFT decoupled from total generation time
  max_tokens: 512
retrieval:
  rerank: true
  rerank_top_n: 50               # the main latency dial in the retrieval path
  top_k: 8
cache:
  semantic: true
  similarity_threshold: 0.97     # conservative: only near-identical queries hit
  ttl: 3600
workers:
  cpu_pool_size: 8
  embedding_pool_size: 2
  queue_depth: 2000
reliability:
  timeouts:
    llm_ms: 30000                # fail to extractive fast rather than hang
  degradation_ladder: true
```

**Why:** streaming plus a tight rerank pool plus a conservative cache attacks latency without abandoning quality; a short LLM timeout means a stalled provider degrades in seconds instead of stalling the request. **Trade-off:** `rerank_top_n: 50` and `top_k: 8` reduce recall headroom on hard queries.

## R5 — Multi-tenant SaaS backend

Serving many customers from one deployment.

> ✅ **The security half of this recipe is enforced** (corrected 2026-08-09; the warning that stood here said the opposite and was two slices out of date). `security.auth` shipped with TASK-0046 ✅ and `security.multi_tenancy` with TASK-0179–0181 ✅: authentication, scopes, per-key rate limiting, and tenant isolation run as ASGI middleware, `max_request_mb` is enforced before the body is read (TASK-0199 ✅), and enabling auth without a usable key refuses to start.
>
> ⚠️ **Two real limits remain.** The per-key rate limiter is an **in-process fixed window**, so N replicas grant N times the configured limit (TASK-0216). And `cost.per_tenant_token_budget` is **not enforced by anything** — the runtime governor is unbuilt, so setting it raises `ConfigError` at startup (TASK-0242); it is omitted below for that reason.

```yaml
security:
  auth: true
  multi_tenancy: true
  tenant_header: X-Tenant-ID
  rate_limit_per_minute: 300
  max_request_mb: 10
cache:
  semantic: true                 # entries are tenant-scoped; a hit can never cross tenants
traces:
  store: true
  retention_days: 14             # bound how long tenant prompts persist
observability:
  dashboard: false               # shows prompts/responses — never expose to tenants
```

**Why:** isolation is enforced at the service layer, so collections, caches, traces, metrics, and ingestion jobs are all tenant-tagged ([security.md](security.md)) — budgets are not, because they do not exist yet. **Trade-off:** the dashboard is operator-only by construction — it renders every tenant's LLM I/O history, so it belongs on an internal network, never in a tenant-facing surface.

## R6 — Retrieval only (bring your own LLM)

Use fasterRag as the retrieval engine and keep generation in your own stack.

```yaml
retrieval:
  hybrid: true
  rerank: true
  top_k: 20                      # return more; your layer decides what to use
generation:
  citations: true
cache:
  semantic: false                # caching answers is meaningless when you generate them
```

```python
from fasterrag import FasterRag

async with FasterRag.from_config("config.yaml") as rag:
    chunks = await rag.retrieve("What are the payment terms?", top_k=20)
    # chunks carry dense_rank, bm25_rank, rrf_score, rerank_score, and full metadata
```

**Why:** `retrieve()` stops before generation, so you get the fused, reranked, cited chunk set and nothing else. **Trade-off:** grounded-or-refuse and faithfulness scoring live in the generation path — those guarantees become your responsibility.

## R7 — Quality locked into CI

Make retrieval regressions impossible to merge.

```yaml
eval:
  regression_gate: true
  recall_tolerance: 0.02
  ndcg_tolerance: 0.02
index:
  lockfile: true
  reindex:
    strategy: blue_green
    eval_gate: true              # a reindex that degrades quality never takes traffic
    rollback_retention_hours: 168
traces:
  store: true
  replay: true                   # explain any change after the fact
```

Then in CI: `fasterrag benchmark --suite eval` (exit 5 blocks the change). Build the golden set first with `fasterrag autopilot run`, review it, and promote records to `source: "human"` ([testing-strategy.md](testing-strategy.md) §1.6).

**Why:** the gate turns retrieval quality into a blocking check like type errors; the eval-gated alias swap extends the same protection to reindexing. **Trade-off:** the gate is only as good as the golden set — an unreviewed generated set lets the system grade its own homework.

## R8 — Very large corpus ingestion

Hundreds of GB, ingestion measured in hours or days.

```yaml
workers:
  cpu_pool_size: 0               # 0 = auto (CPU count)
  embedding_pool_size: 4
  queue_depth: 5000
embeddings:
  batch_size: 128
ingestion:
  dedup: true
  journal:
    enabled: true
    checkpoint_every: 500        # larger batches = less journal overhead
  dlq:
    enabled: true
    max_retries: 5
vector_db:
  mode: external                 # dedicated DB host
  host: 10.0.0.42
  collection:
    shard_number: 6
    replication_factor: 2
```

Run the API and workers as separate processes (`fasterrag serve` / `fasterrag worker`) so ingestion load can never starve live queries — the pools and queues are bulkheaded ([reliability.md](reliability.md)).

**Why:** checkpointing makes a multi-day ingest resumable rather than restartable; sharding and replication scale the index; a dedicated DB host removes resource contention. **Trade-off:** more moving parts — run `fasterrag doctor` on every host, and remember remote Qdrant needs both 6333 and 6334 reachable.

## R9 — Evaluating fasterRag before committing

Minimal, fast, no external dependencies — for a fair look on your own data.

> ⚠️ This recipe previously suggested `provider: chroma`. **Chroma is not built** (TASK-0049) and the loader refuses it by name, so the recipe most likely to be a reader's first config could not start. It now uses the reference backend, which `fasterrag provision qdrant` brings up for you.

```yaml
vector_db:
  provider: qdrant               # reference backend; `fasterrag provision qdrant` runs it for you
  mode: docker
embeddings:
  provider: huggingface
  model: BAAI/bge-small-en-v1.5
retrieval:
  hybrid: true
  rerank: false                  # add it later and measure the delta yourself
chunking:
  chunk_size: 768
autopilot:
  enabled: true
```

Then: ingest a representative slice → `fasterrag autopilot run` for a measured baseline → flip `retrieval.rerank: true` and re-run to see the delta on **your** corpus. **Why:** this is the honest way to evaluate — our defaults are informed by published evidence ([references.md](references.md)), but your corpus is the only benchmark that decides your configuration.

---

## Composing recipes

| Combination | Verdict |
|---|---|
| R1 + R2 | Excellent — maximum accuracy, fully local. Watch ingest time: enrichment runs through your local LLM. |
| R3 + R4 | Good — cheap and fast; verify recall with R7 before trusting it in production. |
| R5 + R3 | Natural pairing, and now complete: tenant isolation, preflight estimation, *and* a runtime per-tenant budget (TASK-0242 ✅). Size the budget for your replica count — it is counted per process. |
| R2 + R4 | Conflicting: `rerank_top_n` 300 vs 50 is the same dial pulled in opposite directions. Pick your point on the curve and measure it. |
| Any + R7 | Always worth adding. A tuned config with no regression gate degrades silently over time. |
