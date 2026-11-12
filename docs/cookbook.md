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
  chunk_size: 512                # smaller chunks narrow the span each match returns
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
  chunk_size: 1024               # larger chunks mean fewer chunks: fewer embeddings and less storage
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
```

**Why:** streaming plus a tight rerank pool plus a conservative cache attacks latency without abandoning quality; a short LLM timeout means a stalled provider degrades in seconds instead of stalling the request. **Trade-off:** `rerank_top_n: 50` and `top_k: 8` reduce recall headroom on hard queries.

## R5 — Multi-tenant SaaS backend

Serving many customers from one deployment.

> ✅ **The security half of this recipe is enforced** (corrected 2026-08-09; the warning that stood here said the opposite and was two slices out of date). `security.auth` shipped with TASK-0046 ✅ and `security.multi_tenancy` with TASK-0179–0181 ✅: authentication, scopes, per-key rate limiting, and tenant isolation run as ASGI middleware, `max_request_mb` is enforced before the body is read (TASK-0199 ✅), and enabling auth without a usable key refuses to start.
>
> ⚠️ **Retrieval is not tenant-scoped unless you say so (TASK-0339, 2026-11-09).** `multi_tenancy: true` scopes collections, caches, traces, jobs and metrics — it does **not** scope the query path. Without the `tenant_scoped_retrieval` line below, a valid key for one tenant retrieves another tenant's chunks from the shared collection, and `POST /v1/query` puts them in the answer; this was observed, not inferred ([security.md](security.md) §3). The flag is off by default because it changes what an existing corpus answers — backfill the `tenant` payload field with `fasterrag index backfill-tenant` (run `fasterrag index tenants` first to see how many points carry no tag) and add a backend payload index on it **before** turning it on; the ordering is in [security.md](security.md) §3.
>
> ⚠️ **Two real limits remain.** The per-key rate limiter is an **in-process fixed window**, so N replicas grant N times the configured limit (TASK-0216). And `cost.per_tenant_token_budget` **is** enforced now (TASK-0242 ✅, `services/budget.py`: `BUDGET_EXCEEDED`/402 before the provider is called) — but its counter is in-process, like the rate limiter above, so N replicas grant N times the budget; it is omitted below only because sizing it correctly depends on your replica count.

```yaml
security:
  auth: true
  multi_tenancy: true
  tenant_header: X-Tenant-ID
  tenant_scoped_retrieval: true  # read the warning above before enabling on existing data
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

**Why:** isolation is enforced at the service layer, so collections, caches, traces, metrics, and ingestion jobs are all tenant-tagged ([security.md](security.md)), and `tenant_scoped_retrieval` extends that to the one surface it did not cover — retrieval itself. **Trade-off:** the dashboard is operator-only by construction — it renders every tenant's LLM I/O history, so it belongs on an internal network, never in a tenant-facing surface.

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

Ingestion runs inside the API process, so size that process for both loads. A separate worker process that takes ingestion load off the API, fed by a queue backend, is not built (TASK-0130). What exists today is `fasterrag ingest`, which runs the job inline in its own process: started on another host against the same vector database it keeps parsing and embedding off the API host, but its job lives in that host's journal, not in the API's `GET /v1/ingest/{job_id}`.

**Why:** checkpointing makes a multi-day ingest resumable rather than restartable; sharding and replication scale the index; a dedicated DB host removes resource contention. **Trade-off:** more moving parts — run `fasterrag doctor` on every host, and remember remote Qdrant needs both 6333 and 6334 reachable.

## R9 — Evaluating fasterRag before committing

Minimal, fast, no external dependencies — for a fair look on your own data.

> Chroma shipped on 2026-11-03 and is now a genuine zero-install option: it runs embedded against a local directory, with no server and no Docker. Install it with `pip install "fasterrag[chroma]"` and set `provider: chroma`. The keyword (BM25) leg is computed in-process, so it reads every posting a query's terms match — fine for a first look at your own data, not for a large corpus; see [integrations.md](integrations.md). The reference backend below stays the recommendation for anything beyond an evaluation.

```yaml
vector_db:
  provider: qdrant               # reference backend; `fasterrag provision qdrant` runs it for you
  mode: docker
embeddings:
  provider: huggingface
  model: BAAI/bge-small-en-v1.5
retrieval:
  hybrid: true
  rerank: false                  # the default; add it later and measure the delta yourself
chunking:
  chunk_size: 768
autopilot:
  enabled: true
```

Then: ingest a representative slice → `fasterrag autopilot run` for a measured baseline → flip `retrieval.rerank: true` and re-run to see the delta on **your** corpus. **Why:** this is the honest way to evaluate — our defaults are informed by published evidence ([references.md](references.md)), but your corpus is the only benchmark that decides your configuration.

## R10 — Choosing a chunk size by measurement rather than folklore

`chunking.chunk_size` and `chunking.overlap` are the two settings people copy from a blog post and never revisit, because checking them is expensive: each value is a *different index*, so comparing two of them means building two of them. `fasterrag autopilot run` (R9) deliberately will not touch them for that reason — it searches query-time settings, where one index serves every candidate.

The index-time search is a separate operation with a separate price tag, in `services/autopilot.py`:

```python
from fasterrag.services.autopilot import plan_index_sweep, render_plan, sweep_index_time

plan = plan_index_sweep(dataset, settings, collection="docs")
print(render_plan(plan))        # rebuilds, chunks, embedding tokens, list-price cost

outcome = await sweep_index_time(
    plan, dataset, settings, adapter, router,
    rebuild=my_rebuild,          # builds one scratch collection from candidate settings
    approved_rebuilds=plan.rebuilds,   # defaults to 0 — the sweep refuses until you raise it
)
```

```yaml
autopilot:
  enabled: true
cost:
  estimator: true                # the plan is priced with D9's estimator; off means no plan
chunking:
  token_counter: estimate        # the plan chunks the corpus once per candidate
```

Four things to know before you run it:

- **It rebuilds the eval dataset's corpus, not your production corpus.** A golden set only means something against the corpus it was written against, and that corpus is small enough to build four or five times. Keep it representative or the answer transfers to nothing.
- **The cost is shown before it is paid, and the approval is per run.** `plan_index_sweep` prices every candidate with the same estimator `fasterrag estimate` uses; `sweep_index_time` refuses to start unless `approved_rebuilds` covers what the plan reported. The default is `0`, so forgetting the argument costs nothing.
- **Each candidate is built into its own `autopilot-sweep-*` collection and dropped afterwards**, inside a `finally` that runs over names recorded *before* each build started. The live collection is scored as-is and no alias is ever moved. If a drop itself fails, the name comes back in `outcome.leaked` — drop those by hand.
- **The suggestion is still only a suggestion, and an index-time one needs a reindex.** Pasting a new `chunk_size` into `config.yaml` changes nothing the live index already contains; run `fasterrag index reembed` afterwards or you have changed the label, not the index.

**Status (2026-11-04, TASK-0145):** this is a Python surface only. There is no `fasterrag autopilot sweep` command and no config key gating it yet — the opt-in is the `approved_rebuilds` argument. **Trade-off:** you supply the `rebuild` callable, which keeps the sweep independent of how you ingest; it must create the scratch collection at the candidate's own dimensions.

---

## Composing recipes

| Combination | Verdict |
|---|---|
| R1 + R2 | Excellent — maximum accuracy, fully local. Watch ingest time: enrichment runs through your local LLM. |
| R3 + R4 | Good — cheap and fast; verify recall with R7 before trusting it in production. |
| R5 + R3 | Natural pairing, and now complete: tenant isolation, preflight estimation, *and* a runtime per-tenant budget (TASK-0242 ✅). Size the budget for your replica count — it is counted per process. |
| R2 + R4 | Conflicting: `rerank_top_n` 300 vs 50 is the same dial pulled in opposite directions. Pick your point on the curve and measure it. |
| Any + R7 | Always worth adding. A tuned config with no regression gate degrades silently over time. |
| R9 + R10 | The full evaluation: R9 settles the query-time knobs for free against one index, then R10 pays for the index-time ones. Run R9 first — if it finds a large win cheaply, you may not need to buy the rebuilds at all. |
