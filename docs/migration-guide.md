# migration-guide.md — Arriving From Another RAG Framework

For teams moving an existing pipeline (LangChain, LlamaIndex, Haystack, or a hand-rolled stack) onto fasterRag. This is a **concept mapping and procedure**, not a competitive comparison: the provable-claims policy forbids performance claims against tools we haven't measured with a committed harness ([benchmarks.md](benchmarks.md)), and the honest trade-offs are listed at the end.

## The one mental-model shift

Most RAG frameworks are **libraries you compose in code**: you write the chain, wire the retriever to the LLM, and your Python *is* the configuration. fasterRag inverts that — the pipeline is fixed and **configuration is the interface**:

| You used to | You now |
|---|---|
| Write code to change chunk size, top_k, or the retriever mix | Edit `config.yaml`; restart (or reindex if it changes chunks) |
| Swap a vector store by importing a different class and rewriting call sites | Change `vector_db.provider`; no code changes ([ADR-0002](adr/ADR-0002-adapter-factory-pluggability.md)) |
| Discover quality regressions in production | The regression gate blocks them at CI (D7) |
| Re-index by dropping and rebuilding | Blue/green reindex with an eval-gated alias swap (D2) |
| Read prompts from your own source | Read them from [prompts.md](prompts.md); override them as versioned templates |

If you want library-style composition back, you still have it: `fasterrag.parsing`, `.chunking`, `.retrieval`, `.rerank`, and `.evals` are importable standalone, and `rag.retrieve()` stops before generation ([python-api.md](python-api.md), cookbook recipe R6).

## Concept mapping

| Their concept | fasterRag equivalent | Notes |
|---|---|---|
| `DocumentLoader` / `Reader` | Ingestion sources (`path`, `url`, `inline`) + the parsing stage | Parser selection is automatic by MIME type; OCR and layout extraction are built in |
| `TextSplitter` / `NodeParser` | `chunking.strategy` (`fixed`, `recursive`, `semantic`, `layout`, `late`) | Behavior is property-tested: offsets in-bounds, overlap respected, no empty chunks |
| `Embeddings` class | `embeddings.provider` + `model` | Plus tiered routing by document class, which most frameworks leave to you |
| `VectorStore` | `vector_db.provider` via `VectorDBAdapter` | Every implementation passes one shared contract suite |
| `Retriever` | `retrieval.*` (hybrid legs, RRF, filters) | Hybrid + RRF(k=60) is the default, not an add-on |
| `EnsembleRetriever` / manual fusion | `retrieval.hybrid` + `rrf_k` | Rank-based fusion; no score normalization to tune |
| Reranker wrapper | `retrieval.rerank` + `reranker_model` | Same cross-encoder idea, config-gated with a degradation path |
| `RetrievalQA` / query engine | `POST /v1/query` · `rag.query()` · `fasterrag query` | One pipeline behind all three surfaces |
| Prompt templates in your code | [prompts.md](prompts.md) P1–P4 contracts | Overridable, versioned, and regression-gated |
| Callbacks / tracing integration | OTel spans + local trace store + optional Langfuse | Trace id flows through logs, problems, and responses |
| Custom eval scripts | Eval harness (recall@k, MRR, nDCG, faithfulness) + golden sets | Wired into CI as a blocking gate |
| Your own retry/timeout wrappers | `reliability.*` (timeouts, backoff, breakers, ladder) | Mandatory on every external call, not opt-in |

**Concepts with no direct equivalent** (because they don't exist in most frameworks): the index lockfile (D1), checkpointed exactly-once ingestion with a DLQ (D3), the degradation ladder (D4), time-travel replay (D8), the preflight cost estimator (D9), and `doctor` (D10). These are the parts you stop building yourself.

## Migration procedure

**1. Inventory what you have.** Write down, from your current code: chunk size and overlap, embedding model **and version**, vector store and distance metric, top_k, whether you rerank, and your prompt text. That list *is* your first `config.yaml`, and the model+version pair is what determines whether you can reuse vectors.

**2. Translate into `config.yaml`.** Start from the canonical default ([`config.yaml`](../config.yaml)) and change only what your inventory dictates. Then `fasterrag config validate` — cross-field rules catch mismatches (e.g. `top_k > rerank_top_n`) that were silent bugs in the old stack.

**3. Choose re-embed or reuse.**

| Situation | Path |
|---|---|
| Same embedding model, same dimensions, and the target adapter supports vector import | Reuse vectors: build an archive and import with `--include-vectors` ([archive-format.md](archive-format.md)) |
| Different model, different dimensions, or unknown model version | **Re-ingest from source documents.** This is the recommended default — it produces a valid `index.lock` and lets you use fasterRag's chunkers |
| Source documents no longer available | Import chunks as documents (`--reembed`), accepting that chunk boundaries are inherited from the old system |

Unknown provenance is a real answer, not a failure: if you cannot state the exact model version that produced your existing vectors, re-embed. That uncertainty is precisely the problem D1's lockfile exists to eliminate going forward.

**4. Estimate before you spend.** `fasterrag estimate ./corpus --all-providers` gives token counts, projected cost, and projected time — do this before a large re-embed, not after.

**5. Build a golden set from the old system's behavior.** Take queries your current pipeline handles well and record what it retrieves. Turn them into a golden set ([testing-strategy.md](testing-strategy.md) §1.6) — now you have an objective before/after instead of vibes. `fasterrag autopilot run` can bootstrap the set; review it before trusting it.

**6. Run both in parallel, then cut over.** Point fasterRag at the same corpus, replay your golden set through both, and compare recall@k and nDCG. Cut over when the numbers justify it — and note the baseline is *your old system*, measured by you, which is exactly the standard we hold our own claims to.

**7. Turn on the guarantees you couldn't have before.** `eval.regression_gate: true`, `index.lockfile: true`, `traces.store: true`, and — if hallucination is a live concern — `generation.grounded_or_refuse: true`.

## What you give up

Stated plainly, because a migration guide that only lists wins is marketing:

- **Arbitrary in-code pipeline surgery.** If your value comes from a bespoke chain (custom routing between multiple indexes, agentic multi-hop retrieval, tool-calling mid-retrieval), fasterRag's fixed pipeline will feel restrictive. Use it as the retrieval engine (R6) and keep your orchestration above it.
- **Agent frameworks.** fasterRag answers queries over corpora; it is not an agent framework, and that is an explicit non-goal ([scope.md](scope.md)).
- **Multi-modal retrieval.** Text only in beta; images/audio are future scope.
- **The ecosystem's long tail.** Mature frameworks ship hundreds of integrations. We ship **two** vector DBs today — Qdrant and pgvector, with Milvus, Weaviate, Pinecone, and Chroma specified but unbuilt (TASK-0049) — four embedding providers, and any OpenAI-compatible LLM endpoint, plus an entry-point plugin contract so the tail is addable without forking ([python-api.md](python-api.md)). If your exotic connector matters more than reliability guarantees, stay where you are.
- **Maturity.** fasterRag is pre-beta and has no measured production track record yet. The [benchmark ledger](benchmarks.md) holds **two entries, both explicitly non-citable** (taken on a loaded developer laptop), so there is no citable performance figure for fasterRag anywhere — [performance.md](performance.md) is goals and method only, and every SLO target in [slo.md](slo.md) is still `TBD-until-measured`. Where a doc quotes a wall-clock number it is a dated one-off observation, labelled as such, never a benchmark.

## Coming from a hand-rolled stack

You are usually the easiest case to migrate, and the one with the most to gain from the operational parts. The mapping is usually: your loader → ingestion sources; your `text_splitter.py` → `chunking.*`; your `pinecone_client.py` → an adapter; your retry decorator → `reliability.*`; your eval notebook → the eval harness plus a CI gate. The parts you probably never built — journal/DLQ, lockfile, degradation ladder, replay, doctor — are the ones that turn a working prototype into something operable.

Start with cookbook recipe **R9** (evaluate before committing): minimal config, ingest a slice, measure on your own data, then decide.
