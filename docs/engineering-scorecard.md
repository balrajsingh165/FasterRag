# engineering-scorecard.md — What "Good" Means Here, And How Far We Are From It

> **Who this is for.** Any contributor, human or AI, who is about to change this repository and wants the change to make the system more trustworthy and not merely larger. It states the rubric the project is scored against, the evidence behind the current score, what a score of 90 in each category would require, and the working rules for getting there. It is deliberately long: it is meant to be sufficient on its own, so that someone who has read nothing else can pick a task, do it correctly, and prove it.
>
> **Status of the numbers.** The scores are one auditor's judgement against a written rubric, derived on 2026-11-11 from reading the source and running the unit tier. They are not measurements, they are not a benchmark-ledger entry, and nothing here may be quoted as a performance claim ([benchmarks.md](benchmarks.md) is the only place those live). Re-derive them whenever a P0 task below closes; a stale score is the same defect as a stale count.
>
> **Not published.** This file is excluded from the documentation site for the reason [todo.md](todo.md) is: it describes unfixed weaknesses in enough detail to act on.

---

## 1. The goal, stated so it can be checked

fasterRag's ambition is to be a one-stop RAG system people can *trust*: install it, point it at a corpus, and rely on what it says it did. Trust is the product. Every decision below follows from one rule, which is already project policy:

> **A claim without a measurement is a bug.** A capability counts when the code does it, a test proves it, and a document describes exactly that — in that order. A class that exists, a paragraph that promises, or a test that exercises a fake is not a capability.

The target is **90/100 overall** on the rubric in §2. That number is a direction, not a date. Three categories cannot reach 90 by writing code in a session — measured performance needs reference hardware, advanced retrieval needs an eval set large enough to tell better from worse, and ecosystem maturity needs users — and pretending otherwise would be the exact failure this project exists to avoid. §4 says what each category honestly needs.

## 2. The rubric

Each category is scored 0-100 on the same scale, then weighted.

| Band | Meaning |
|---|---|
| 0-19 | Fundamentally missing or unusable |
| 20-39 | Very immature |
| 40-59 | Basic implementation |
| 60-69 | Functional but limited |
| 70-79 | Good engineering |
| 80-89 | Strong, mature implementation |
| 90-95 | Excellent, competitive |
| 96-100 | Exceptional |

Scoring rules, applied without exception:

1. **Implemented beats documented.** No points for a feature that is specified, stubbed, or present only as an interface.
2. **Reachable beats present.** Code that no entry point can reach (an unwired cache, a resume path nothing calls) scores as absent.
3. **Proven beats plausible.** A behaviour verified only against a hand-written fake is recorded as unverified. Three vector-adapter defects and one serial "parallel" pool were invisible for months for exactly this reason.
4. **Measured beats asserted.** Performance and quality score on ledger entries only.
5. **A defect that silently loses or leaks data caps its category at 59**, whatever else is good about it.

## 3. Where the system stands

Two columns: the audit as first derived on commit `03a11f2`, and the re-score after the fixes that landed the same day (TASK-0427 to TASK-0435). The second column moved only where a defect was closed *and* a test now proves it.

| Category | Weight | Audit | After same-day fixes | What holds it down now |
|---|---|---|---|---|
| Architecture and design | 10% | 52 | 58 | Import cycles; services built per request; three process surfaces that do nothing; two-thirds of `services/` is peripheral |
| Core RAG functionality | 12% | 58 | 62 | Two diverged copies of the query pipeline; single-turn only; no delete or update; no LLM retry |
| Document ingestion and processing | 8% | 52 | 60 | Semantic chunking refused in the pipeline; HTML, URL and inline-source parsing defects; seven formats |
| Retrieval and search quality | 10% | 45 | 45 | English-only sparse leg; no threshold, diversity or query transformation; quality unmeasured |
| Embedding and vector infrastructure | 7% | 58 | 60 | Embedding cache unwired; two adapters unproven in CI; heavy adapter duplication with real drift |
| Advanced RAG capabilities | 7% | 15 | 15 | Nothing beyond a refuse-gate, replay and a regression gate |
| Extensibility and modularity | 8% | 50 | 51 | Plugins blocked by the config schema; chunkers, parsers and rerankers are not pluggable |
| Performance and scalability | 8% | 18 | 24 | No queue, one process, O(N) file scans, and no citable number of any kind |
| Reliability and error handling | 7% | 42 | 58 | No resume, no shutdown drain, no `fsync`; half-open breaker admits a stampede |
| Testing and quality assurance | 7% | 66 | 68 | The real process pool is never run; fakes stand in for two adapters and two providers; toy eval sets |
| Security | 5% | 55 | 64 | Tenant scoping of ingest and query is opt-in and incomplete; prompt injection unmitigated |
| Developer experience and APIs | 5% | 58 | 62 | No zero-infrastructure start; heavy first run; facade cannot act as a tenant |
| Documentation | 3% | 55 | 62 | Vast and candid, but several documents still describe surfaces that do not exist |
| Packaging and release readiness | 3% | 35 | 35 | No tag, no release, no publish pipeline, heavy mandatory dependencies |
| **Weighted total** | 100% | **47** | **52** | |

The arithmetic for the second column, so it can be re-done by hand: 5.80 + 7.44 + 4.80 + 4.50 + 4.20 + 1.05 + 4.08 + 1.92 + 4.06 + 4.76 + 3.20 + 3.10 + 1.86 + 1.05 = 51.82.

**Read this honestly.** Five points in a day came from fixing defects that were each a few dozen lines once understood. The remaining thirty-eight are not like that. Roughly: closing every P0 task in [todo.md](todo.md) reaches the low 60s; P1 reaches the mid 70s; the last fifteen points are measurement, breadth and time.

## 4. What 90 requires, category by category

Each block states the bar, what already meets it, and the tasks that close the gap. A category is at 90 when every line under "The bar" is true and proven by a test or a ledger entry.

### 4.1 Architecture and design — 58 → 90

**The bar.** One dependency direction, enforced by a gate (`api`, `cli` → `services` → `workers` → `core` ← `adapters`). Long-lived resources are built once per process and injected. Every process the documents name does the work described. The runtime package contains product code only. One implementation of each pipeline.

**Already true.** Routers hold no business logic. Vendors sit behind adapter interfaces and factories and do not leak vendor types. REST, CLI and the Python facade call one service layer. Errors are one typed taxonomy rendered as RFC 9457 everywhere. Breakers, the cost governor and the reranker model are process-wide.

**To close.** TASK-0456 (cycles and an import contract), TASK-0440 with TASK-0130 (the worker, `app.workers`), TASK-0438 (one query pipeline), TASK-0457 (scope), TASK-0437 (state that is safe with more than one process).

**The design rule to hold while doing it.** State that must outlive a unit of work — a breaker count, a spend window, a loaded model, a journal — lives in a process-scoped object reached through one `shared_*` function, with a `reset_*` twin called around every test in `tests/conftest.py`. Never construct such an object inside a per-request or per-job code path. This single mistake accounted for four separate findings.

### 4.2 Core RAG functionality — 62 → 90

**The bar.** Ingest, update and delete a document. Answer with citations over stream and non-stream with identical semantics. Multi-turn. Provider failure is retried, then falls back, then degrades — observably. Prompts are configurable and fingerprinted.

**To close.** TASK-0443 (delete and update), TASK-0438 (parity), TASK-0449 (retry, fallback, deadline), TASK-0458 (history, templates, structured output), TASK-0441 (two provider adapters verified against real SDKs).

### 4.3 Document ingestion and processing — 60 → 90

**The bar.** Every documented chunking strategy runs in the pipeline. A document is `indexed` only when it is in the index. One bad document costs one document. Parsers are correct on nested and malformed input. Office formats beyond DOCX are covered by a conversion adapter.

**Already true.** The second and third lines, as of 2026-11-11: `workers/ledger.py` makes "indexed" a statement about the vector database, and `parse_and_chunk` converts untyped parser failures into dead-letter entries.

**To close.** TASK-0431 (semantic chunking), TASK-0459 (parser defects, chunk-id fingerprint), TASK-0462 (formats), TASK-0436 (resume).

### 4.4 Retrieval and search quality — 45 → 90

**The bar.** Hybrid retrieval that works on non-English text. A relevance floor. Diversity. At least one query transformation. **Every one of them justified by a number on an eval set that can tell them apart.**

**This category cannot move without TASK-0454.** A retrieval change with no discriminating eval set is an opinion. Do the measurement harness first, then TASK-0447, TASK-0448, TASK-0463.

### 4.5 Embedding and vector infrastructure — 60 → 90

**The bar.** Every adapter proven against a real server on every change. One score convention and one error code across adapters. Embeddings cached correctly. Filters mean the same thing on every backend.

**To close.** TASK-0412 and TASK-0352 (CI for Milvus and Weaviate), TASK-0451 (duplication and drift), TASK-0452 (pgvector), TASK-0442 (embedding cache), TASK-0464 (in-process store, filter grammar), TASK-0049 (Pinecone).

### 4.6 Advanced RAG capabilities — 15 → as far as the evidence supports

**The bar.** Whatever advanced technique ships, ships with a measured gain on a public eval set and a committed harness.

**Guidance.** Do not start here. The comparison set leads with agentic and graph retrieval; fasterRag's opening is the thing none of them ships first-party — proof. TASK-0465 records the decision. A 90 in this category is not a goal for the first beta.

### 4.7 Extensibility and modularity — 51 → 90

**The bar.** A third party can add a vector database, embedder, LLM, chunker, parser or reranker by installing a package and naming it in `config.yaml`, with no edit to this repository, and a test proves each.

**To close.** TASK-0439. The constraint that makes it non-trivial: `config/` must stay fail-fast and must not import `adapters/`.

### 4.8 Performance and scalability — 24 → 90

**The bar.** Citable ledger entries for ingestion throughput and query latency at 100k and 1M chunks. A queue-backed worker that scales horizontally. No O(N)-in-corpus work on a request path. A named competitor measured with a committed harness.

**To close.** TASK-0454 first, then TASK-0130, TASK-0437, TASK-0442, TASK-0408. Until a `BENCH-` entry exists, the only honest sentence about speed is the goal sentence the README already carries.

### 4.9 Reliability and error handling — 58 → 90

**The bar.** A crash at any instant loses no acknowledged work and leaves no lie in the journal. A job is never left non-terminal. Every dependency has a timeout, a retry policy and a breaker that opens. Shutdown drains.

**Already true.** Journalling follows the index write. A job is always settled. A refused write no longer ends a worker. Breakers are shared and can open.

**To close.** TASK-0436 (resume and drain), TASK-0437 (durable journal), TASK-0449 (LLM retry and deadline), TASK-0450 (half-open), TASK-0453 (Redis timeouts), TASK-0407 to TASK-0410 (blocking I/O on the event loop).

### 4.10 Testing and quality assurance — 68 → 90

**The bar.** The production executor, the real client libraries and real backends are each exercised by a gated test. Eval sets discriminate. Mutation testing runs and its survivors are tracked.

**To close.** TASK-0455, TASK-0412, TASK-0441, TASK-0454, TASK-0300.

### 4.11 Security — 64 → 90

**The bar.** Closed by default. A tenant cannot address another tenant's data through any endpoint. Retrieved text cannot steer the model or the grader. Supply chain pinned by digest.

**Already true.** Loopback bind by default with an exposure warning; path and archive confinement; SSRF-guarded fetching; constant-time key comparison; `/v1/estimate` confined; the semantic cache scoped by collection and filter.

**To close.** TASK-0339 and TASK-0381 (tenant scoping of query and ingest — a maintainer decision is pending there; read both before touching the code), TASK-0446 (prompt injection), TASK-0460 (control-plane hardening), TASK-0326 (unauthenticated throttling).

### 4.12 Developer experience, documentation, packaging — 62 / 62 / 35 → 90

**The bar.** `pip install`, one command, an answered question, with no service to start. Every snippet in every document runs. A tagged release exists.

**To close.** TASK-0444 (local profile, slim core), TASK-0445 and TASK-0087 (release), TASK-0461 (facade tenancy), TASK-0440 (documents that describe absent surfaces).

## 5. Working rules for whoever takes a task

These are the conventions that the audit found *were* followed where the code is good and were *not* followed where it is not.

1. **Reproduce first.** Write the failing test before the fix, named for the behaviour and carrying one docstring line that says what the defect was. `test_a_document_that_failed_to_embed_is_not_remembered_as_indexed` is the model.
2. **Test through the real thing at least once.** If a behaviour is only ever exercised against a fake you wrote, say so in the task and add the real-dependency test, even a skipped-without-Docker one marked `must_run`.
3. **One change, one commit, green.** `ruff check .`, `ruff format --check .`, `mypy`, and `pytest -m "not integration and not eval and not load and not soak"` pass before every commit. Single-line Conventional Commit, no trailers, no AI attribution ([CONTRIBUTING.md](CONTRIBUTING.md)).
4. **Docs move in the same commit series.** The reference page for the surface you changed, [CHANGELOG.md](CHANGELOG.md), and the task ticked in [todo.md](todo.md) with `— ✅ YYYY-MM-DD`. Run `python scripts/trim_docstrings.py --apply src` if a docstring passed four lines.
5. **Run the document gates before you push**: `check_doc_truth.py`, `check_claims.py`, `check_unbuilt_claims.py`, `check_task_ids.py`, `check_blockers.py`, `check_comments.py`, `check_error_taxonomy.py`, `check_openapi_drift.py`, `stamp_release.py --check`. They are CI-only, so a local run is the only warning you get.
6. **Never write a number you did not measure.** No throughput, latency, accuracy or "N× faster" without a `BENCH-` entry and a committed harness. If the change is expected to be faster, say "expected, unmeasured" and open or cite the measurement task.
7. **Never describe a capability as shipped before its code lands**, and never deny one that the tree contains. Both directions are gated, and both have happened.
8. **Risky behaviour ships behind a flag defaulting to off**, and a default changes only with an eval or benchmark entry that justifies it.
9. **Sensitive areas need an explicit task**: `config/` (fail-fast must never weaken), `adapters/` (no vendor type may leak, and an interface change needs maintainer review), provisioning, and anything touching `.env`.
10. **When the finding says NOT VERIFIED, verify it before fixing it.** Three items in the task list were read from code and never executed. The first step is the experiment, and "it was not a bug" is a valid, recordable outcome.
11. **Re-score.** When a task closes, update the row in §3 only if a defect is closed *and* proven. Leave the row alone otherwise.

## 6. The order of work

Sequenced by dependency, not by appeal. Do not skip ahead: later items are only meaningful once earlier ones exist.

1. **Make ingestion trustworthy end to end** — TASK-0436, TASK-0437, TASK-0443, TASK-0455.
2. **Make the query path one path** — TASK-0438, TASK-0449, TASK-0441.
3. **Make it installable and startable by a stranger** — TASK-0444, TASK-0445, TASK-0440.
4. **Build the measuring instrument** — TASK-0454. Everything about retrieval quality and speed waits on this.
5. **Open the system up** — TASK-0439, TASK-0451, TASK-0456.
6. **Close the security decisions** — TASK-0339, TASK-0381, TASK-0446, TASK-0460.
7. **Then, with numbers, improve retrieval** — TASK-0447, TASK-0448, TASK-0463, TASK-0431.

## 7. What the comparison set says about where to aim

Researched 2026-11-11 against LlamaIndex, LangChain, Haystack, RAGFlow, txtai, LightRAG, GraphRAG, R2R, Dify and Onyx. Three findings shape the roadmap:

- **Commodity now** — offered by nearly all ten: hybrid dense plus BM25 with rank fusion, reranking, citations, token streaming, pluggable model providers, Docker deployment. Having these earns nothing.
- **Rare or absent across all ten** — idempotent ingestion as a stated contract, per-tenant cost budgets, replay of a past query under a candidate configuration, a retrieval regression gate that runs in CI, a portable cross-store archive, and provider swap by configuration alone with a schema that refuses bad input at load. fasterRag has begun every one of these. Finishing them, and proving them, is the opening.
- **The closest architectural relative** — a REST service with an SDK and configuration profiles — has had no release since mid-2025. There is room for a maintained one.

The strategy that follows is narrow on purpose: be the RAG backend whose claims can be checked. Breadth of integrations and advanced retrieval techniques are where the incumbents are strongest and where catching up costs the most; verifiability is where they are weakest and where this repository's habits already point.
