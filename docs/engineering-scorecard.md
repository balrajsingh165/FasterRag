# engineering-scorecard.md — What "Good" Means Here, And How Far We Are From It

> **Who this is for.** Any contributor, human or AI, who is about to change this repository and wants the change to make the system more trustworthy and not merely larger. It states the rubric the project is scored against, the evidence behind the current score, what a score of 90 in each category would require, and the working rules for getting there. It is deliberately long: it is meant to be sufficient on its own, so that someone who has read nothing else can pick a task, do it correctly, and prove it.
>
> **Status of the numbers.** The scores are one auditor's judgement against a written rubric, derived on 2026-11-11 from reading the source and running the unit tier. They are not measurements, they are not a benchmark-ledger entry, and nothing here may be quoted as a performance claim ([benchmarks.md](benchmarks.md) is the only place those live). Re-derive them whenever a P0 task below closes; a stale score is the same defect as a stale count.
>
> **Not published.** This file is excluded from the documentation site for the reason [todo.md](todo.md) is: it describes unfixed weaknesses in enough detail to act on.
>
> **Companion documents.** [engineering-audit.md](engineering-audit.md) is the analysis behind the numbers — method, findings per category, what each stage of work is expected to be worth, and release readiness. [task-briefs.md](task-briefs.md) is the step-by-step brief for every open task named below, with the file and line each one starts from, the tests to write and the gates that will object. Read this document for *what good means*, the audit for *why the score is what it is*, and the briefs for *how to do the work*.

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

Three columns: the audit as first derived on commit `03a11f2`; the re-score after the fixes that landed the same day (TASK-0427 to TASK-0435); and the re-score after the first batch of P0 tasks closed on 2026-11-12. A column moved only where a defect was closed *and* a test now proves it.

| Category | Weight | Audit | After same-day fixes | After the first P0 batch | What holds it down now |
|---|---|---|---|---|---|
| Architecture and design | 10% | 52 | 58 | 60 | Import cycles; query services and the LLM client still built per request; the idle `worker` command and `app.workers` still present; two-thirds of `services/` is peripheral |
| Core RAG functionality | 12% | 58 | 62 | 71 | Single-turn only; no LLM retry or fallback; no document listing; a degraded answer's trace id resolves to nothing |
| Document ingestion and processing | 8% | 52 | 60 | 68 | Semantic chunking refused in the pipeline; HTML, URL and inline-source parsing defects; seven formats; stale chunks survive when dedup is off |
| Retrieval and search quality | 10% | 45 | 45 | 45 | English-only sparse leg; no threshold, diversity or query transformation; quality unmeasured |
| Embedding and vector infrastructure | 7% | 58 | 60 | 67 | Two adapters unproven in CI; heavy adapter duplication with real drift; the new disk cache write on the embedding path is unmeasured |
| Advanced RAG capabilities | 7% | 15 | 15 | 15 | Nothing beyond a refuse-gate, replay and a regression gate |
| Extensibility and modularity | 8% | 50 | 51 | 66 | Provider plugins are selectable; chunkers, parsers and rerankers still have no plugin point; no sample plugin or author's guide |
| Performance and scalability | 8% | 18 | 24 | 27 | No queue, one process, O(N) journal scans, and no citable number of any kind |
| Reliability and error handling | 7% | 42 | 58 | 59 | Capped at 59 by scoring rule 5 while the journal can lose acknowledged state: its writes are not `fsync`ed and have a window with no live record (TASK-0437). Also: half-open breaker admits a stampede; no LLM retry or request deadline; Redis clients without timeouts |
| Testing and quality assurance | 7% | 66 | 68 | 74 | Two vector adapters skip their contract suite in CI; eval sets cannot discriminate and nothing evaluates the answer; one Python version in CI |
| Security | 5% | 55 | 64 | 65 | Tenant scoping of ingest and query is opt-in and incomplete; prompt injection unmitigated; keys plaintext and unrotatable |
| Developer experience and APIs | 5% | 58 | 62 | 66 | No zero-infrastructure start; heavy first run; facade cannot query as a tenant; the CLI may misread streamed answers |
| Documentation | 3% | 55 | 62 | 65 | Some documents still describe process surfaces that do nothing; no code block is executed by a test |
| Packaging and release readiness | 3% | 35 | 35 | 35 | No tag, no release, no publish pipeline, heavy mandatory dependencies |
| **Weighted total** | 100% | **47** | **52** | **56** | |

The arithmetic for the second column, so it can be re-done by hand: 5.80 + 7.44 + 4.80 + 4.50 + 4.20 + 1.05 + 4.08 + 1.92 + 4.06 + 4.76 + 3.20 + 3.10 + 1.86 + 1.05 = 51.82.

The arithmetic for the third column: 6.00 + 8.52 + 5.44 + 4.50 + 4.69 + 1.05 + 5.28 + 2.16 + 4.13 + 5.18 + 3.25 + 3.30 + 1.95 + 1.05 = 56.50. It was derived on 2026-11-12, after nine P0 tasks closed (TASK-0436, TASK-0438, TASK-0439, TASK-0441, TASK-0442, TASK-0443, TASK-0455, TASK-0470, TASK-0473), each with a test that failed before the fix and CI green on the merged tree. **It was scored by the same auditor who coordinated the changes**, which is the weakest form of re-scoring: treat it as an estimate until someone who did not write the changes re-derives it. Retrieval, advanced RAG and packaging did not move, because nothing that closed touches them; performance moved only for the wired embedding cache, and still has no citable number.

**Read this honestly.** Five points in a day came from fixing defects that were each a few dozen lines once understood. The remaining thirty-eight are not like that. Roughly: closing every P0 task in [todo.md](todo.md) reaches the low 60s; P1 reaches the mid 70s; the last fifteen points are measurement, breadth and time.

## 4. What 90 requires, category by category

Each block states the bar, what already meets it, and the tasks that close the gap. A category is at 90 when every line under "The bar" is true and proven by a test or a ledger entry.

### 4.1 Architecture and design — 58 → 90

**The bar.** One dependency direction, enforced by a gate (`api`, `cli` → `services` → `workers` → `core` ← `adapters`). Long-lived resources are built once per process and injected. Every process the documents name does the work described. The runtime package contains product code only. One implementation of each pipeline.

**Already true.** Routers hold no business logic. Vendors sit behind adapter interfaces and factories and do not leak vendor types. REST, CLI and the Python facade call one service layer. Errors are one typed taxonomy rendered as RFC 9457 everywhere. Breakers, the cost governor and the reranker model are process-wide.

**To close.** TASK-0456 (cycles and an import contract), TASK-0440 with TASK-0130 (the worker, `app.workers`), TASK-0438 (one query pipeline), TASK-0457 (scope), TASK-0437 (state that is safe with more than one process). TASK-0466 (services and the LLM client built once, not per request).

**The design rule to hold while doing it.** State that must outlive a unit of work — a breaker count, a spend window, a loaded model, a journal — lives in a process-scoped object reached through one `shared_*` function, with a `reset_*` twin called around every test in `tests/conftest.py`. Never construct such an object inside a per-request or per-job code path. This single mistake accounted for four separate findings.

### 4.2 Core RAG functionality — 62 → 90

**The bar.** Ingest, update and delete a document. Answer with citations over stream and non-stream with identical semantics. Multi-turn. Provider failure is retried, then falls back, then degrades — observably. Prompts are configurable and fingerprinted.

**To close.** TASK-0443 (delete and update), TASK-0438 (parity), TASK-0449 (retry, fallback, deadline), TASK-0458 (history, templates, structured output), TASK-0441 (two provider adapters verified against real SDKs). TASK-0467 (a disconnected client stops the generation), TASK-0468 (defined behaviour for a missing collection and a blank query), TASK-0469 (list a collection's documents).

### 4.3 Document ingestion and processing — 60 → 90

**The bar.** Every documented chunking strategy runs in the pipeline. A document is `indexed` only when it is in the index. One bad document costs one document. Parsers are correct on nested and malformed input. Office formats beyond DOCX are covered by a conversion adapter.

**Already true.** The second and third lines, as of 2026-11-11: `workers/ledger.py` makes "indexed" a statement about the vector database, and `parse_and_chunk` converts untyped parser failures into dead-letter entries.

**To close.** TASK-0431 (semantic chunking), TASK-0459 (parser defects, chunk-id fingerprint), TASK-0462 (formats), TASK-0436 (resume). TASK-0470 (an empty document is not `indexed`) and TASK-0473 (a parse timeout) are small and come first; then TASK-0471 (deduplication under concurrency), TASK-0472 (encodings, encrypted and scanned PDFs), TASK-0504 (symbolic links).

### 4.4 Retrieval and search quality — 45 → 90

**The bar.** Hybrid retrieval that works on non-English text. A relevance floor. Diversity. At least one query transformation. **Every one of them justified by a number on an eval set that can tell them apart.**

**This category cannot move without TASK-0454.** A retrieval change with no discriminating eval set is an opinion. Do the measurement harness first, then TASK-0447, TASK-0448, TASK-0463. TASK-0481 extends the harness from retrieval to the generated answer; TASK-0480 (reranker limits) and TASK-0485 (language awareness) follow.

### 4.5 Embedding and vector infrastructure — 60 → 90

**The bar.** Every adapter proven against a real server on every change. One score convention and one error code across adapters. Embeddings cached correctly. Filters mean the same thing on every backend.

**To close.** TASK-0412 and TASK-0352 (CI for Milvus and Weaviate), TASK-0451 (duplication and drift), TASK-0452 (pgvector), TASK-0442 (embedding cache), TASK-0464 (in-process store, filter grammar), TASK-0049 (Pinecone). TASK-0477 (refuse a query against an index built under a different model), TASK-0478 (batches that adapt to provider limits), TASK-0479 (index tuning in configuration).

### 4.6 Advanced RAG capabilities — 15 → as far as the evidence supports

**The bar.** Whatever advanced technique ships, ships with a measured gain on a public eval set and a committed harness.

**Guidance.** Do not start here. The comparison set leads with agentic and graph retrieval; fasterRag's opening is the thing none of them ships first-party — proof. TASK-0465 records the decision. A 90 in this category is not a goal for the first beta. It cannot be ignored either: at 7% of the total, a 15 here removes six points from the ceiling, and [engineering-audit.md](engineering-audit.md) §8 shows that every other task written down reaches about 86. TASK-0482 (follow-up questions), TASK-0483 (corrective retrieval) and TASK-0484 (self-query filters) are the three smallest measured steps, each gated on TASK-0454.

### 4.7 Extensibility and modularity — 51 → 90

**The bar.** A third party can add a vector database, embedder, LLM, chunker, parser or reranker by installing a package and naming it in `config.yaml`, with no edit to this repository, and a test proves each.

**To close.** TASK-0439, then TASK-0486 (a sample plugin run against the contract suite in CI, an interface version, a public-surface snapshot, an author's guide). The constraint that makes TASK-0439 non-trivial: `config/` must stay fail-fast and must not import `adapters/`.

### 4.8 Performance and scalability — 24 → 90

**The bar.** Citable ledger entries for ingestion throughput and query latency at 100k and 1M chunks. A queue-backed worker that scales horizontally. No O(N)-in-corpus work on a request path. A named competitor measured with a committed harness.

**To close.** TASK-0454 first, then TASK-0130, TASK-0437, TASK-0442, TASK-0408. TASK-0466 (stop rebuilding clients per request), TASK-0487 (timings that localise a slow request), TASK-0488 (the query suite, load and soak on a schedule). Until a `BENCH-` entry exists, the only honest sentence about speed is the goal sentence the README already carries.

### 4.9 Reliability and error handling — 58 → 90

**The bar.** A crash at any instant loses no acknowledged work and leaves no lie in the journal. A job is never left non-terminal. Every dependency has a timeout, a retry policy and a breaker that opens. Shutdown drains.

**Already true.** Journalling follows the index write. A job is always settled. A refused write no longer ends a worker. Breakers are shared and can open.

**To close.** TASK-0436 (resume and drain), TASK-0437 (durable journal), TASK-0449 (LLM retry and deadline), TASK-0450 (half-open), TASK-0453 (Redis timeouts), TASK-0407 to TASK-0410 (blocking I/O on the event loop). TASK-0473 (parse timeout), TASK-0474 (idempotent dead-letter retry, journal retention), TASK-0475 (journal appends inside the taxonomy), TASK-0476 (a timeout on embedded Chroma).

### 4.10 Testing and quality assurance — 68 → 90

**The bar.** The production executor, the real client libraries and real backends are each exercised by a gated test. Eval sets discriminate. Mutation testing runs and its survivors are tracked.

**To close.** TASK-0455, TASK-0412, TASK-0441, TASK-0454, TASK-0300. TASK-0481 (evaluate the answer), TASK-0489 (property and fuzz coverage for hostile input), TASK-0490 (a lane for every claimed Python version, a per-test timeout), TASK-0499 (execute the documents' code).

### 4.11 Security — 64 → 90

**The bar.** Closed by default. A tenant cannot address another tenant's data through any endpoint. Retrieved text cannot steer the model or the grader. Supply chain pinned by digest.

**Already true.** Loopback bind by default with an exposure warning; path and archive confinement; SSRF-guarded fetching; constant-time key comparison; `/v1/estimate` confined; the semantic cache scoped by collection and filter.

**To close.** TASK-0339 and TASK-0381 (tenant scoping of query and ingest — a maintainer decision is pending there; read both before touching the code), TASK-0446 (prompt injection), TASK-0460 (control-plane hardening), TASK-0326 (unauthenticated throttling). TASK-0491 (prove a key never reaches a log), TASK-0492 (hashed, rotatable keys), TASK-0493 (an audit trail), TASK-0494 (CORS and response headers), TASK-0495 (a written threat model), TASK-0496 (supply-chain statements made true).

### 4.12 Developer experience, documentation, packaging — 62 / 62 / 35 → 90

**The bar.** `pip install`, one command, an answered question, with no service to start. Every snippet in every document runs. A tagged release exists.

**To close.** TASK-0444 (local profile, slim core), TASK-0445 and TASK-0087 (release), TASK-0461 (facade tenancy), TASK-0440 (documents that describe absent surfaces). TASK-0497 (the blocking facade in a notebook), TASK-0498 (runnable examples), TASK-0499 (executed snippets), TASK-0500 (every error code with a remedy), TASK-0501 (link checking, versioned documentation), TASK-0502 (a configuration schema version and an upgrade path), TASK-0503 (community files).

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
12. **Read the brief, then delete it.** [task-briefs.md](task-briefs.md) gives each open task's starting points, steps, tests and traps, and §0.4 there lists the tests and gates that fire when you add a config key, a route, a flag, an error code or a facade method. When you tick a task in [todo.md](todo.md), delete its brief in the same commit.

## 6. The order of work

Sequenced by dependency, not by appeal. Do not skip ahead: later items are only meaningful once earlier ones exist.

1. **Make ingestion trustworthy end to end** — TASK-0470 and TASK-0473 first (small; they stop a lost document and a hung job), then TASK-0436, TASK-0437, TASK-0443, TASK-0455.
2. **Make the query path one path** — TASK-0438, TASK-0449, TASK-0441.
3. **Make it installable and startable by a stranger** — TASK-0444, TASK-0445, TASK-0440.
4. **Build the measuring instrument** — TASK-0454. Everything about retrieval quality and speed waits on this.
5. **Open the system up** — TASK-0439, TASK-0451, TASK-0456.
6. **Close the security decisions** — TASK-0339, TASK-0381, TASK-0446, TASK-0460.
7. **Then, with numbers, improve retrieval** — TASK-0447, TASK-0448, TASK-0463, TASK-0431.

The scenario tasks added on 2026-11-12 (TASK-0466 to TASK-0504) slot into these steps by category; [task-briefs.md](task-briefs.md) §5 lists every task under the category it raises.

## 7. What the comparison set says about where to aim

Researched 2026-11-11 against LlamaIndex, LangChain, Haystack, RAGFlow, txtai, LightRAG, GraphRAG, R2R, Dify and Onyx. Three findings shape the roadmap:

- **Commodity now** — offered by nearly all ten: hybrid dense plus BM25 with rank fusion, reranking, citations, token streaming, pluggable model providers, Docker deployment. Having these earns nothing.
- **Rare or absent across all ten** — idempotent ingestion as a stated contract, per-tenant cost budgets, replay of a past query under a candidate configuration, a retrieval regression gate that runs in CI, a portable cross-store archive, and provider swap by configuration alone with a schema that refuses bad input at load. fasterRag has begun every one of these. Finishing them, and proving them, is the opening.
- **The closest architectural relative** — a REST service with an SDK and configuration profiles — has had no release since mid-2025. There is room for a maintained one.

The strategy that follows is narrow on purpose: be the RAG backend whose claims can be checked. Breadth of integrations and advanced retrieval techniques are where the incumbents are strongest and where catching up costs the most; verifiability is where they are weakest and where this repository's habits already point.
