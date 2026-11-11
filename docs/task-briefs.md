# task-briefs.md — How To Do Each Open Task, Step By Step

> **Who this is for.** Whoever is about to take a task from [todo.md](todo.md) — a human contributor or an AI agent starting with no memory of this repository. Each brief is meant to be enough on its own: where the code is, what to change, which tests to write, which gates will object, and how to know the task is finished.
>
> **What this is not.** It is not a task file. [todo.md](todo.md) is the only place a task is defined, ticked or tracked. This document holds no checkbox and no status; every brief is keyed by a `TASK-` id that already exists there. **When a task is ticked, delete its brief in the same commit** — a brief for finished work is a stale instruction.
>
> **How the facts were gathered.** Every file path, line number, function name and test name below was read from the tree at commit `d67fdf6` on 2026-11-12. Line numbers drift; names drift less. If a line number is wrong, search for the name — and if the name is gone, the brief is stale and the code wins. A statement marked **NOT VERIFIED** was read from code and never executed: run the experiment before writing the fix (working rule 10 in [engineering-scorecard.md](engineering-scorecard.md) §5).
>
> **Not published.** Excluded from the documentation site, for the reason [todo.md](todo.md) is.

---

## 0. Read this before any brief

### 0.1 The loop every task follows

1. **Read** the task's entry in [todo.md](todo.md), then its brief here, then the category's bar in [engineering-scorecard.md](engineering-scorecard.md) §4.
2. **Reproduce.** Write the failing test first. Name it for the behaviour (`test_a_document_that_failed_to_embed_is_not_remembered_as_indexed`), give it a one-line docstring saying what the defect was, and watch it fail for the right reason.
3. **Fix** the smallest thing that makes it pass. Do not refactor nearby code in the same commit.
4. **Run the local gate set** (§0.3). All of it. The document gates run only in CI, so a local run is the only warning you get.
5. **Update the documents** in the same commit series: the reference page for the surface you changed, [CHANGELOG.md](CHANGELOG.md) under `Unreleased`, the task ticked in [todo.md](todo.md) with `— ✅ YYYY-MM-DD`, its brief deleted from this file, and the row in scorecard §3 only if a defect is closed *and* proven.
6. **Commit** as one single-line Conventional Commit, at most 100 characters, no body, no trailers, no AI attribution. Then push and watch CI: two of the last three pushes to this repository failed on something only CI could see.

### 0.2 Environment

- **Interpreter and dependencies.** `uv sync --locked --extra dev --python 3.12`. CI runs Python 3.12 against `uv.lock`; a venv resolved any other way has disagreed with it on dozens of packages, and stricter type stubs in CI have failed a push that was green locally.
- **Windows.** `python` must resolve to the venv: in Git Bash, `export PATH="$PWD/.venv/Scripts:$PATH"`. The pre-commit hook runs ruff, the formatter, mypy and the whole unit tier for any commit containing Python (about six minutes).
- **Docker** is needed for the integration tier only. The unit tier has a network guard in `tests/unit/conftest.py` and must stay offline.
- **This checkout may be shared with another session.** Never run `git stash`, `git stash pop`, `git reset --hard` or `git checkout -- .` here. Work in a worktree if you need a clean tree.
- **Never read, print or commit `.env`.** `.env.example` is fine.

### 0.3 The local gate set

```bash
ruff check . && ruff format --check .
mypy                                   # configured for src, tests AND scripts; strict
pytest -m "not integration and not eval and not load and not soak"   # what CI runs on every push
python scripts/check_doc_truth.py && python scripts/check_openapi_drift.py
python scripts/check_blockers.py && python scripts/check_claims.py
python scripts/check_unbuilt_claims.py && python scripts/check_task_ids.py
python scripts/check_error_taxonomy.py && python scripts/check_comments.py
python scripts/stamp_release.py --check
python scripts/trim_docstrings.py --apply src     # only if a docstring grew past four lines
```

The unit selection above deliberately does **not** exclude `chaos`: the non-integration chaos tests run in the Ubuntu and Windows unit lanes in CI, while a bare `pytest` on your machine skips them. Run the command exactly as written.

Tiers that need more: `pytest -m integration` (Docker; Qdrant and pgvector start themselves, Chroma is embedded, Milvus and Weaviate need `docker compose -f deploy/compose.test-backends.yml up -d` and otherwise skip), `pytest -m eval` (downloads models), and the billed provider probes, which also need `FASTERRAG_LIVE_PROVIDER_TESTS=1`.

### 0.4 Tripwires — tests and gates that fire when you add a thing

These are the checks that have failed real changes. Find the row for what you are adding before you add it.

| You add or change | What else must change, or a test fails |
|---|---|
| A config key | The field in `src/fasterrag/config/schema.py` (models are `extra="forbid"`); the same key and its default in `config.yaml`; a row in [config-reference.md](config-reference.md) — `tests/unit/docs/test_reference_parity.py` and `tests/unit/config/test_canonical_config.py` compare all three. The key must be *read* somewhere in `src/` (`tests/unit/config/test_settings_are_wired.py`; a leaf name that collides with an existing attribute passes vacuously, so check by hand). |
| A key under `retrieval.*`, `chunking.*` or `embeddings.*` | All of the above, **plus** classify it in `COMPARABILITY_FIELDS` or `BEHAVIOUR_NEUTRAL_FIELDS` with a `MUTATIONS` entry (`tests/unit/services/test_comparability.py`). A compared field changes the eval baseline fingerprint, so `tests/eval/datasets/handbook/baseline.json` must be re-recorded or the `eval-gate` job reports `NOT_COMPARABLE`. It also moves `retrieval_config_hash` (`core/identity.py:177`), so every existing `index.lock` reports drift — say so in the changelog. |
| A CLI flag or command | [cli-reference.md](cli-reference.md) (`tests/unit/docs/test_cli_flag_parity.py`). |
| A REST route | [api-reference.md](api-reference.md) (`scripts/check_openapi_drift.py`). `DELETE` and other mutating verbs are picked up by the idempotency middleware (`api/idempotency.py:52`). |
| A request-model field | A handler must read it (`tests/unit/api/test_schema_fields_are_wired.py`). |
| An `ErrorCode` | `PROBLEM_SPECS` in `src/fasterrag/errors.py` (`tests/unit/test_errors.py:69` asserts the two sets are equal), the hard-coded `DOCUMENTED_STATUSES` dict at `tests/unit/test_errors.py:22`, and the error table in [api-reference.md](api-reference.md). `scripts/check_error_taxonomy.py` gates how codes are raised. |
| A facade method or parameter | Both `src/fasterrag/facade.py` and `src/fasterrag/sync.py`, and the member table in [python-api.md](python-api.md) (`tests/unit/docs/test_python_api_parity.py` compares against `inspect.signature`; `tests/unit/test_sync_facade.py::test_it_mirrors_the_async_surface`). |
| A `Journal` method | Keep the two-faced shape: an `async` method plus a `*_blocking` twin. `tests/unit/test_no_blocking_journal_in_async.py` enumerates them with `vars(Journal)`. |
| An adapter or factory | `_BUILTIN_ADAPTERS` must stay an annotated dict *literal* in `adapters/<kind>/factory.py` — `scripts/check_unbuilt_claims.py` reads it with `ast`. `tests/unit/docs/test_python_api_parity.py` requires every `ENTRY_POINT_GROUP` and abstract member to appear in [python-api.md](python-api.md). |
| A degradation rung | `_BUILTIN_RUNGS` / `_PLANNED_RUNGS` in `services/degradation.py` (import-time check, and the same gate script). |
| A chaos scenario | `tests/chaos/test_chaos.py::test_every_scenario_defined_in_this_module_is_scripted` hard-codes the scenario list. |
| A row or proof in [failure-modes.md](failure-modes.md) | The proof column must cite a test that exists (`tests/unit/docs/test_fmea_parity.py`). |
| A file or directory | [structure.md](structure.md) — the gate checks every listed path exists and that nothing marked *(pending)* is present. |
| A number in a document | If the tree can count it, a test may pin it (`tests/unit/docs/test_counts_parity.py`). Prefer not stating counts. |
| A superlative ("fastest", "most reliable") | It must cite a `BENCH-` entry or be phrased as a goal (`tests/unit/docs/test_superlative_scan.py`). |
| A sentence saying something is not built | It must cite an **open** task (`scripts/check_claims.py`), and must not name a registered adapter, an existing test tier or a built rung (`scripts/check_unbuilt_claims.py`). |
| A prompt's wording | `P1_TEMPLATE_VERSION` / `P3_TEMPLATE_VERSION`, the verbatim copies in [prompts.md](prompts.md), and the wording asserts in `tests/unit/core/test_generation.py:26-29` and `tests/unit/core/test_faithfulness.py:47-53`. |
| A function or class docstring over four lines | Run `scripts/trim_docstrings.py --apply src`; it moves the prose to [code-notes.md](code-notes.md). |
| A comment | Only `# CRITICAL:`, `# TODO:` and `# BLOCKED:` (plus continuation lines) are allowed in `src/` and `scripts/`. A `# TODO:` may only cite an open task. |
| A test that must run somewhere | Mark it `must_run`; `scripts/check_must_run.py` fails the build if it skipped in both the Ubuntu and Windows lanes. |

### 0.5 Platform traps

- **Windows uses `spawn`** for process pools: a child re-imports the module, so the target of a pool task must be a module-level function, and a `monkeypatch` applied in the parent does not exist in the child.
- **Windows has no `SIGKILL`** and cannot `fsync` a directory; `os.replace` onto a file another handle holds open can fail. Any file-locking or durability code needs a Windows branch and a test in the Windows lane.
- **`asyncio.CancelledError` is a `BaseException`.** `except Exception` does not catch it, which is why cancellation paths in this codebase currently settle nothing.

### 0.6 Where the test doubles live

| Double | Location |
|---|---|
| Thread-backed executor factory `threads()` | `tests/unit/services/test_ingestion.py:38`, `tests/unit/workers/test_cpu_pool.py:38` |
| `StubEmbedder` (ingestion) | `tests/unit/services/test_ingestion.py:43` |
| `RecordingAdapter` (vector database that records upserts) | `tests/unit/workers/test_indexer.py:25` |
| `MemoryVectorDB`, `MemoryEmbedder`, `MemoryRouter` | `tests/unit/api/test_query_tenancy.py:58`, `:123`, `:137` |
| `ScriptedLLM`, `ScriptedRetrieval`, `ScriptedEmbedder` | `tests/unit/services/test_generation.py:26`, `:64`, `:486` |
| `ScriptedAdapter`, `ScriptedReranker` | `tests/unit/services/test_querying.py:58`, `:335` |
| `FakeCrossEncoder` | `tests/unit/core/rerank/test_cross_encoder.py:12` |
| `FakeRedis`, `BrokenRedis`, `FakePipeline` | `tests/unit/core/cache/test_redis_store.py:20`, `:74`; `tests/unit/core/test_rate_limit.py:35`, `:55` |
| `FakeJournal(Journal)` | `tests/unit/cli/test_pipeline_commands.py:546`; everywhere else use a real `Journal(tmp_path / ...)` |
| Vector contract base class `VectorDBContract` | `tests/contract/vectordb.py:52` — subclass it and supply the `adapter`, `collection` and `hybrid_collection` fixtures |
| API test client fixtures | `tests/unit/api/conftest.py` |

Process-wide singletons (`shared_breaker`, the cross-encoder model cache, `shared_governor`) are reset around every test by an autouse fixture in `tests/conftest.py`. **If you add a `shared_*` function, add its `reset_*` twin and call it there**, or tests will leak state into each other in an order-dependent way.

### 0.7 The shape of a brief

**Goal** — the behaviour that must be true afterwards. **Where** — the code as it is today. **Steps** — the order to do the work in. **Tests** — what to write, and existing tests that pin the old behaviour and must change. **Documents** — what to update. **Traps** — what has bitten or will bite. **Done when** — the acceptance check. **Moves** — the scorecard categories this is expected to raise.

---

## 1. P0 — blocks a credible public alpha

Recommended order inside P0: TASK-0470 and TASK-0473 (small, and they stop data loss and hangs), then TASK-0436, TASK-0437, TASK-0443, TASK-0438, TASK-0441, TASK-0439, TASK-0442, TASK-0431, TASK-0440, TASK-0444, TASK-0445.

### TASK-0431 — Semantic chunking in the ingestion pipeline

**Goal.** A job with `chunking.strategy: semantic` and a local embedding model completes, with every chunk inside the token budget.

**Where.** `workers/cpu_pool.py:98` `_parse_and_chunk(task, settings)` calls `create_chunker(settings)` with no embedder. `core/chunking/__init__.py:55` `create_chunker(settings, *, counter=None, embedder=None)` raises `ConfigError` at `:76-80` for the semantic strategy without one. `services/ingestion.py:373` `_require_pipeline_chunker`, called from `IngestionService.accept` at `:99`, refuses the job up front. `core/chunking/semantic.py:110` `SemanticChunker`; its `split` at `:132` never calls `within_budget` (`core/chunking/models.py:209`) and falls back to `hard_split` with a character limit at `:146-147`. The `SentenceEmbedder` protocol is at `semantic.py:49`. A bridge already exists and is called by nothing in the pipeline: `adapters/embeddings/sync.py:25` `SentenceEmbedderBridge`.

**Steps.**
1. Fix the budget first, independently: make `SemanticChunker.split` enforce `within_budget` the way `recursive.py:108` does. This is a self-contained commit.
2. Give the worker process an embedder. The pool task receives only `(task, settings)` through `run_in_executor`, so a model object cannot be passed in: build it lazily inside the child, once per process, in a module-level cache keyed by model name. Support `embeddings.provider: huggingface` only.
3. For hosted providers, keep a refusal at `accept` — narrowed to "semantic chunking needs a local embedding model" — or move boundary detection to the async side the way contextual enrichment (`core/chunking/enrichment.py`) already is. Pick one and record why in the changelog.
4. Add `semantic_percentile` and the sentence-embedder model to `chunker_config_hash` (`core/identity.py:163`): today it covers only strategy, size and overlap, so boundaries can move under unchanged chunk ids. This changes ids for semantic collections only; see the tripwire for `chunking.*`.
5. Delete `_require_pipeline_chunker` or reduce it to the hosted-provider case.

**Tests.** `tests/unit/services/test_ingestion.py::test_semantic_chunking_is_refused_before_a_job_is_recorded` (`:502`) pins the refusal and must be inverted. Add a job-level test in the same file using a deterministic embedder. In `tests/unit/core/chunking/test_invariants.py`, `dense_chunkers()` (`:195-202`) excludes the semantic chunker from the token-budget property — include it (the `RotatingEmbedder` at `:34` is the double to use).

**Documents.** [troubleshooting.md](troubleshooting.md) (the "refused with `chunking.strategy is 'semantic'`" entry at `:48`), [config-reference.md](config-reference.md) `## chunking`, [architecture.md](architecture.md) §5, [code-notes.md](code-notes.md) `### _require_pipeline_chunker`.

**Traps.** Under `spawn`, each worker process loads its own copy of the model: memory is `cpu_pool_size ×` model size. Say so in the config reference and consider capping the pool for this strategy.

**Done when.** The job-level test passes, the property test includes the semantic chunker, and the refusal is gone for local models. **Moves:** ingestion.

### TASK-0436 — Resume an interrupted job; drain on shutdown

**Goal.** After any process exit, a restart leaves no job in `running`, and no document both absent from the collection and present in the dedup file.

**Where.** `Journal.resume_index` (`services/journal.py:369`) is read only at `services/ingestion.py:161`, inside `_run`; the status becomes `running` at `:163-164`. `api/ingest.py:175` schedules a run only when `record.status in {"queued"}`, so replaying the idempotency key of a stuck job returns the stuck record. Background work is `BackgroundTasks.add_task(_run_job, ...)` at `api/ingest.py:179` and `:245`: no registry and no task handle exists. `_ensure_settled` (`:80`) and `IngestionService.run` (`:112`) settle `failed` on `Exception`, not on cancellation. `api/main.py:137` `lifespan` closes the metric pusher, embeddings, cache, idempotency store and adapter (`:167-193`) and waits for nothing. The journal is created lazily in `api/dependencies.py:60-65`, not in the lifespan. The `ingest` parser (`cli/parser.py:97-112`) has no `--resume`.

**Steps.**
1. **Track the work.** Replace `BackgroundTasks` for ingestion with tasks created through a small registry on `app.state` (a set of `asyncio.Task`, discarded on completion). Everything below depends on being able to enumerate in-flight jobs.
2. **Settle on cancellation.** In `IngestionService.run` and `_run_job`, catch `asyncio.CancelledError`, settle the job `failed` with reason `INTERRUPTED` from the last checkpoint, and re-raise.
3. **Drain.** Add `reliability.shutdown_grace_seconds`. In the lifespan shutdown: stop accepting jobs (a flag `POST /v1/ingest` checks, answering 503 with a retryable problem), wait up to the grace period for the registry to empty, then cancel what remains and await it. Only then close the adapters.
4. **Reap at startup.** Create the journal in the lifespan. Find jobs whose status is `running` and whose last checkpoint is older than `reliability.stale_job_seconds`; settle each `failed`/`INTERRUPTED`. Do not auto-resume in this change — resuming needs the original sources, which for inline payloads no longer exist.
5. **Resume on request.** `fasterrag ingest --resume <job_id>` and, over REST, replaying the idempotency key of an `INTERRUPTED` job re-runs it from `resume_index`.

**Tests.** Unit: cancel a running job and assert `failed`/`INTERRUPTED`; a lifespan test that starts a slow job, begins shutdown and asserts it settles inside the grace period; a startup-reaper test over a journal directory prepared with a `running` record. Existing tests to keep green: `test_resuming_skips_what_a_checkpoint_already_covered` (`tests/unit/services/test_ingestion.py:313`), `test_the_watermark_starts_where_a_resumed_job_does` (`tests/unit/workers/test_ledger.py:169`), `test_the_journal_resumes_from_its_checkpoint` (`tests/chaos/test_chaos.py:367`). Chaos: a subprocess test that kills the process mid-job and restarts; on Windows use `Popen.kill()`, there is no `SIGKILL`.

**Documents.** [failure-modes.md](failure-modes.md) row 31, [reliability.md](reliability.md) §3, [differentiators.md](differentiators.md) §D3, [api-reference.md](api-reference.md) `GET /v1/ingest/{job_id}`, [cli-reference.md](cli-reference.md) `fasterrag ingest`, [architecture.md](architecture.md) §12, [config-reference.md](config-reference.md).

**Traps.** Two new config keys and one CLI flag: see §0.4. A new chaos scenario must be added to the hard-coded list. `ReliabilitySettings` (`schema.py:651-658`) has no journal-root key either — `DEFAULT_JOURNAL_ROOT` is a constant at `journal.py:52`. TASK-0130 (the queue backend) is open; do not design this so that it must be rewritten when that lands — the registry is the seam.

**Done when.** The kill-and-restart test passes in the Ubuntu and Windows lanes. **Moves:** reliability, ingestion, architecture.

### TASK-0437 — A journal that survives a crash and two processes

**Goal.** No interleaving of writes and crashes produces a missing or torn job record, and looking up a job by idempotency key does not read every job file.

**Where.** `services/journal.py:143` `_write_atomically(path, payload)` does, in order: compute `path + ".tmp"`; create the parent; **if the live file exists, `os.replace(path, path + ".prev")`** (`:148-149`); write the temporary file; `os.replace(tmp, path)`; on `OSError`, unlink the temporary file and raise `IngestionError(INTERNAL)`. Between the third and fifth step there is no live file, and `load_job_blocking` (`:286`) checks `path.exists()` first (`:289`) and raises `NOT_FOUND` even though `.prev` is intact. Nothing in `src/` calls `fsync` or takes a file lock. Appends (`record_document_blocking` `:379-385`, `remember_content_blocking` `:503-509`) use `open("a")` unlocked and are **not** wrapped, so a raw `OSError` escapes the taxonomy (TASK-0475). `find_by_idempotency_key_blocking` (`:321`) globs and reads every `jobs/*.json`; `known_content_blocking` (`:515`) reads the whole dedup file, once per `process()` (`workers/cpu_pool.py:235`). File layout: `jobs/<id>.json`, `jobs/<id>/documents.jsonl`, `collections/<c>/content-hashes.jsonl` (`:54-58`). The full public surface — `create_job`, `save_job`, `load_job`, `find_by_idempotency_key`, `checkpoint`, `resume_index`, `record_document`, `record_indexed`, `dead_letter`, `documents`, `dead_lettered`, `remember_content`, `known_content`, `document_hashes`, `is_duplicate`, `counts` — is at `:188-550`, with `create_journal(settings, root)` at `:559`.

**Steps.**
1. **Decide the backend and record it as an ADR.** Either (a) SQLite in WAL mode behind the existing `Journal` interface — one writer, indexed lookups, atomic by construction, in the standard library; or (b) keep files and fix them: write the temporary file, `fsync` it, `os.replace` over the live file (never rename the live file away first), `fsync` the directory where the platform allows, take an advisory lock around appends, and keep an index file for idempotency keys. (a) is the smaller amount of code that can be wrong.
2. Whichever is chosen, fix the read side first as its own commit: `load_job` must fall back to `.prev` when the live file is missing, not only when it is unreadable.
3. Make the idempotency lookup indexed, then re-run `scripts/measure_idempotency_reclaim.py` and add a new `BENCH-` entry (do not edit BENCH-0004; ledger entries are append-only).
4. If the backend changes, write a one-shot migration from the file layout and a test for it. Operators have journals on disk.

**Tests.** A Hypothesis stateful test that interleaves writes, simulated crashes (an injected failure between any two file operations) and reads, asserting a reader never sees a missing or torn record. Keep green: `test_a_torn_write_falls_back_to_the_previous_good_record` (`tests/unit/services/test_journal.py:91`), `test_writes_never_leave_a_temporary_file_behind` (`:127`), `test_a_full_filesystem_becomes_a_typed_error_not_a_raw_oserror` (`:218`), and the class `TestTheAsyncFaceKeepsFileIoOffTheEventLoop` (`:254`). A two-process test (subprocesses appending to one job) for the lock.

**Documents.** [failure-modes.md](failure-modes.md) rows 32 and 33, [reliability.md](reliability.md) §3, [benchmarks.md](benchmarks.md), a new ADR, [disaster-recovery.md](disaster-recovery.md) if the on-disk layout changes.

**Traps.** [api-reference.md](api-reference.md) `:113` states that the idempotency-key scan "will not be indexed; it retires with the field at 1.0 (TASK-0358)". That sentence and this task disagree: read TASK-0358 and resolve it in the same change, in whichever direction the maintainer decides. `tests/unit/test_no_blocking_journal_in_async.py` requires the two-faced method shape on any replacement. Windows cannot `fsync` a directory.

**Done when.** The stateful test passes, the lookup is indexed, and the new ledger entry exists. **Moves:** reliability, performance, architecture.

### TASK-0438 — One query pipeline behind `answer()` and `stream()`

**Goal.** Every degradation, refusal, flag, metric and trace is identical whether a query is streamed or not.

**Where.** `services/generation.py`: `answer` at `:583`, `stream` at `:737`, shared `_prepare` at `:385`. The verified differences:

| Step | `answer()` | `stream()` |
|---|---|---|
| Budget check | `_charge_budget` at `:619`, before anything is returned | Yields `meta` first (`:769-776`), then charges at `:778` — so a refusal reaches the client as an SSE `error` over HTTP 200 (`api/query.py:67-86`) |
| Retryable provider failure | Degrades to `extractive_answer`, increments `DEGRADED_RESPONSES` (`:656`), returns citations (`:657-663`) | Yields `error` and returns (`:811-819`): no fallback, no metric, no `done`, no trace |
| Token counts | From the provider | Local estimate (`:826`); the provider's usage chunk is discarded |
| `generation` span | Recorded (`:670`) | Missing |
| Refusal (`grounded_or_refuse`) | Stores a trace (`:681`) | Yields `insufficient_evidence` then `done`, stores no trace (`:833-835`) |
| `truncated`, `evidence_dropped`, `estimated_cost_usd` on the `Answer` | Set (`:715-716`) | Absent from the streamed `Answer` (`:872-881`) |

SSE event names, serialised by `api/query.py:51` `encode_event`: `meta`, `token`, `citations`, `usage`, `done`, `error`, `insufficient_evidence`.

**Steps.**
1. Move the budget charge in `stream()` above the first `yield`. One commit, one test, and it fixes the 402-as-200 defect on its own.
2. Introduce one private async generator that yields typed stage results (prepared, token delta, completed, degraded, refused) and owns breaker accounting, metrics, spans and the trace.
3. Re-express `stream()` as a mapper from stage results to `QueryEvent`, and `answer()` as a consumer that collects them into an `Answer`.
4. Give the streamed path the extractive rung: on a retryable failure before the first token, emit the extractive answer as `token` events followed by `citations`, `usage` and `done`, with `mode` set in `meta` or `usage`. After the first token a fallback cannot un-send text: emit `error`, and document that boundary.

**Tests.** One parametrised test that runs each case — success, cache hit, retryable failure, non-retryable failure, budget refusal, grounded refusal, truncation, evidence dropped — through both surfaces and compares mode, flags, metric deltas and the stored trace. `test_a_retryable_stream_failure_still_emits_an_error_event` (`tests/unit/services/test_generation.py:197`) pins the old behaviour and must be rewritten. `test_a_streamed_query_over_the_budget_is_refused_before_any_token` (`:898`) must additionally assert that no `meta` event was produced. Keep `test_a_streamed_trace_matches_the_non_streamed_one` (`:1036`) and `tests/chaos/test_chaos.py:308`.

**Documents.** [api-reference.md](api-reference.md) "Streaming semantics (SSE)" (`:218-242`), [reliability.md](reliability.md) §2, [prompts.md](prompts.md) P1 "Streaming".

**Traps.** `api/query.py:67` also catches a `FasterRagError` raised mid-generator; keep that path. The `usage` event's shape is documented and parity-checked.

**Done when.** The parametrised parity test passes. **Moves:** core RAG, reliability, architecture.

### TASK-0439 — Plugins that configuration can select

**Goal.** A third party installs a package, names its provider in `config.yaml`, and it loads — for vector databases, embedders, LLMs, chunkers, parsers and rerankers — with no edit to this repository and no weakening of fail-fast validation.

**Where.** `config/schema.py:59-61` declares `VectorDbProvider`, the embeddings provider and the LLM provider as closed `Literal`s, used at `:253`, `:352`, `:388` and in `TierRule.provider` (`:323`); section models are `extra="forbid"` (`:121`). `VectorDbProvider` includes `"pinecone"`, which validates and then fails in the factory. The three factories already resolve entry points: `adapters/vectordb/factory.py` (group `fasterrag.vectordb` `:30`, `_BUILTIN_ADAPTERS` `:34-40`, `_PLANNED_ADAPTERS` `:43`, `resolve_adapter_class` `:78`), `adapters/embeddings/factory.py` (`fasterrag.embeddings` `:26`), `adapters/llm/factory.py` (`fasterrag.llm` `:26`). No plugin point exists for chunkers (`core/chunking/__init__.py:55-87` is an if-chain), parsers (`core/parsing/__init__.py:80` `_PARSERS`, `:93` `_TUNED_PARSERS`) or rerankers (`api/dependencies.py:126` constructs `CrossEncoderReranker` directly). [python-api.md](python-api.md) `:235` promises `provider: mydb`; `:222` carries a status note admitting it fails.

**Steps.**
1. In `config/`, add a helper that returns the names registered under an entry-point group using `importlib.metadata` only. `config/schema.py` imports nothing from `fasterrag` today and must not start importing `adapters/`.
2. Change the three provider fields from `Literal` to `str` with a validator: accept a built-in name or a registered entry-point name; otherwise fail at load, naming the installed alternatives. Keep the built-in name list in `config/` as plain strings and add a test asserting it equals each factory's `_BUILTIN_ADAPTERS` keys, so the two cannot drift.
3. Remove `pinecone` from the accepted names until TASK-0049 lands, or keep it and make the validator refuse planned adapters with the task id.
4. Revisit `PROVIDERS_REQUIRING_EMBEDDING_KEY` and `PROVIDERS_NOT_REQUIRING_LLM_KEY` (`:65-66`): a plugin name currently falls into "requires `llm.api_key_env`". Let a plugin declare whether it needs a key.
5. Repeat the factory pattern for `fasterrag.chunkers`, `fasterrag.parsers` and `fasterrag.rerankers`, one commit each, replacing the if-chain, the private dicts and the hard-wired constructor.

**Tests.** `tests/unit/adapters/test_plugin_entry_points.py` calls `resolve_adapter_class` directly (`:41`) and never touches `Settings`. Add the end-to-end case: monkeypatch `_plugin_entry_points` to return an `EntryPoint` for a fake adapter, load a `config.yaml` naming it through the real loader, build the service, serve a query. Add the negative case: an unknown name fails at load and the message lists what is installed.

**Documents.** [python-api.md](python-api.md) "Extending the framework (plugin contract)" (`:218`) — remove the status note — and "Stability guarantees" (`:237`); [architecture.md](architecture.md) §10; [internals.md](internals.md) §5; [config-reference.md](config-reference.md).

**Traps.** `config/` is a sensitive folder: fail-fast must not weaken, and this task is the explicit authorisation to touch it — nothing wider. `scripts/check_unbuilt_claims.py:239-274` reads every `adapters/**/factory.py` with `ast`; a new `adapters/rerankers/factory.py` joins that gate automatically. `tests/unit/docs/test_python_api_parity.py:302-311` requires every new group in the docs. `vectordb/factory.py`'s `__all__` omits `resolve_adapter_class` while the other two export it — make them consistent.

**Done when.** The end-to-end plugin test passes for all six kinds. **Moves:** extensibility, architecture. Unblocks TASK-0462, TASK-0458 (hosted rerankers), TASK-0486.

### TASK-0440 — Process surfaces that do what the documents say

**Goal.** Every process, flag and setting the documents name either does the work described or does not exist.

**Where.** `cli/commands/processes.py:106` `run_worker` builds an `IngestionService` (`:120`), calls `adapter.health()` (`:122`), prints "worker ready" (`:137`) and then `await asyncio.Event().wait()` (`:140`). `run_serve` (`:42`) starts one `uvicorn.Server` (`:81`). `AppSettings.workers` (`schema.py:129`) is read nowhere. `docker-compose.yml:72-75` and `deploy/compose.{small,medium,large}.yml` each run a `worker` service. Status notes already exist in [python-api.md](python-api.md) `:216`, [deployment.md](deployment.md) `:6`, [cli-reference.md](cli-reference.md) `:56` and [config-reference.md](config-reference.md) `:211`.

**Steps.**
1. This is a **maintainer decision**, shared with TASK-0130: build the queue-backed worker, or remove the surface. Ask before writing code; record the answer in both task entries.
2. If *remove*: delete the `worker` subcommand, `app.workers`, the worker services in all four compose files, and every sentence describing them. Keep a changelog entry under `Removed`.
3. If *build*: that is TASK-0130; this task then reduces to making the documents match.
4. Either way, add a process registry the gate can read — a literal mapping of documented process surfaces to the callable that implements each, like `_BUILTIN_RUNGS` — and teach `scripts/check_unbuilt_claims.py` to compare the documents against it.

**Tests.** `tests/unit/cli/test_processes_commands.py` has three worker tests (`:163`, `:233`, `:246`) that will be deleted or rewritten. Add a test for the registry gate in `tests/unit/scripts/test_check_unbuilt_claims.py`.

**Documents.** All four above, plus [architecture.md](architecture.md) §2 and §11, [cookbook.md](cookbook.md), `README.md`, `CLAUDE.md` (the "Run workers" line).

**Traps.** `tests/unit/config/test_settings_are_wired.py:59-69` passes `app.workers` only because the leaf name `.workers` appears elsewhere in `src/` — the test cannot see this defect.

**Done when.** No document names a process that does nothing, and the gate decides it from the registry. **Moves:** architecture, documentation.

### TASK-0441 — Cohere and Ollama adapters against the real SDKs

**Goal.** Each LLM and embedding adapter is proven against a response recorded from the real client library. **NOT VERIFIED — run the experiment first.**

**Where.** `adapters/llm/ollama.py:67` and `:81` gate on `isinstance(response, dict)`; the pin is `ollama>=0.3` (`pyproject.toml:60`, `:77`), and newer clients return typed response objects. `adapters/llm/cohere.py:30` imports `AsyncClient` (the v1 client) but calls `.chat(messages=…)` (`:57`) and `.chat_stream` (`:82`) and parses v2 shapes (`:84-87`, `:121-126`); the pin is `cohere>=5.5`. A `TypeError` has no `status_code`, so `classify_llm_failure` (`adapters/llm/base.py:79`) reports it as unreachable and retryable (`:118`) — a programming error would look like an outage. `adapters/embeddings/ollama.py:47` reads its host from `settings.llm.base_url`, `:78` calls the per-text `.embeddings(model, prompt)` route, `:92` checks for a dict. **Correction to the task entry:** the Cohere and Ollama *LLM* call paths have no call-path test at all — `tests/unit/adapters/llm/test_call_paths.py` fakes only OpenAI (`FakeCompletions` `:55`). The hand-written fakes exist for *embeddings* (`tests/unit/adapters/embeddings/test_usage.py`: `FakeCohereClient` `:72`, `FakeOllamaClient` `:97`, which returns a dict).

**Steps.**
1. Install the currently resolved versions (`uv.lock`) and, with no network, construct each client and feed it a recorded HTTP response through an `httpx` mock transport. Record what the client actually returns. Write down the outcome even if it is "this was not a bug".
2. Fix the shapes that are wrong. For Cohere choose the v2 client explicitly.
3. Give the Ollama embedder its own base-URL setting (or document that it shares the LLM's) and move to the batch route.
4. Pin the lower bounds to what the code supports and add an upper bound if a major version is known to break.
5. Make the fakes faithful to the recordings, and add call-path tests for both LLM adapters mirroring the OpenAI ones.
6. Make a `TypeError`/`AttributeError` from an SDK call a non-retryable `INTERNAL` error.

**Tests.** A contract-style test per adapter that imports the real client library and replays a recorded response; skip with a reason when the extra is not installed and mark it `must_run`. The billed leg lives in `tests/eval/test_live_providers.py` (`:413`).

**Documents.** [integrations.md](integrations.md), [config-reference.md](config-reference.md) `llm`, [architecture.md](architecture.md) "Embedding and LLM factories".

**Done when.** Both adapters pass against recorded real responses. **Moves:** core RAG, testing, embedding infrastructure.

### TASK-0442 — Wire the embedding cache, correctly

**Goal.** Re-ingesting an unchanged corpus with dedup off makes zero provider calls, and a query-type and a document-type embedding of the same text never collide.

**Where.** `core/cache/embedding.py:62` `CachingEmbeddingAdapter` and `core/cache/__init__.py:67` `create_embedding_store` have no call site in `src/`; the package docstring (`core/cache/__init__.py:3`) says the cache is on by default. The key (`embedding.py:40`) is `emb:{model}:{model_version}:{sha256}` — it omits the provider, the configured dimensions and the input type, and `embed_query` uses the same scheme (`:148-162`). Gets and sets are per text and sequential (`:116-146`). `DiskStore._evict` globs the directory on every write (`core/cache/store.py:202-226`). The wiring point is `create_embedding_router` in `adapters/embeddings/tiering.py:120`. Settings: `EmbeddingCacheSettings` (`schema.py:304-316`, `enabled=True`, `backend="disk"`).

**Steps.** In this order, each its own commit: (1) fix the key — add provider, dimensions and input type; (2) batch the store calls (`get_many`/`set_many` on `CacheStore`, pipelined in `RedisStore`); (3) make eviction amortised, not per write; (4) wrap the adapters in `create_embedding_router` when `embeddings.cache.enabled`; (5) correct the package docstring. Also remove the second embedding of the question: a cache-enabled query currently embeds it twice (`services/generation.py:377` and `services/querying.py:235`).

**Tests.** Existing: `tests/unit/core/cache/test_embedding.py` (`test_the_key_separates_models` `:93`). Add: key separates input types, providers and dimensions; a re-ingest unit test counting provider calls; a query test asserting one embedding call.

**Documents.** [architecture.md](architecture.md) §7 (remove the status note), [config-reference.md](config-reference.md) `:252-253`.

**Traps.** `core/cache/embedding.py` imports from `adapters/` — one of the cycle edges TASK-0456 removes. Wiring the cache inside `adapters/embeddings/tiering.py` adds an `adapters → core` edge; put the wrapper where the composition lives instead. `embeddings.cache.enabled` passes the wired-settings test only by a leaf-name collision.

**Done when.** The zero-provider-call test passes. **Moves:** performance, embedding infrastructure.

### TASK-0443 — Delete and update a document

**Goal.** A document can be removed, and re-ingesting a document that shrank leaves no stale chunk.

**Where.** Nothing in `services/`, `workers/`, `api/` or `cli/` calls `VectorDBAdapter.delete`. The interface has it: `adapters/vectordb/base.py:321` `delete(selector: PointSelector) -> None`, with `PointSelector(collection, point_ids=None, filters=None)` at `:190` (exactly one of the two). A point's id is its `chunk_id` (`workers/indexer.py:127`) and its payload carries `document_id` (`indexer.py:43`), so delete-by-filter on `document_id` works today. The contract suite already proves `test_delete_by_ids_removes_only_those_points` (`tests/contract/vectordb.py:315`) and `test_delete_by_filter_removes_matching_points` (`:327`). The dedup file is append-only (`journal.py:503-509`). `SemanticCache.invalidate` (`core/cache/semantic.py:211`) is called only at `services/ingestion.py:244`.

**Steps.**
1. Service: `services/documents.py` with `delete(collection, document_id, tenant)` — delete by payload filter, write a tombstone to the journal, invalidate the semantic cache.
2. Dedup: make `known_content` and `document_hashes` honour tombstones, so removed content can be ingested again and the lockfile stops listing it (`services/ingestion.py:296`).
3. Shrinking re-ingest: when a known `document_id` is committed with *n* chunks, delete points of that document whose `chunk_index >= n`. Do it in the ledger's commit callback, after the new chunks are written.
4. Surfaces: `DELETE /v1/collections/{name}/documents/{document_id}`, `fasterrag documents delete`, `FasterRag.delete_document` on both facades.
5. Add a contract case: delete by `document_id` filter removes every chunk of that document and no other.

**Tests.** Service unit tests with `RecordingAdapter`; a shrinking-document test asserting no chunk index ≥ the new count remains; a tombstone round-trip test; the new contract case on all five adapters.

**Documents.** [api-reference.md](api-reference.md), [cli-reference.md](cli-reference.md), [python-api.md](python-api.md), [data-model.md](data-model.md), [architecture.md](architecture.md) §9, [disaster-recovery.md](disaster-recovery.md) "The re-ingest trap".

**Traps.** A new route, a new command and two new facade methods: three rows of §0.4. Tenant scoping applies — a tenant must not delete another tenant's document; read TASK-0339 and TASK-0381 first. The contract case is proven in CI only on Qdrant, pgvector and Chroma until TASK-0412 lands; run it locally against the other two.

**Done when.** The contract case passes on all five adapters and the shrinking-document test finds no stale chunk. **Moves:** core RAG, ingestion.

### TASK-0444 — A first run that needs nothing but `pip`

**Goal.** In a clean virtual environment with no network services, the local profile ingests a Markdown file and answers a `retrieve` call.

**Where.** Mandatory dependencies (`pyproject.toml:32-48`): `beautifulsoup4`, `fastapi`, `httpx`, `lxml`, `markdown-it-py`, `pdfplumber`, `psutil`, `python-docx`, `pydantic`, `pydantic-settings[yaml]`, `python-dotenv`, `pyyaml`, `qdrant-client`, `snowballstemmer`, `uvicorn[standard]`. The default configuration (`config.yaml`) uses Qdrant in Docker mode, a Hugging Face embedder and a reranker — the last two need extras the base install does not include. **Correction to the task entry:** the default configuration demands three environment variables, not two — `QDRANT_API_KEY`, `OPENAI_API_KEY` and `FASTERRAG_API_KEY` — because `config/loader.py:240` `_require_referenced_env_vars` requires every `*_env` field and `security.api_key_env` is a plain `str` (`schema.py:726`), required even with `security.auth: false`. `fasterrag config init` (`cli/commands/diagnostics.py:52`) has only `--path` and `--force`. An embedded Chroma adapter exists (`ChromaSettings`, `schema.py:218`).

**Steps.**
1. Make an unset `*_env` variable an error only when the feature that reads it is enabled (auth off ⇒ no API key needed; a keyless Qdrant ⇒ no Qdrant key).
2. Add `--profile local` to `config init`, writing a **second packaged template**: embedded Chroma (or the in-process store from TASK-0464), a small local embedding model, no reranker, no keys.
3. Slim the core. Move the web server to `[server]`, the Qdrant client to `[qdrant]`, and the parsers to `[pdf]`, `[docx]`, `[html]`; make each import lazy and raise an error that names the extra. Keep `all`.
4. Extend `scripts/check_package.py` to run the local profile end to end in the clean venv it already builds.

**Tests.** `tests/unit/cli/test_config_init.py` (thirteen tests; `test_the_written_config_actually_loads` sets environment variables with `monkeypatch`). `tests/unit/scripts/test_check_package.py`. Add import-error tests asserting the message names the extra.

**Documents.** [quickstart.md](quickstart.md), [python-api.md](python-api.md) "Installation", `README.md`, [release.md](release.md) §3.

**Traps.** These are the reasons this task is larger than it looks. `facade.py:181`, `:191` and `:214` lazily import `fasterrag.api.dependencies`, which imports `fastapi` — so the library facade needs the web framework; do TASK-0456's `services/composition.py` first. `cli/main.py:36` imports `cli/commands/processes.py`, which imports `fasterrag.api.main` at module top, so the whole CLI needs it too. `core/parsing/__init__.py:20-37` imports every parser eagerly, and `PUBLIC_IMPORTS` in `scripts/check_package.py:19-28` includes `fasterrag.parsing`. The packaged `config.yaml` must stay byte-identical to the repository copy, which is why the local profile is a second template and not an edit. `adapters/embeddings/huggingface.py:35-39` tells the user to `pip install sentence-transformers`, not the extra.

**Done when.** `scripts/check_package.py` proves the local profile in a clean venv. **Moves:** developer experience, packaging.

### TASK-0445 — The first release

**Goal.** A tagged pre-release exists on PyPI, installed and exercised on a clean machine by the pipeline that published it.

**Where.** Version: `src/fasterrag/__init__.py:10` `__version__ = "0.1.0.dev0"`, read by hatch. Classifier `2 - Pre-Alpha` (`pyproject.toml:22`). Workflows: `ci.yml`, `nightly.yml`, `security.yml` — none publishes. [release.md](release.md) §3 describes a manual, token-based flow (`twine upload`). `scripts/stamp_release.py` keeps the version, the changelog heading, the classifier and three banners coherent; `--check` runs in CI.

**Steps.**
1. Reconcile the version with the maintainer: the task entry says `v0.1.0a1`; the release documents have decided `0.1.0-beta.1`. `stamp_release._SEMVER` accepts `alpha|beta|rc` labels, so an alpha is stamped as `0.1.0-alpha.1`.
2. Add `.github/workflows/release.yml`, triggered by a `v*` tag: run the full gate set, build the sdist and wheel, publish through PyPI trusted publishing (OIDC, no stored token), build and push the container image, generate an SBOM and attach it.
3. Make the image reproducible: install from `uv.lock`, pin the base image by digest (TASK-0496).
4. After publishing, a job installs the published version in a clean container and runs the local-profile quickstart.
5. Rewrite [release.md](release.md) §3 to describe the workflow.

**Documents.** [release.md](release.md), [release-readiness.md](release-readiness.md), [CHANGELOG.md](CHANGELOG.md), `README.md` (the "not yet on PyPI" sentence is checked by `stamp_release.py --check` — stamp, do not hand-edit).

**Traps.** Depends on TASK-0087 and on TASK-0444 for the quickstart. The gate in [release.md](release.md) §1 is a confidence score of at least 85 for a beta; the last recorded score is 62. **Publishing is irreversible and outward-facing: it needs the maintainer's explicit instruction, not an agent's judgement.**

**Done when.** The release exists and the post-publish job passes. **Moves:** packaging.

### TASK-0470 — An empty document must not be "indexed"

**Goal.** A document that produces no chunks is recorded as skipped with a reason, and never enters the dedup file. **NOT VERIFIED — read from code.**

**Where.** `workers/ledger.py:104-108`: `expect(..., chunks=0)` finishes the document at once, which runs its commit callback — `record_indexed` plus the content hash. The hash of empty bytes then matches every later empty file, which is reported `deduplicated` against a document that was never in the index.

**Steps.** Reproduce with a zero-byte `.md` and a whitespace-only `.txt` through `IngestionService`. Then give the parse side a third outcome beside *indexed* and *dead-lettered*: `skipped` with reason `EMPTY_DOCUMENT`, settled through `ledger.settle(index)` with no commit. Decide whether a skipped document counts toward a job's failure total (recommended: no) and surface the count in the job record.

**Tests.** `tests/unit/services/test_ingestion.py`: an empty file is `skipped`, the dedup file is unchanged, a second empty file is also `skipped`. Update `tests/unit/workers/test_ledger.py` for the zero-chunk case.

**Documents.** [api-reference.md](api-reference.md) (job document statuses), [data-model.md](data-model.md), [troubleshooting.md](troubleshooting.md).

**Done when.** Both tests pass. **Moves:** ingestion, reliability.

### TASK-0473 — A parser that hangs must cost one document

**Goal.** A document whose parse exceeds a configured time is dead-lettered with `PARSE_TIMEOUT` and the job continues.

**Where.** `workers/cpu_pool.py:322` awaits the pool future with no timeout. A size limit exists (`ingestion.max_document_mb`, `schema.py:560`, enforced at `cpu_pool.py:130-151`), a time limit does not.

**Steps.** Add `ingestion.parse_timeout_seconds`. Wrap the await in `asyncio.wait_for`. A timed-out task in a `ProcessPoolExecutor` keeps running and keeps its worker: `Future.cancel()` does not stop it. Either recycle the pool on timeout (build a fresh executor, let the stuck process be terminated with the old one) or run each parse in a worker that enforces its own deadline. Recycling is simpler; make sure in-flight documents in the old pool are resubmitted, not lost.

**Tests.** A module-level slow parser (it must be importable by a spawned child) with a short timeout: the slow document is dead-lettered, the others index, and a later document still parses — proving the worker slot was recovered. Mark it `must_run`; it needs the real process pool (TASK-0455).

**Documents.** [config-reference.md](config-reference.md), [failure-modes.md](failure-modes.md) (new row), [troubleshooting.md](troubleshooting.md).

**Traps.** New config key and, if a new code is added, the `ErrorCode` row of §0.4.

**Done when.** The recovery test passes on Ubuntu and Windows. **Moves:** reliability, ingestion.

---

## 2. P1 — blocks a beta

Do TASK-0454 early: retrieval, performance and half of testing cannot move without it.

### TASK-0446 — Retrieved text is data, not instructions

**Goal.** A document cannot break out of its context block, forge a citation, or instruct the grader — and a committed adversarial eval set proves it.

**Where.** `core/generation.py`: `P1_TEMPLATE_VERSION = "1.0.0"` (`:39`), `P1_SYSTEM_PROMPT` (`:41-49`, no clause about context being data), `_MARKER` (`:51`), `build_context_block` (`:71-76`, raw chunk text and source), `build_prompt` (`:79-85`, `<context>\n{block}\n</context>\n\nQuestion: {q}`, nothing escaped), `resolve_citations` (`:88-114`, drops markers that name no supplied chunk — but a marker naming a *real* supplied chunk, injected from chunk text, passes). `core/faithfulness.py`: `P3_TEMPLATE_VERSION` (`:43`), `P3_SYSTEM_PROMPT` (`:45-54`), `_OBJECT = \{.*\}` greedy with DOTALL (`:57`), `build_grading_prompt` (`:87`, same raw block), `_extract_object` (`:97`), `parse_verdict` (`:114`).

**Steps.**
1. Neutralise, in `build_context_block`, the three things chunk text can forge: the closing delimiter, the citation-marker syntax, and a line beginning `Question:`. Prefer a per-request random delimiter over escaping.
2. Add the data-not-instructions clause to P1 and P3. Bump both template versions.
3. Replace the greedy pattern with a strict decoder that takes the last complete JSON object.
4. Commit an adversarial set under `tests/eval/datasets/` — instruction override, delimiter break-out, forged citation, grader manipulation — with a deterministic scripted-LLM unit tier (the prompt contains no unescaped delimiter) and a model-backed eval tier.

**Tests.** Wording asserts at `tests/unit/core/test_generation.py:26-29` and `tests/unit/core/test_faithfulness.py:47-53` will change. Add unit tests per attack on prompt construction and on verdict parsing (`test_a_verdict_wrapped_in_prose_is_parsed`, `:71`, must still pass).

**Documents.** [prompts.md](prompts.md) P1 and P3 (the code blocks duplicate the prompts verbatim), [security.md](security.md).

**Traps.** [prompts.md](prompts.md) `:14` says the template version is part of the trace `config_snapshot`; `core/tracing.py:39-56` does not include it. Add it there in this change, since a prompt change must be visible to replay. The `eval-gate` job measures retrieval only, so nothing committed gates a prompt change today (TASK-0481).

**Done when.** The adversarial set is committed and gated. **Moves:** security, core RAG.

### TASK-0447 — A sparse leg that works beyond English

**Goal.** Text in any script yields sparse terms, and the tokenizer is covered by property tests.

**Where.** `core/retrieval/bm25.py`: `K1` (`:52`), `B` (`:53`), `AVERAGE_DOCUMENT_LENGTH = 256` (`:54`), `_TERM_MASK = 0xFFFFFFFF` (`:57`), `_TOKEN = [a-z0-9]+(?:[._-][a-z0-9]+)*` applied to lower-cased text (`:59` — the task entry quotes a simpler pattern), English stop words (`:61-124`) and stemmer (`:126`), `term_index` using CRC32 (`:130-132`), `tokenize` (`:135`), `encode_document` (`:149`), `encode_query` (`:172`). When `encode_query` returns nothing, `services/querying.py:268-271` skips the sparse leg with an info log.

**Steps.** (1) Unicode tokenisation: NFKC-normalise, case-fold, match `\w+`; segment CJK by bigram. (2) Make stemming and stop-listing per language, chosen by `retrieval.bm25_language` (default `english`, plus `none`). (3) Keep 32-bit term indices: Qdrant and Milvus sparse indices are 32-bit unsigned, and existing tests assert the bound (`tests/unit/core/retrieval/test_bm25.py:57`, `test_bm25_properties.py:320`). Measure the collision rate on a real vocabulary before proposing 64 bits, and record what was found. (4) Either store a real average document length per collection or keep the constant and record what it costs on the TASK-0454 set. (5) Add a tokenizer version to `retrieval_config_hash` (`core/identity.py:177`): tokenizer code is not part of it today, so a tokenizer change would not show as lockfile drift.

**Tests.** Property tests over `WILD_TEXT` (`test_bm25_properties.py:94`) extended to non-Latin scripts: non-empty input with at least one letter yields at least one term. `tests/unit/services/test_querying.py:170` `test_a_query_with_no_indexable_terms_skips_the_keyword_leg` stays, with a different example.

**Documents.** `docs/adr/ADR-0007`, [architecture.md](architecture.md) §6, [config-reference.md](config-reference.md).

**Traps.** Every change here moves sparse vectors: it is a **reindex** for all five adapters. New `retrieval.*` key: §0.4.

**Done when.** A multilingual eval set shows a non-empty sparse leg per script. **Moves:** retrieval quality.

### TASK-0448 — Relevance floor, diversity, query transformation

**Goal.** Three independent, flagged, measured capabilities. Each ships off by default and changes default only with a ledger entry.

**Where.** `services/querying.py:220` slices the top `limit` whatever the scores. With zero chunks the LLM is still called (`services/generation.py:628`), and `tests/unit/services/test_generation.py:318` `test_an_empty_corpus_still_produces_a_well_formed_stream` pins that. The only diversity mechanism is the Jaccard near-duplicate drop in `core/context.py:184` (threshold 0.9). [prompts.md](prompts.md) states there are "exactly four" LLM call sites.

**Steps.** Three commits. (1) `retrieval.min_score`: applied to the reranker score when reranking and to the fused score otherwise; when nothing clears it, answer "no evidence" without calling the LLM. (2) `retrieval.mmr_lambda`: MMR over the reranked shortlist using the dense vectors already retrieved. (3) `retrieval.query_transform: none | rewrite | multi_query | hyde`, as a fifth prompt contract (P5) with its own template version, budget charge, breaker and trace span; fuse multi-query results with the existing `rrf_fuse`.

**Tests.** Unit tests per mode with scripted doubles; the empty-corpus test at `:318` gains a twin for `min_score`. Extend `services/autopilot.py`'s candidate grid (`:106-108`, `:351-358`).

**Documents.** [config-reference.md](config-reference.md), [prompts.md](prompts.md) (five call sites), [architecture.md](architecture.md) §6.

**Traps.** Three new `retrieval.*` keys: the comparability and baseline rows of §0.4, three times. Scores are not comparable across adapters until TASK-0451 fixes the convention — document `min_score` as applying to the reranker score, which is adapter-independent.

**Done when.** Each is measured on the TASK-0454 set and recorded, whichever way the number goes. **Moves:** retrieval quality, advanced RAG.

### TASK-0449 — Retry, fallback and a deadline for generation

**Goal.** A 429 is retried; a provider outage falls back to a second provider before the extractive rung; a hung reranker degrades; no request outlives its deadline.

**Where.** `max_retries=0` in `adapters/llm/openai.py:44`, `adapters/llm/anthropic.py:43` and `adapters/embeddings/openai.py:69`; the Cohere and Ollama LLM adapters leave the SDK default, so retry behaviour differs by provider. `GenerationService` never retries. `classify_llm_failure` (`adapters/llm/base.py:79`) reads only `status_code` — no `Retry-After`. The retry policy to reuse is `backoff_delay` (`workers/embed_pool.py:104`). `TimeoutSettings` (`schema.py:587-592`) has `vector_db_ms`, `embeddings_ms`, `llm_ms` and nothing for the reranker; `CrossEncoderReranker` runs through `to_thread` with no timeout (`core/rerank/cross_encoder.py:162`).

**Steps.** (1) Set `max_retries=0` on all four LLM clients so the policy lives in one place. (2) Carry `retry_after` on the typed error. (3) Wrap the non-streamed call, and the streamed call up to its first token, in the shared retry policy. (4) `llm.fallback`: a second provider block with its own breaker key (the shared breaker is keyed by the provider string). (5) `reliability.timeouts.rerank_ms`, degrading to `hybrid_only` (`services/querying.py:199-218` already degrades on `FasterRagError`). (6) `reliability.request_deadline_ms` with `asyncio.timeout` around the pipeline in the service, so every surface gets it.

**Tests.** Chaos scenarios — a 429 storm, an outage with fallback, a hung reranker — added to the hard-coded list. Existing: `tests/chaos/test_chaos.py:308`, `:327`, `tests/chaos/test_reranker_down.py`, `tests/unit/services/test_generation.py:818`.

**Documents.** [reliability.md](reliability.md) §2, [failure-modes.md](failure-modes.md), [config-reference.md](config-reference.md).

**Traps.** `to_thread` cannot be cancelled: a timed-out rerank keeps its thread until the model returns. Bound the damage with a semaphore on concurrent reranks.

**Done when.** The three chaos scenarios pass. **Moves:** reliability, core RAG.

### TASK-0450 — One probe in half-open

**Goal.** With N callers waiting on an open circuit, exactly one is admitted when the timeout expires.

**Where.** `core/breaker.py`: `CircuitState` (`:47`), `CircuitBreaker` (`:71`), `state` (lazy OPEN→HALF_OPEN, `:87`), `check` (`:127-138`, raises only when OPEN), `record_success` (`:140`, closes unconditionally — a late success from a call admitted before the circuit opened closes it), `record_failure` (`:145`), `shared_breaker` (`:179`). The module docstring (`:13-14`) promises one probe. Callers use `check()`, `record_success()`, `record_failure()` with no token: `services/generation.py:627`, `:665`, `:784`, `:822`; `services/querying.py:237`, `:241`, `:244`; and the embedding pool.

**Steps.** Make `check()` return an admission token carrying the state generation it was issued in; in HALF_OPEN issue one token and refuse the rest until it resolves; make `record_success(token)` and `record_failure(token)` ignore a token from an earlier generation. Update every call site in one commit.

**Tests.** A concurrency test with N waiters observing one admission; a late-success test. Existing tests that encode today's semantics: `tests/unit/core/test_breaker.py:162`, `:171`, `:371`.

**Documents.** [reliability.md](reliability.md) §2.

**Done when.** The concurrency test passes. **Moves:** reliability.

### TASK-0451 — One scaffold, one score convention, one error code for vector adapters

**Goal.** New contract cases for Euclidean and dot scoring, list-valued filters and `$ne` on a missing field pass on all five adapters.

**Where.** Duplicated helpers: `_slug`/`_digest` in `pgvector.py:204/210`, `chroma.py:175/181`, `milvus.py:214/220`, `weaviate.py:243/249`; `collection_name_for` in `chroma.py:186`, `milvus.py:225`, `weaviate.py:254`; `_Collection` in `pgvector.py:238`, `chroma.py:248`, `milvus.py:377`, `weaviate.py:503`; `_distance_of` in `pgvector.py:225`, `chroma.py:196`, `milvus.py:306`, `weaviate.py:375`; `_idf` in `milvus.py:294`, `weaviate.py:359`; `_rank` in `chroma.py:1092`, `weaviate.py:1377`. Drift: Euclidean score is positive distance on pgvector (`:165-172`) and negated on Chroma (`:208-212`), Milvus (`:299-303`) and Weaviate (`:364-372`, squared, `:392`); vector-database failures are `EMBED_PROVIDER_ERROR` in four adapters and `RETRIEVAL_FAILED` in Qdrant (`qdrant.py:225`, `:250`, `:256`). The list-valued-filter difference is **NOT VERIFIED**.

**Steps.** (1) Write the missing contract cases first and record which adapters fail which — that is the evidence. (2) State one convention in `VectorDBAdapter`'s docstring: a higher score is closer, for every distance. (3) Introduce one error code for vector-database failures (a new `ErrorCode`: §0.4). (4) Extract the shared scaffolding into `adapters/vectordb/_common.py`, one helper per commit.

**Tests.** `tests/contract/vectordb.py` (about 38 cases today; `test_closer_scores_higher` at `:194` is the model). `DIMENSIONS = 4` at `:38`.

**Documents.** [architecture.md](architecture.md) §10, [testing-strategy.md](testing-strategy.md) §1.5, [api-reference.md](api-reference.md) error table.

**Traps.** An adapter-contract change needs the maintainer review TASK-0182 asks for — get it before merging. Milvus and Weaviate skip in CI (TASK-0412): run `docker compose -f deploy/compose.test-backends.yml up -d` and the two suites locally, and say in the commit that you did.

**Done when.** The new cases pass on all five. **Moves:** embedding and vector infrastructure, architecture.

### TASK-0452 — pgvector: filtered recall and high dimensions

**Goal.** `limit=100` under a selective filter returns 100 rows; a 3,072-dimension collection can be created. **NOT VERIFIED — run the experiment first.**

**Where.** `adapters/vectordb/pgvector.py`: the column is `embedding vector({dimensions})` (`:643`), the index `USING hnsw` (`:648`), the search `ORDER BY embedding <op> %s LIMIT %s` with a `WHERE` (`:1153-1172`). `SET LOCAL` is used only for `statement_timeout` (`:388`). No `ef_search`, `iterative_scan` or `halfvec` anywhere.

**Steps.** Write the contract case first (seed several hundred points; filter to a subset larger than 100; ask for 100) and see whether it fails. If it does: `SET LOCAL hnsw.ef_search` from the limit, enable iterative scan where the server version supports it, and use `halfvec` above 2,000 dimensions.

**Traps.** The pgvector suite *does* run in CI's integration job, so this one is fully provable there.

**Done when.** The contract case passes on pgvector. **Moves:** embedding and vector infrastructure.

### TASK-0453 — Redis timeouts and pipelining

**Goal.** A hung Redis degrades within a configured time.

**Where.** `core/cache/redis.py:104` `Redis.from_url(url, decode_responses=False)`; `core/rate_limit.py:194` `Redis.from_url(url)`. A cache `get` hit is `GET` + `INCR` + `ZADD` — three round trips (`redis.py:124-148`).

**Steps.** Add `reliability.timeouts.redis_ms`; pass `socket_timeout` and `socket_connect_timeout`; pipeline the get and set sequences.

**Tests.** A local socket that accepts and never answers; assert the cache misses and the limiter fails open within the timeout. Existing: `tests/unit/core/test_rate_limit.py:205` `test_a_backend_outage_serves_the_request_and_says_so`.

**Done when.** The non-responding-socket test passes. **Moves:** reliability.

### TASK-0454 — Build the measuring instrument

**Goal.** Citable numbers for throughput, latency and retrieval quality, with the harness committed — or the goal sentence stays a goal because the number said so.

**Where.** Targets: [performance.md](performance.md) §2 (`:20-29`), all "TBD-until-measured". Ledger: [benchmarks.md](benchmarks.md) BENCH-0001 to BENCH-0006, every one marked not citable. Harness code: `services/benchmark.py` (cold/warm split `:198-263`), `fasterrag benchmark --suite ingest|query|eval`, `tests/load/` (`harness.py`, `run_load.py`, `locustfile.py`, `preflight.py`), `tests/soak/`. Eval: `core/evals/metrics.py` has `recall_at_k`, `reciprocal_rank`, `ndcg_at_k` and no precision; `core/evals/harness.py:121-122` counts a hit by chunk id **or** document id; `services/evaluation.py:120` builds `RetrievalService(settings, adapter, router)` with no reranker. Datasets: `tests/eval/datasets/handbook` (6 documents, baseline saturated at 1.0) and `policies` (12), both with empty `relevant_chunk_ids`. The CI `eval-gate` job runs with `rerank=false`, `top_k=5`.

**Steps.**
1. **Quality first; it needs no special hardware.** Add a BEIR subset (start with two small ones) as a downloadable dataset with a pinned checksum, not committed bytes. Fix the harness so passage-level truth is scored at passage level. Pass the reranker into `score_collection`. Record with-reranker and without.
2. **Throughput and latency.** A corpus generator with a fixed seed; runs at 100k and 1M chunks; hardware, OS, versions and the `working tree:` line recorded per ledger rule 7. TASK-0084 is blocked on isolated hardware — if none is available, record the run as non-citable and say why.
3. **A baseline.** The same corpus through one named, widely used framework, with its harness committed under `benchmarks/`. A speed claim is only permitted relative to a baseline measured here.
4. Replace or keep the README's goal sentence according to what was measured.

**Documents.** [benchmarks.md](benchmarks.md), [performance.md](performance.md) §2, §4, §5, [slo.md](slo.md), `README.md`.

**Traps.** Changing how hits are counted changes every recorded eval number: re-record `baseline.json` in the same commit and say so. Nightly numbers are non-citable by design.

**Done when.** At least one citable quality entry and one throughput entry exist, or the reason they cannot is recorded. **Moves:** performance, retrieval, testing — the largest single lever.

### TASK-0455 — Run the real process pool in a test

**Goal.** Three documents ingest through the production executor, and a killed worker yields the typed error, in both CI lanes.

**Where.** The only construction is `workers/cpu_pool.py:183-185`: `executor_factory or (lambda workers: ProcessPoolExecutor(max_workers=workers))`. The task crossing the boundary is `parse_and_chunk(task: DocumentTask, settings: Settings) -> ParseOutcome` (`:74`); `BrokenProcessPool` maps to `_unguarded_entry_point_error()` (`:154`, `:323-324`). Every test injects threads: `tests/unit/workers/test_cpu_pool.py:38`, `tests/unit/services/test_ingestion.py:38`, `tests/integration/test_ingest_pipeline.py:98`.

**Steps.** Add `tests/unit/workers/test_real_process_pool.py`, marked `must_run`: (1) ingest three small Markdown files with no `executor_factory`; (2) a module-level parse target that calls `os._exit(1)`, asserting the typed error and that the job settles. Use the fixed chunking strategy with the character-based token counter so children load no model.

**Traps.** Pickling of `Settings` (a pydantic settings object) and of the frozen slots dataclasses in `workers/queues.py` (`DocumentTask` `:38`, `ChunkPayload` `:55`, `ParseOutcome` `:81`) has **never been exercised** — this test may find a real defect; if it does, that is the point. Under `spawn` the target must be importable: no closures, no `monkeypatch`. Children re-import `fasterrag`; keep the test's configuration minimal so that is fast.

**Done when.** Both tests run (not skip) on Ubuntu and Windows. **Moves:** testing, reliability.

### TASK-0456 — One dependency direction, enforced

**Goal.** `api`, `cli` → `services` → `workers` → `core` ← `adapters`, checked by a gate.

**Where.** `services → workers`: `services/ingestion.py:51-55`, `estimation.py:32`, `golden_sets.py:26`. `workers → services`: `cpu_pool.py:42` (journal), `embed_pool.py:41` (budget) and `:359` (function-level), `indexer.py:31` (tenancy). `core → adapters`: `faithfulness.py:27`, `retrieval/bm25.py:38`, `evals/generator.py:31`, `cache/embedding.py:26-27`, `chunking/enrichment.py:25`. `adapters → core`: `embeddings/huggingface.py:21-22`. `services → api`: `evaluation.py:175`. `facade → api`: `facade.py:181`, `:191`, `:214`, `:328`. `config → observability`: `config/loader.py:49`. No import-linter configuration exists.

**Steps.** (1) Add the contract to CI first, with today's violations listed as explicit, named exceptions — the gate then stops new ones immediately. (2) Move the service builders out of `api/dependencies.py` into `services/composition.py`; point the API, the CLI and the facade at it. This alone removes the `facade → api` and `services → api` edges and unblocks TASK-0444. (3) Move the protocols `workers` needs (`Journal`, `CostGovernor`, tenancy helpers) into `core/` as interfaces. (4) Decide the `core ↔ adapters` rule: `core` may depend on adapter *base interfaces* only; move those interfaces or invert the five imports. (5) Delete each exception as its edge goes.

**Traps.** Function-level imports hide cycles from a module-level scan — configure the tool to see them. `scripts/check_unbuilt_claims.py` reads [structure.md](structure.md): moving a module means updating it.

**Done when.** The contract has no exceptions. **Moves:** architecture.

### TASK-0457 — Product code only in the wheel

**Goal.** A core install imports neither the CI-status reader nor the provisioning code.

**Where.** `services/ci_status.py` (533 lines). Importers at module top: `cli/parser.py:23-24`, `cli/commands/diagnostics.py:27`, `:40`, `cli/commands/infrastructure.py:23`, `api/admin.py:32-33`, `services/doctor.py:45`. Provisioning: `services/provisioning.py` (Docker through `create_subprocess_exec`, `:116`), `grafana.py`, `langfuse.py`, `provision_registry.py`.

**Traps — read before starting.** The `branch-verdict` job (`ci.yml:650`) runs `fasterrag ci-status`; moving the module to `scripts/` means changing that job in the same commit. `services/provisioning.py` also backs the default `vector_db.mode: docker` and `doctor`, and the integration conftests import `docker_available` and `run_docker` from it — it cannot simply move behind an extra. `scripts/check_package.py` asserts absence only for `torch`, `transformers` and `sentence_transformers` (`:30`, `:74-84`).

**Steps.** Move `ci_status` to `scripts/` and keep `fasterrag ci-status` only if the maintainer wants it as a product feature. Make provisioning imports lazy at their call sites; do not gate the Docker helpers behind an extra. Extend `check_package.py` with a module-absence assertion.

**Done when.** `check_package.py` asserts it. **Moves:** architecture, packaging.

### TASK-0458 — Conversation, templates, structured output, hosted rerankers

**Goal.** Five separate changes, each with a test and a matching [prompts.md](prompts.md) entry.

**Where.** `LLMAdapter` (`adapters/llm/base.py:123`): `complete(prompt, *, system=None)` (`:145`), `stream(prompt, *, system=None)` (`:149`) — no message list. Facade `query` (`facade.py:236`), `query_stream` (`:250`), sync twins (`sync.py:135`, `:148`). `P1_SYSTEM_PROMPT` is `Final` (`core/generation.py:41`). The OpenAI adapter always sends `temperature` and `max_tokens` (`openai.py:58-59`), asks for `include_usage` (`:63`) and skips events with empty `choices` (`:99-100`), which is exactly where the usage chunk arrives.

**Steps.** In this order: (1) read the streamed usage chunk — small, and it fixes estimated token counts in TASK-0438; (2) omit `temperature`/`max_tokens` when unset, and support the newer token-limit parameter name; (3) `generation.prompt_template` — a validated path, hashed into the trace snapshot; (4) `messages` on `query`/`query_stream` across the four adapters and both facades, with a token-budgeted history; (5) a JSON-schema response mode; (6) hosted rerankers through the registry from TASK-0439.

**Traps.** `generation.*` is not part of `retrieval_config_hash`, so a template change is invisible to the lockfile — it must be visible to the trace. The facade parity test compares signatures. The comment at `services/generation.py:845` asserts "a stream never reports usage"; it becomes false.

**Moves:** core RAG, advanced RAG (conversation is a prerequisite for TASK-0482).

### TASK-0459 — Parser correctness

**Goal.** One failing-then-passing test per named input, in `tests/unit/core/parsing/`.

**Where.** `services/sources.py:89` `_extension_for(uri)` uses only the URL path suffix and falls back to `.txt`; it is applied to inline payload text too (`:384`). `_decode_inline` (`:105`) tries strict base64 first. `core/parsing/docx.py:51-86` walks only top-level paragraphs and tables of the body — no headers, footers, footnotes or nested tables (`_table_text`, `:41`) — and has no decompression guard (`Document(io.BytesIO(data))`, `:54`). `core/identity.py:163` `chunker_config_hash` covers strategy, size and overlap only. `core/parsing/html.py:40` uses BeautifulSoup with lxml; the double-emit and dropped-`div`-text claims are **NOT VERIFIED** — reproduce them.

**Steps.** One fix and one regression test per item: HTML nesting; `Content-Type`-driven parser choice for URLs; explicit `media_type` and `filename` on inline sources (a request-model change: §0.4); DOCX headers, footers, footnotes, nested tables and a size cap on the unzipped archive; text normalisation; `token_counter` and parsing options added to the chunk-id fingerprint.

**Traps.** Golden snapshots under `tests/unit/core/parsing/golden/` will change — review each diff, do not regenerate blindly. Widening the hash changes every chunk id: a reindex, and `tests/chaos/test_config_mutation.py` checks the hashed subset.

**Moves:** ingestion.

### TASK-0460 — Control-plane hardening

**Goal.** One change per finding, each recorded in [security.md](security.md) §5.7 (`:303-312`).

**Where and what.**
- **A key with no tenant acts for any tenant.** `api/auth.py:351` `_tenant_refusal` refuses only when `key.tenant is not None`; `:327` sets `tenant = declared or key.tenant`. Under `multi_tenancy`, refuse an unbound key unless it carries an explicit cross-tenant scope.
- **`multi_tenancy: true` with `auth: false`** passes validation (`schema.py:774-783` validates only `tenant_scoped_retrieval`). Make it a load-time error.
- **Placeholder keys.** `load_keys` (`auth.py:146-179`) has no placeholder or length check; `.env.example` ships `change-me`. Refuse it, and a key under a minimum length, at startup.
- **`/readyz`** (`api/health.py:66-87`) returns adapter detail unauthenticated (`PUBLIC_PATHS`, `auth.py:70`). Return status only when unauthenticated. The container `HEALTHCHECK` calls `/readyz` without a key — keep the status code meaningful.
- **Problem `instance`** echoes the request path (`api/problems.py:148`, `:169`, `:186`, `:203`; `auth.py:347`). Decide whether that is a leak for path parameters and record the decision.
- **Dashboard bind host.** `processes.py:85-95` reuses the API host; there is no `dashboard_host`. Add one, defaulting to loopback.
- **Actions pinned by tag.** Pin to commit SHAs. Add CodeQL and an image scan.
- **Unauthenticated non-loopback bind.** Today a warning (`processes.py:32`, `:55-60`). Consider a refusal overridden by `security.allow_unauthenticated: true`.

**Tests.** `tests/unit/api/test_auth.py` (47 tests), `test_health.py`, `test_problems.py`, `tests/unit/cli/test_processes_commands.py`.

**Traps.** Tenant scoping of ingest and query is TASK-0339 and TASK-0381, not this task. Each new key is a §0.4 row.

**Moves:** security.

### TASK-0461 — The facade as a tenant; honest error codes

**Goal.** `tenant=` on `query`, `query_stream` and `retrieve` of both facades; caller errors carry a caller-error code.

**Where.** `facade.py`: `_retrieval()` (`:179`) calls `build_retrieval(settings, vector_db, embeddings)` with no tenant; `_generation_service()` (`:185`) is built once and cached, also with none. The builders accept it: `build_retrieval(..., tenant=None)` (`api/dependencies.py:115`), `build_generation(..., tenant=None)` (`:130`). `RetrievalService._tenant_scoped` (`services/querying.py:165-192`) raises `TENANT_FORBIDDEN` for a missing tenant when scoping is on. `CONFIG_INVALID` is raised for misuse at `facade.py:148`, `:307` and `sync.py:79`, `:109`. The CLI emits `"VECTOR_DB_UNAVAILABLE"` (`cli/commands/processes.py:125`), which is not in `ErrorCode` (`errors.py:49-80`, 29 members).

**Steps.** Thread `tenant` through; because `GenerationService` takes the tenant in its constructor, either pass it per call or stop caching the service. Add `USAGE_ERROR`. Add the CLI's code to the enum or replace it.

**Traps.** Three §0.4 rows: facade parameter, `ErrorCode`, documents.

**Done when.** `tests/unit/docs/test_python_api_parity.py` passes with the new signatures. **Moves:** developer experience, security.

### TASK-0466 — Build long-lived things once

**Where.** Per call: `build_retrieval` (`api/dependencies.py:115`), `build_generation` (`:130`, which calls `create_llm_adapter` at `:142` — a new provider client per query, closed at `api/query.py:88` and `:122`), `build_ingestion` (`:150`, a new lock store at `:169`). Already process-scoped on `app.state`: the vector-database adapter, the journal, the embedding router and the semantic cache.

**Steps.** After TASK-0456's `services/composition.py` exists, build the LLM adapter and the stateless services in the lifespan, store them on `app.state`, and close them at shutdown. Keep the tenant a per-call argument, not constructor state.

**Tests.** Two queries through the test client observe one LLM adapter instance; shutdown closes it once.

**Moves:** architecture, performance.

### TASK-0467 — A disconnected client stops paying for tokens

**Where.** `api/query.py:57-88`: the streaming response relies on the server cancelling the generator; the `finally` closes the service. Nothing checks `request.is_disconnected()`, and no test covers a disconnect.

**Steps.** Write the test first: a scripted LLM that yields slowly; close the client mid-stream; assert the LLM stream was closed and a partial trace was stored with `aborted: true`. Fix what it finds.

**Moves:** core RAG, reliability.

### TASK-0468 — Defined behaviour at the query edge

**Where.** A query against a missing collection has no service or API test, and the code raised differs by adapter. `QueryRequest` (`api/schemas.py:73-75`) enforces `min_length=1`, so a whitespace-only query passes. The facades and services validate nothing.

**Steps.** One code — `NOT_FOUND` with the collection name — for a missing collection on every adapter (a contract case: `tests/contract/vectordb.py:348` accepts any `FasterRagError` today). Strip and reject blank queries in the service, so REST, CLI and the facade agree.

**Moves:** core RAG, developer experience.

### TASK-0469 — List the documents in a collection

**Where.** The only listing is per job: `GET /v1/ingest/{job_id}/documents` (`api/ingest.py:190`). `Journal.document_hashes` (`journal.py:528`) already maps a collection's documents.

**Steps.** `GET /v1/collections/{name}/documents` (paginated), `fasterrag documents list`, `FasterRag.documents()`. Build it with TASK-0443 so tombstones are honoured from the start.

**Moves:** core RAG.

### TASK-0471 — Deduplication under concurrency and with the switch off

**NOT VERIFIED.** `workers/cpu_pool.py:256` and `:281` update the in-job `known` map in submission order — and `:281` does so even when `ingestion.dedup: false`, so the switch is ignored within a job. `known` is loaded once per `process()` (`:235`), so two concurrent jobs carrying the same content under different names both miss and both index.

**Steps.** Reproduce each with a unit test. Honour the switch. For the cross-job race, check the dedup record again at commit time under the journal's lock (TASK-0437), and treat a late duplicate as `deduplicated`, deleting the chunks just written.

**Moves:** ingestion, reliability.

### TASK-0472 — Inputs the pipeline has never been shown

**Where.** `core/parsing/plaintext.py:20-27` `decode`: UTF-8 with BOM handling, then `errors="replace"` and an `ENCODING_FALLBACK` flag — Latin-1 and UTF-16 text becomes replacement characters and is indexed. An encrypted PDF falls into a broad catch (`pdf.py:171-172`) with no specific reason code and no test. [failure-modes.md](failure-modes.md) row 2 records "no scanned-document fixture".

**Steps.** Detect UTF-16 by BOM and try a small ordered list of encodings before falling back; dead-letter, not index, a document that is mostly replacement characters. Give encrypted PDFs their own reason code. Generate a scanned-page PDF fixture with `reportlab` (already a dev dependency) for the OCR path.

**Moves:** ingestion, testing.

### TASK-0474 — Dead letters: idempotent retry, bounded growth

**Where.** `POST /v1/ingest/{job_id}/retry-dlq` (`api/ingest.py:217-247`) accepts a new job with no idempotency key, so a repeated call starts repeated jobs. `dlq.max_retries` (`schema.py:551`) is only recorded as `attempts` (`cpu_pool.py:370`). Neither `documents.jsonl` nor the dedup file is ever pruned.

**Steps.** Derive the retry job's idempotency key from the source job and its dead-letter set. Enforce `max_retries`. Add `ingestion.journal.retention_days` with a prune pass, alongside the existing trace retention (`services/traces.py:216-220`).

**Moves:** reliability.

### TASK-0475 — Journal appends inside the error taxonomy

**Where.** `journal.py:379-385` and `:503-509` append with no `try`; a full disk raises a raw `OSError`. `tests/chaos/test_chaos.py:341` accepts `(IngestionError, OSError)` — it tolerates the defect.

**Steps.** Wrap both appends like `_write_atomically` does (`:153-165`); tighten the chaos assertion to `IngestionError` only. Do it with or before TASK-0437.

**Moves:** reliability.

### TASK-0476 — A timeout on embedded Chroma

**Where.** `reliability.timeouts.vector_db_ms` (`schema.py:590`) is applied by Qdrant, pgvector, Milvus and Weaviate. Chroma runs a `PersistentClient` (`chroma.py:340`) through `asyncio.to_thread` with no `wait_for` (`:324-327`).

**Steps.** Wrap in `asyncio.wait_for`, raising the typed timeout. The thread cannot be cancelled; bound concurrency so timed-out calls cannot exhaust the thread pool. Add the case to the Chroma unit tests.

**Moves:** reliability.

### TASK-0477 — Detect a query against the wrong index

**Where.** `services/lockfile.py`: `detect_drift` (`:200`), used by `index list` and `index lock verify` only. `services/querying.py` never consults the lock. A changed embedding model with the same dimension returns confident, wrong results.

**Steps.** At service construction (not per query), compare the collection's lock with the live configuration; on embedding-model or chunker drift refuse with a typed error naming `fasterrag index reembed`, behind `retrieval.verify_lock` (default on for new installs after one release as a warning).

**Moves:** embedding infrastructure, reliability.

### TASK-0478 — Embedding batches that adapt

**Where.** Batching is a fixed slice (`adapters/embeddings/base.py:189-197`); retries use `backoff_delay` (`workers/embed_pool.py:104`, `:251-284`). No truncation of an over-long input, no split on a payload-too-large or token-limit error, no `Retry-After`. The only guard is a warning when `chunk_size > 1024` (`config/loader.py:274-283`).

**Steps.** On a size error, halve the batch and retry; an input that fails alone is one failed chunk (reported through `ledger.failed`), not a failed batch. Honour `Retry-After` (shared with TASK-0449).

**Moves:** embedding infrastructure, reliability.

### TASK-0481 — Evaluate the answer, not only the retrieval

**Where.** `services/evaluation.py:166` `score_faithfulness` has no production caller. Metrics are recall, reciprocal rank and nDCG; there is no precision and no answer-relevance measure. The `eval-gate` job gates retrieval only, so a prompt change is gated by nothing.

**Steps.** Add `precision_at_k`. Wire `score_faithfulness` into `fasterrag benchmark --suite eval` behind a flag (it costs model calls). Record a generation baseline beside the retrieval one; gate it in the nightly, not on every push.

**Moves:** testing, retrieval, core RAG. Prerequisite for TASK-0446's "gated".

### TASK-0486 — A plugin author's kit

**Where.** No sample plugin, no author guide beyond [python-api.md](python-api.md) "Extending" and [internals.md](internals.md) §5, no adapter-interface version, and `fasterrag.__all__` (`__init__.py:17`) is membership-tested but not snapshotted.

**Steps.** After TASK-0439: `examples/plugin-vectordb/` — a tiny installable package registering an in-memory store — installed and exercised in CI against the contract suite; an `ADAPTER_API_VERSION` constant per base interface, checked at registration with a clear message; a snapshot test of the public surface; a "Writing a plugin" guide.

**Moves:** extensibility, documentation.

### TASK-0487 — Timings that localise a slow request

**Where.** `services/generation.py:403-439` records spans for retrieval, reranker (started at the *retrieval* start, so not a rerank duration), context assembly, and generation (non-streamed only). No embed, dense, sparse or fuse breakdown; `timings_ms` holds retrieve, assemble, generate, grade, cache. `observability/otel_export.py` exports four query spans and no ingestion spans.

**Steps.** Time each leg in `RetrievalService._run_legs` and return it with the result; fix the reranker span's start; add parse, chunk, embed and upsert spans to ingestion, labelled on the existing `stage_duration_seconds` metric.

**Moves:** performance, developer experience.

### TASK-0488 — Measure on a schedule

**Where.** `nightly.yml` `measure` runs `pytest -m eval`, then `fasterrag benchmark --suite ingest` (parse and chunk over the handbook corpus), then `scripts/compare_nightly.py --tolerance 1.25`. The query suite is unmeasured. `tests/load/` and `tests/soak/` run in no workflow. Related open tasks: TASK-0405, TASK-0406, TASK-0243, TASK-0084.

**Steps.** Add the query suite against a service-container Qdrant to the nightly; add a weekly short soak (`FASTERRAG_SOAK_SECONDS`) and a load smoke; add an import-time budget test for `import fasterrag` (the existing guard is `tests/unit/test_facade.py:57`).

**Moves:** performance, testing.

### TASK-0489 — Property and fuzz coverage where it is thin

**Where.** `tests/unit/core/chunking/test_invariants.py:195-202` excludes the semantic chunker from the token-budget property, and its `TEXT` alphabet is ASCII (`:61-65`). `tests/unit/core/parsing/test_invariants.py` fuzzes plaintext, Markdown, CSV and HTML only (`TEXT_PARSERS`, `:89`). No corrupt-DOCX, zip-bomb or deeply nested HTML test. `core/parsing/tabular.py:110` `_walk` recurses without bound.

**Steps.** Widen alphabets to full Unicode; add truncated and bit-flipped DOCX and PDF cases; add depth-bounded JSON and HTML cases and make `_walk` iterative or depth-limited with a typed error.

**Moves:** testing, ingestion, security.

### TASK-0490 — Test what is claimed to be supported

**Where.** Every CI job runs Python 3.12; the classifiers (`pyproject.toml:21-31`) claim 3.13; the unit tier runs on Ubuntu and Windows, nothing on macOS. No per-test timeout, so a hung test hangs a lane. Random ordering is an open decision in TASK-0354.

**Steps.** Add a 3.13 unit lane, or remove the classifier. Add `pytest-timeout` with a generous default. Add a macOS unit lane or state in [deployment.md](deployment.md) that it is untested.

**Moves:** testing, packaging.

### TASK-0491 — Prove a key never reaches a log

**Where.** Auth logs only the key's name (`api/auth.py:231`, `:343`) and `config show` redacts (`cli/commands/diagnostics.py:140`), but no test asserts the bearer value is absent from logs, traces and problem bodies, and the logging module has no redaction filter.

**Steps.** A test that sends a distinctive key through success, 401, 403, 422 and 500 paths with `caplog` and a trace store, asserting the value appears nowhere. Add a logging filter that masks anything matching a loaded key as defence in depth.

**Moves:** security.

### TASK-0492 — API keys: hashed, rotatable, expirable

**Where.** `load_keys` (`api/auth.py:146`) reads `secret[:scopes[:tenant]]` entries from one environment variable; `ApiKey` holds the plaintext (`:89-104`); a restart is the only way to change keys.

**Steps.** Accept a hashed form (`sha256:<hex>`) and compare digests; reload keys on `SIGHUP` or a file-watch of a key file; optional `expires=` per key. Keep the plaintext form working.

**Moves:** security.

### TASK-0493 — An audit trail for administrative actions

**Where.** Nothing in `src/` records who dropped a collection, ran a restore, provisioned a tool or imported an archive.

**Steps.** A structured audit logger (`fasterrag.audit`) emitting key name, tenant, action, target and outcome for every route requiring the `admin` or `collections` scope and the destructive CLI commands; documented in [security.md](security.md) and [observability.md](observability.md).

**Moves:** security.

### TASK-0494 — CORS and response headers

**Where.** No CORS middleware and no security headers in `src/`; `vercel.json` sets them for the documentation site only.

**Steps.** `security.cors_allowed_origins` (default empty: no cross-origin access), and `X-Content-Type-Options`, `Referrer-Policy` and `Cache-Control: no-store` on API responses.

**Moves:** security.

### TASK-0496 — Supply-chain claims with nothing behind them

**Where.** [security.md](security.md) `:339` says an SBOM is generated at each tagged release — there is no SBOM tooling and no release. `:340` says images are pinned by tag or digest — the `Dockerfile` base is a tag, and it installs from `pyproject.toml`, not `uv.lock`. `:341` says dependency-update pull requests run the full pyramid — nothing produces such pull requests. No licence check exists.

**Steps.** Status notes were added to those three lines on 2026-11-12. To remove them: a Dependabot or Renovate configuration; the base image pinned by digest; the image built from `uv.lock`; a licence allow-list check in the `supply-chain` job; the SBOM in TASK-0445's release workflow.

**Moves:** security, packaging.

### TASK-0497 — The blocking facade in a notebook

**Where.** `sync.py:68-80` refuses to start inside a running event loop (`CONFIG_INVALID`), and its module docstring names "a notebook" as a use. Jupyter always has a running loop. Test: `tests/unit/test_sync_facade.py:72`.

**Steps.** When a loop is running, run the facade's own loop on a dedicated daemon thread and marshal calls with `run_coroutine_threadsafe`. Keep the refusal only for a call made from that thread itself.

**Moves:** developer experience.

### TASK-0498 — Runnable examples

**Where.** There is no `examples/` directory.

**Steps.** `examples/` with a handful of small scripts — ingest a folder and ask, retrieve only, stream an answer, swap a provider by configuration, replay a trace — each executed by a CI job against the local profile (after TASK-0444).

**Moves:** developer experience, documentation.

### TASK-0499 — Execute the documents' code

**Where.** `tests/unit/docs/test_cli_flag_parity.py` parses fenced blocks for flags only. Nothing runs a snippet from `README.md`, [quickstart.md](quickstart.md) or [python-api.md](python-api.md) — which is how two examples came to call a method that never existed (TASK-0435).

**Steps.** A test that extracts fenced `python` blocks marked runnable, executes them against in-memory doubles, and fails on an exception; a marker for blocks that are illustrative; the same for `bash` blocks limited to `fasterrag ... --help`-safe commands.

**Moves:** documentation, testing.

### TASK-0500 — Every error code, documented with a fix

**Where.** `tests/unit/test_errors.py:69` ties `PROBLEM_SPECS` to `ErrorCode`, and `:73` compares statuses to a hard-coded dict, not to the documents. [troubleshooting.md](troubleshooting.md) is organised by symptom and does not cover every code.

**Steps.** A table in [api-reference.md](api-reference.md) with code, status, meaning and remedy for all members, and a parity test that parses it.

**Moves:** documentation, developer experience.

### TASK-0502 — Configuration that can be upgraded

**Where.** No version key in `config.yaml` or the schema; [migration-guide.md](migration-guide.md) covers arriving from other frameworks, not moving between fasterRag versions.

**Steps.** `schema_version` at the top of `config.yaml`; the loader names the migration when it meets an older one; an "Upgrading" section per release in the changelog. Decide this before the first release, not after.

**Moves:** packaging, developer experience.

---

## 3. P2 — competitiveness; each needs a measurement before it becomes a default

### TASK-0462 — More formats through one conversion adapter

**Where.** `core/parsing/__init__.py:80-91` `_PARSERS` (plain text, Markdown, HTML, JSON, DOCX) and `:93-97` `_TUNED_PARSERS` (CSV/TSV, PDF): seven formats. `SUPPORTED_SUFFIXES` (`:99`), re-exported by `src/fasterrag/parsing.py`. OCR: `core/parsing/pdf.py:79` `_ocr_page`, with a resolution setting and no language setting.

**Steps.** After TASK-0439's parser registry: one optional parser adapter over a document-conversion library, behind an extra with a lazy import that names the extra, registered for the suffixes it adds (presentations, spreadsheets, e-books, email, rich text, images). Add `parsing.ocr_language`.

**Done when.** The parser invariants suite (`tests/unit/core/parsing/test_invariants.py`) passes on its output. **Moves:** ingestion.

### TASK-0463 — Parent-document retrieval

**Where.** `TextChunk` (`core/chunking/models.py:45`) has page, section and metadata, no parent. The stored payload (`workers/indexer.py:39-58`) has `document_id`, `chunk_index`, `span` and no parent reference.

**Steps.** `chunking.parent_child`: small retrieval chunks carrying `parent_id` and the parent's span; context assembly (`core/context.py`) substitutes the parent text, deduplicating children of one parent. Measure against the flat baseline on the TASK-0454 set.

**Traps.** A new `chunking.*` key: the comparability, baseline and lockfile rows of §0.4. **Moves:** retrieval, advanced RAG.

### TASK-0464 — An in-process store and a real filter grammar

**Where.** `MemoryVectorDB` exists only at `tests/unit/api/test_query_tenancy.py:58`. Filter operators (`adapters/vectordb/base.py:68`): `$eq $ne $gt $gte $lt $lte $in $nin`, flat AND; `validate_filter` at `:215`. Qdrant's `create_collection` (`qdrant.py:268-296`) creates no payload index.

**Steps.** Promote the store to `adapters/vectordb/memory.py`, make it pass the whole contract suite, register it (it then joins the gates in §0.4). Add `$and`, `$or`, `$not` to the neutral grammar with contract cases, then implement per adapter. Create payload indexes for `tenant`, `document_id` and configured fields.

**Moves:** embedding and vector infrastructure, developer experience (TASK-0444's local profile can use it).

### TASK-0465 — The decision on advanced retrieval

**What this task is.** A decision record, not code: which of agentic, graph, corrective and routing retrieval fasterRag will build, in what order, and what measured gain each must show to stay. Write it as an ADR under `docs/adr/`. The audit's recommendation, and the arithmetic in [engineering-audit.md](engineering-audit.md) §8, are the inputs: this category is 7% of the score and cannot stay at 15 if the total is to reach 90, but nothing here ships before TASK-0454 can measure it. TASK-0482, TASK-0483 and TASK-0484 are the three smallest candidates.

### TASK-0479 — Index tuning in configuration

**Where.** No HNSW `m`/`ef`, quantisation or on-disk keys anywhere. Qdrant `create_collection` sets size, distance, sparse IDF, shards and replication (`qdrant.py:275-294`) and upserts a collection's batch in one call (`:513-529`); pgvector builds a default HNSW index (`:648`). Related: TASK-0378.

**Steps.** A neutral `vector_db.index` block (`m`, `ef_construct`, `ef_search`, `quantization`, `on_disk`) mapped by each adapter, ignored with a logged notice where a backend has no equivalent; `vector_db.upsert_batch_size`. Defaults unchanged until measured (TASK-0454).

**Moves:** embedding and vector infrastructure, performance.

### TASK-0480 — Reranker input length and batch size

**Where.** `core/rerank/cross_encoder.py:61` constructs the model and `:143` calls `predict(pairs)` with no `max_length` and no `batch_size`; library defaults apply and nothing has measured them (BENCH-0003 measured cost only).

**Steps.** `retrieval.rerank_max_length` and `retrieval.rerank_batch_size`; measure quality and latency across a small grid on the TASK-0454 set; set defaults from the result.

**Moves:** retrieval, performance.

### TASK-0482 — Follow-up questions (query condensation)

**Depends on** TASK-0458 (conversation history) and TASK-0448 (the P5 query-transform stage). Condense history plus a follow-up into a standalone query before retrieval; measure on a conversational eval set; ship off by default. **Moves:** advanced RAG.

### TASK-0483 — Corrective retrieval

**Depends on** TASK-0448 and TASK-0454. When the top reranker score is under `retrieval.min_score`, or the P3 grade is under `generation.faithfulness_threshold`, retry once with a transformed query before refusing; bounded to one extra retrieval and charged to the cost governor; visible in the trace as a second retrieval span. **Moves:** advanced RAG, retrieval.

### TASK-0484 — Self-query metadata filters

**Depends on** TASK-0464's filter grammar. Extract a metadata filter from the question against a declared schema of filterable fields, validate it with `validate_filter`, and never widen a tenant or caller-supplied filter — only narrow it. An injection test set is mandatory (TASK-0446). **Moves:** advanced RAG.

### TASK-0485 — Language awareness

**Where.** No language detection in `core/parsing/`; no language handling in P1 or P3; the sparse leg is English-only until TASK-0447.

**Steps.** Detect a document's language at parse time into the payload; choose the sparse stemmer per language (with TASK-0447); instruct P1 to answer in the question's language (a prompt-contract change: version bump, [prompts.md](prompts.md), baselines). **Moves:** retrieval, core RAG.

### TASK-0495 — A written threat model

**Where.** [security.md](security.md) §7 "Threat-model notes" is three bullets. Traces store query text and retrieved chunks with a retention setting (`traces.retention_days`, `schema.py:665`) and no scrubbing.

**Steps.** A STRIDE table per trust boundary — caller to API, API to vector database, API to model provider, document content to prompt, operator to journal and traces — each row naming the mitigation and its test, or the open task. Include a data-at-rest section: what is stored, where, for how long, and how to delete it (with TASK-0443 and TASK-0474).

**Moves:** security, documentation.

### TASK-0501 — Link checking and versioned documentation

**Where.** The site is built with `mkdocs build --strict` (`vercel.json`), which checks internal links only. No external link check, no versioned site.

**Steps.** A scheduled link-check job (failures open an issue, they do not fail a push); versioned documentation from the first release tag.

**Moves:** documentation.

### TASK-0503 — Community files

**Where.** Present: `.github/SECURITY.md`, issue templates, a pull-request template. Absent: a code of conduct and `CODEOWNERS`.

**Steps.** Add both. `CODEOWNERS` should name an owner for the sensitive folders listed in `CLAUDE.md`. **Moves:** packaging and release readiness.

### TASK-0504 — Symbolic links during directory expansion

**Where.** `cli/sources.py:33-54` expands directories with no symlink or loop handling of its own. An escaping link inside an expanded directory reaches `_guard_path` in `resolve_sources` (`services/sources.py:363`) and fails the **whole job**, and only when `allowed_source_roots` is set.

**Steps.** Test a self-referential link and an escaping link; skip an escaping link as one dead-lettered document with a reason code instead of refusing the job. **Moves:** ingestion, security.

---

## 4. Earlier tasks that are on the path

These predate the audit and are defined in [todo.md](todo.md). They have no brief here beyond what a newcomer must know before touching them.

| Task | What to know first |
|---|---|
| TASK-0339, TASK-0381 | Tenant scoping of query and ingest. **A maintainer decision is recorded as pending in the entries — read both in full.** State today: `/v1/retrieve` and `/v1/query` take the tenant and push a payload filter only when `security.tenant_scoped_retrieval` is on (default off; `schema.py:735`); chunks without a `tenant` payload then match nothing, which is why a backfill (`services/backfill.py`, `fasterrag index backfill`) must run first. Collection names are scoped by `scoped_name` (`services/tenancy.py:76`) on the collections and admin routes only — the ingest and query paths do not apply it. |
| TASK-0326 | Nothing throttles an unauthenticated or wrong-key request: the limiter runs only after `KeyRegistry.resolve` succeeds (`api/auth.py:280-305`). A decision on trusted-proxy headers is needed before per-address limiting can be correct. |
| TASK-0412, TASK-0352 | The Milvus and Weaviate contract suites skip in CI. The stack is `deploy/compose.test-backends.yml`. `scripts/check_contract_ran.py` exists and is called by no workflow. |
| TASK-0130 | The queue-backed worker. Decide together with TASK-0440. |
| TASK-0407 to TASK-0410 | Blocking file I/O on the event loop: the lock store, the trace store, the archive, and the remainder. Pattern to copy: `Journal`'s async face over `*_blocking` twins. |
| TASK-0300 | 63 surviving mutants from the first mutation run. The nightly `mutation` job runs `scripts/mutation_run.py` with a ratchet (`CEILING`, `:376-383`); some line numbers in the entry have shifted. |
| TASK-0087, TASK-0084, TASK-0085 | Release mechanics; measurement on reference hardware (blocked on hardware); the disaster-recovery restore drill ([disaster-recovery.md](disaster-recovery.md) §2 is a manual procedure today). |
| TASK-0079, TASK-0049 | Export/import and the Pinecone adapter — the two open slice items named in `CLAUDE.md`. |
| TASK-0405, TASK-0406, TASK-0243 | Findings from the load and soak tiers, which no workflow runs (see TASK-0488). |
| TASK-0354 | The decision on randomised test ordering (see TASK-0490). |

---

## 5. Index: which tasks raise which category

Use this to choose work by the category you want to move. The bar each category must clear is in [engineering-scorecard.md](engineering-scorecard.md) §4.

| Category | Tasks |
|---|---|
| Architecture and design | TASK-0456, TASK-0466, TASK-0440, TASK-0130, TASK-0438, TASK-0457, TASK-0437 |
| Core RAG functionality | TASK-0438, TASK-0443, TASK-0449, TASK-0441, TASK-0458, TASK-0467, TASK-0468, TASK-0469, TASK-0485 |
| Document ingestion and processing | TASK-0470, TASK-0473, TASK-0431, TASK-0436, TASK-0459, TASK-0471, TASK-0472, TASK-0462, TASK-0504 |
| Retrieval and search quality | TASK-0454, TASK-0447, TASK-0448, TASK-0481, TASK-0463, TASK-0480 |
| Embedding and vector infrastructure | TASK-0442, TASK-0451, TASK-0452, TASK-0412, TASK-0352, TASK-0477, TASK-0478, TASK-0464, TASK-0479, TASK-0049 |
| Advanced RAG capabilities | TASK-0465, TASK-0448, TASK-0482, TASK-0483, TASK-0484, TASK-0463 |
| Extensibility and modularity | TASK-0439, TASK-0486 |
| Performance and scalability | TASK-0454, TASK-0130, TASK-0437, TASK-0442, TASK-0466, TASK-0487, TASK-0488, TASK-0407 to TASK-0410 |
| Reliability and error handling | TASK-0436, TASK-0437, TASK-0473, TASK-0449, TASK-0450, TASK-0453, TASK-0474, TASK-0475, TASK-0476 |
| Testing and quality assurance | TASK-0455, TASK-0412, TASK-0441, TASK-0454, TASK-0481, TASK-0489, TASK-0490, TASK-0300 |
| Security | TASK-0339, TASK-0381, TASK-0446, TASK-0460, TASK-0326, TASK-0491, TASK-0492, TASK-0493, TASK-0494, TASK-0495, TASK-0496 |
| Developer experience and APIs | TASK-0444, TASK-0461, TASK-0497, TASK-0498, TASK-0500 |
| Documentation | TASK-0440, TASK-0499, TASK-0500, TASK-0501, TASK-0486 |
| Packaging and release readiness | TASK-0445, TASK-0087, TASK-0444, TASK-0502, TASK-0496, TASK-0503 |
