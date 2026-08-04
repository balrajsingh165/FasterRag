# cli-reference.md — Terminal Commands

The CLI is one half of the control plane (the other is the [REST API](api-reference.md)); both call the same service layer, so behavior is identical. Root command: `fasterrag`.

Global flags (valid on every command):

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `./config.yaml` | Config file to load. |
| `--collection NAME` | config default | Target collection. |
| `--set KEY=VALUE` | — | Override any setting in [config-reference.md](config-reference.md) for this invocation. Repeatable. |
| `--json` | off | Machine-readable JSON output (stable schemas; intended for scripts and CI). |
| `--quiet` / `-q` | off | Errors only. |
| `--verbose` / `-v` | off | Debug logging. |

Exit codes: `0` success · `1` generic failure · `2` usage/validation error · `3` dependency unreachable · `4` doctor/preflight failure · `5` regression gate blocked.

### `--set` — overriding configuration per invocation

Takes a dotted key and a value, and applies it over `config.yaml` before validation. Useful for trying a setting without editing the file, and for scripting a sweep across values.

```console
$ fasterrag ingest ./docs --set chunking.chunk_size=512 --set chunking.strategy=layout
$ fasterrag query "expense limit" --set retrieval.rerank=false
```

Values are read as YAML scalars, so `512`, `0.75`, `true`, and `null` arrive as the types the schema expects; a bare word is a string.

An override is held to **exactly** the rules a file value is — range, type, unknown-key, and cross-field. `--set retrieval.top_k=200` fails the same way writing it into the file would, and `--set chunking.chunk_sise=512` is rejected as an unknown key rather than silently ignored. Overrides are merged before validation rather than assigned afterwards, which is what keeps the cross-field rules running.

Run `fasterrag config show` to see every key `--set` accepts.

---

## `fasterrag serve`

Run the API server (`app.host`/`app.port` from config).

| Flag | Description |
|---|---|
| `--reload` | Dev auto-reload. |
| `--host`, `--port` | Override config bind address. |

## `fasterrag worker`

Run the pipeline worker pools (CPU pool + embedding pool + indexer) for the configured queues.

| Flag | Description |
|---|---|
| `--pools cpu,embed,index` | Restrict which pools this process runs (default: all). |
| `--cpu-workers N`, `--embed-workers N` | Override `workers.*` sizes. |

## `fasterrag ingest <path|url> [...]`

Submit sources for ingestion (async job; same path as `POST /v1/ingest`).

| Flag | Description |
|---|---|
| `--metadata KEY=VALUE` | Repeatable; merged into chunk metadata. |
| `--priority-class NAME` | Tiered-embedding routing class (D9). |
| `--recursive` | Recurse into directories. |
| `--watch` | Follow the job and print per-stage progress until completion. |
| `--dry-run` | Parse + chunk only; report what would be indexed (no embedding, no writes). |

## `fasterrag query "<question>"`

Run a query (same path as `POST /v1/query`).

| Flag | Description |
|---|---|
| `--top-k N` | Override `retrieval.top_k`. |
| `--filter KEY=VALUE` | Repeatable metadata filter. |
| `--no-stream` | Wait for the full answer instead of streaming tokens. |
| `--show-chunks` | Print retrieved chunks with scores (retrieval debugging). |
| `--show-timings` | Print per-stage latency breakdown. |

## `fasterrag index <subcommand>`

| Subcommand | Description |
|---|---|
| `list` | Collections with vector counts, embedding model+version, drift status. |
| `create NAME` | Create a collection (`--distance`, `--shards`, `--replicas`). |
| `delete NAME` | Drop a collection (`--force` required if alias target). |
| `reembed NAME` | D2 zero-downtime re-embed: blue/green build → eval gate → alias swap (`--no-eval-gate` for dev; `--watch`). |
| `rollback NAME` | Alias flip back to the retained previous collection (within `index.reindex.rollback_retention_hours`). |
| `lock verify [NAME]` | D1: verify `index.lock` against current config/corpus; report any drift (model version, config hash, content hashes). Exit 1 on drift. |

## `fasterrag provision <tool>`

Config-driven provisioning for `qdrant`, `langfuse`, `grafana` (idempotent; **doctor-gated**; zero code changes). Prints the running URL on success (Langfuse: `http://<host>:3000`).

| Flag | Description |
|---|---|
| `--status` | Show provisioning/health state instead of provisioning. |
| `--down` | Stop the managed containers for the tool (data volumes preserved). |

## `fasterrag status`

One-screen system status: API reachability, worker pools, queue depths, DLQ depth, vector DB health per adapter, circuit-breaker states, cache hit rates, active jobs.

## `fasterrag doctor`

D10 preflight diagnostics. Checks: Docker present and running · required ports free (API, dashboard, 6333 **and** 6334 for Qdrant, Langfuse stack ports) · disk space · RAM/GPU availability · vector DB reachable in the configured mode (docker / external local / remote host:port) · API keys valid (non-destructive provider ping) · config schema valid. **Every failed check prints a concrete fix-it instruction.** `doctor` must pass before any auto-provisioning runs.

| Flag | Description |
|---|---|
| `--fix` | Apply safe automatic fixes (e.g. create missing named volume). |
| `--json` | Machine-readable report (same schema as `GET /v1/admin/doctor`). |

## `fasterrag estimate <path|url> [...]`

D9 preflight cost estimator — BEFORE ingestion: document/token counts and projected embedding cost per configured provider.

With `chunking.contextual_enrichment: true` the report also carries the **enrichment** cost, separately from embedding rather than blended into it — enrichment is a *generation* charge on a different model at different rates, and one combined number would hide which knob to turn. It is one call per chunk, each sending the whole parent document, so the prompt cost scales with document tokens × chunk count. That figure is quoted **uncached**: prompt caching is what makes enrichment affordable, but its discount depends on the provider and the cache window, so quoting a number fasterRag cannot verify would understate a real bill. An over-estimate an operator can reason about is the safer error.

Wall-clock time is deliberately **not** projected — throughput has not been measured on reference hardware, and an unmeasured projection would be a claim without a measurement ([benchmarks.md](benchmarks.md)).

| Flag | Description |
|---|---|
| `--provider NAME` | Compare against a specific provider instead of the configured one. |
| `--all-providers` | Cost table across all configured/known providers. |

## `fasterrag traces <subcommand>`

D8 trace inspection. A trace id is a 32-character hex string nobody retains between the query and the investigation, so listing is what makes stored traces reachable at all.

| Subcommand | Description |
|---|---|
| `list` | Recent trace ids, newest first (`--limit N`). |
| `show <trace_id>` | One trace in full: query, collection, candidate count, and each span with its duration and attributes. |

## `fasterrag replay --trace <id>`

D8 time-travel replay: re-execute a past query under a candidate config and show a side-by-side diff of retrieval sets and answers.

| Flag | Description |
|---|---|
| `--candidate candidate.yaml` | Candidate config to replay under. Defaults to the current config, which is the determinism check: an unchanged config must reproduce the retrieval set exactly. |
| `--diff-only` | Only show what changed (added/removed/reordered chunks, answer diff) — omits the full answer text. |

Replay never writes: it stores no trace of its own and never populates the semantic cache, so investigating an incident cannot alter the evidence. The retrieval set is what replay guarantees is reproducible; answer *wording* is not, because a provider at non-zero temperature is not deterministic — which is why the diff reports citations alongside the text.

## `fasterrag benchmark`

Run the benchmark suite from [performance.md](performance.md) and print/append ledger-formatted results ([benchmarks.md](benchmarks.md)).

| Flag | Description |
|---|---|
| `--suite ingest\|query\|eval\|all` | Which suite (default `all`). |
| `--dataset NAME` | Named dataset fixture. |
| `--ledger` | Emit a ready-to-commit benchmark ledger entry (includes hardware fingerprint + commit hash). |

## `fasterrag export --out <archive>`

D11 portability: export documents, chunks, metadata, and the index manifest to a portable archive (vendor-neutral format).

| Flag | Description |
|---|---|
| `--include-vectors` | Include raw vectors (enables direct vector copy on import where dimensions match). |

## `fasterrag import <archive>`

D11: import a previously exported archive.

| Flag | Description |
|---|---|
| `--reembed` | Re-embed with the currently configured model instead of copying vectors. |
| `--target-collection NAME` | Import into a specific collection. |

## `fasterrag config init`

Write the canonical `config.yaml` into the current directory, plus `.env.example` if absent. **This is the first command to run after `pip install fasterrag`** — every other command needs a `config.yaml`, and an installed package has no repository to copy one from.

The file written is byte-identical to the repository's `config.yaml`: it is force-included into the wheel rather than duplicated, so the template cannot drift from the one [config-reference.md](config-reference.md) documents.

| Flag | Description |
|---|---|
| `--path PATH` | Where to write; defaults to `./config.yaml`. Parent directories are created. |
| `--force` | Overwrite an existing file. Without it an existing `config.yaml` is never touched — exit 2. |

The secrets template is always written as `.env.example`, never as `.env`: the loader reads `.env`, so writing placeholders there could overwrite real credentials, and a file of `change-me` values that startup treats as configured is worse than no file. An existing `.env.example` is left alone.

```
$ fasterrag config init
wrote config.yaml
wrote .env.example
next: copy .env.example to .env, fill in what config.yaml references,
      then run 'fasterrag doctor'
```

## `fasterrag config validate`

Validate `config.yaml` + `.env` presence of referenced env vars without starting anything. Exit 0/2. Same fail-fast checks as startup ([config-reference.md](config-reference.md) cross-field rules).

Startup also **refuses to run** under settings the schema accepts but nothing enforces — `security.auth`, `security.multi_tenancy`, and either `cost.*_token_budget`. Enabling one raises `CONFIG_INVALID` naming the slice that will implement it. Reporting a protection the system does not have is the one failure mode worse than an outage.

## `fasterrag config show`

List every setting with its effective value and its default, walking nested sections so a new schema field appears without being registered anywhere. Answers "what can I tune?" without reading [config-reference.md](config-reference.md). Exit 0/2.

Unlike `config validate`, a referenced-but-missing environment variable does **not** stop the listing — this command is most useful on the half-configured installation that `validate` refuses. Invalid configuration still exits 2, because printing values that failed validation would show settings nothing will use.

| Flag | Description |
|---|---|
| `--changed` | List only settings that differ from their default — the fastest way to see what a deployment customised. |

```console
$ fasterrag config show --changed
* chunking.chunk_size                              512                      default=768
* chunking.token_counter                           'model'                  default='auto'
```

## `fasterrag autopilot run`

D6 eval-driven auto-tuning (requires `autopilot.enabled: true`). Generates a golden Q&A set from the corpus, searches chunk size / top_k / hybrid weights / rerank settings, and writes a **suggested config diff with measured deltas** (e.g. recall@10 before/after) to stdout and `autopilot-suggestion.yaml`. **Never applies changes** — a human reviews and applies the diff.

| Flag | Description |
|---|---|
| `--budget-minutes N` | Search time budget. |
| `--golden-set PATH` | Reuse an existing golden set instead of generating one. |

## `fasterrag autopilot generate-golden-set <sources...>`

Generate a golden Q&A set from a corpus with the P4 contract ([prompts.md](prompts.md)) and write it as JSONL. This is the prerequisite for both `autopilot run` and the eval regression gate (D7), so it exists as its own command rather than only as a step inside tuning.

Sources are parsed and chunked exactly as ingestion would — no collection is needed, and none is read. A golden set is usually wanted *before* the index exists, which is the point of tuning against one.

| Flag | Description |
|---|---|
| `--out PATH` | Where to write it; defaults to `./golden.jsonl`. An existing file is never overwritten. |
| `--size N` | How many records to aim for (default 100). |
| `--seed N` | Makes sampling and adversarial selection reproducible. |

A fraction of the set is generated deliberately **unanswerable** from the corpus, so the harness measures whether retrieval declines to return something confident when nothing relevant exists.

The output is generated text and should be reviewed before it becomes a baseline: `source` is recorded as `autopilot` rather than `human` precisely so a later reader can tell which records were curated.

```
$ fasterrag autopilot generate-golden-set ./corpus/*.md --out golden.jsonl --size 6
wrote golden.jsonl
records         6
  adversarial   1
  dropped       0
  generated     6
review the questions before using them as a baseline; they are generated
```
