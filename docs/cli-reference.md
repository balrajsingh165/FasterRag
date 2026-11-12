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

`FASTERRAG_SET` is the environment-variable form, for containers and anything else without a command line to add flags to. It takes the same `dotted.key=value` pairs, comma-separated:

```console
$ FASTERRAG_SET="chunking.chunk_size=512,retrieval.top_k=5" fasterrag serve
```

An explicit `--set` beats the environment, because an operator reaching for the flag is overriding what the deployment already set and losing to a variable they cannot see would be silent. A value containing a comma has to use the flag, which needs no separator. Environment overrides are validated exactly as file values and flags are.

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

> **Status (2026-11-11): this process does no pipeline work.** `fasterrag worker` checks that the vector database answers, prints `worker ready`, and waits. Ingestion runs inside the process that accepted the job — a task held by the API's job registry (`services/jobs.py`), or inline in the CLI and the Python facade — because there is no queue backend for a separate process to consume (TASK-0130). Running it is harmless and gains nothing; whether to build the queue-backed worker or remove the command is TASK-0440.

| Flag | Description |
|---|---|
| `--pools cpu,embed,index` | Restrict which pools this process runs (default: all). |
| `--cpu-workers N`, `--embed-workers N` | Override `workers.*` sizes. |

## `fasterrag ingest [path|url] [...]`

Submit sources for ingestion (async job; same path as `POST /v1/ingest`). Give at least one source, or `--resume JOB_ID` and none; with neither it exits `2`.

| Flag | Description |
|---|---|
| `--resume JOB_ID` | Run an interrupted job again from its last checkpoint instead of submitting sources (TASK-0436). The job must be one that was settled `failed` with `reason_code: INTERRUPTED` — cancelled by a shutdown, or left `running` by a process that was killed and settled by the next server to start. A job still `running` whose record and document log have both been quiet for `reliability.stale_job_seconds` is settled and then resumed, so a killed `fasterrag ingest` can be continued without a server; one written to more recently is refused, because another process may be working on it. Any other job is refused with `CONFLICT` and exit `1`, so a finished corpus is never re-run by mistake. It prints `resuming        from document N` and then the usual summary. The job's sources come from its record, so passing sources, or `--dry-run`, with `--resume` exits `2`. `--metadata` and `--priority-class` are **not** stored with a job: pass them again if the original run used them, or the remaining documents are indexed without them. `--watch` works as it does for a new job. |
| `--metadata KEY=VALUE` | Repeatable; merged into chunk metadata. |
| `--priority-class NAME` | Tiered-embedding routing class (D9). Sets `priority_class` in chunk metadata, which is what `embeddings.tiering.rules[].match` matches on. |
| `--recursive` | Descend into subdirectories. Without it a directory source contributes only its immediate files. Hidden files and `.git`, `node_modules`, `__pycache__`, and virtualenv directories are skipped either way; files from a directory are ingested in sorted order so the same command produces the same job. |
| `--watch` | Follow the job while it runs: the job id is printed as soon as the job is accepted, then a `progress` line each time the per-document tally in the journal changes (polled once a second), then the usual summary. Progress lines are appended, never redrawn, so a terminal and a redirected file get the same output; there is nothing TTY-specific about it. Under `--json` or `--quiet` it prints a notice to stderr and is skipped, because both modes suppress every human line and the result document is unchanged either way. Under `--dry-run` it prints a notice and is skipped: a dry run creates no job. |
| `--dry-run` | Parse + chunk only; report what would be indexed (no embedding, no writes). This is the D9 estimator, so it obeys `cost.estimator`: with the estimator off it exits `2` naming the setting instead of pricing the corpus anyway. |

## `fasterrag query "<question>"`

Run a query (same path as `POST /v1/query`).

| Flag | Description |
|---|---|
| `--top-k N` | Override `retrieval.top_k`. |
| `--filter KEY=VALUE` | Repeatable metadata filter. |
| `--no-stream` | Wait for the full answer instead of streaming tokens. Streaming is the default at a terminal; `--json` and `--show-chunks` imply `--no-stream`, because a partial document is not a document and the candidate set only exists on the non-streaming path. |
| `--show-chunks` | Print the retrieved candidate set with ranks and scores (retrieval debugging). Implies `--no-stream`. |
| `--show-timings` | Print per-stage latency breakdown. |

## `fasterrag index <subcommand>`

| Subcommand | Description |
|---|---|
| `list` | Collections with vector counts, embedding model+version, drift status. |
| `create NAME` | Create a collection (`--distance`, `--shards`, `--replicas`). |
| `delete NAME` | Drop a collection (`--force` required if alias target). |
| `reembed NAME <sources...>` | D2 zero-downtime re-embed: blue/green build → eval gate → alias swap (`--no-eval-gate` for dev, `--dataset PATH` to gate on a specific eval dataset). `--recursive` expands directory sources; `--watch` follows the green build the same way `ingest --watch` does, printing the build's job id and a `progress` line per tally change before the gate and the swap are reported. |
| `rollback NAME` | Alias flip back to the retained previous collection (within `index.reindex.rollback_retention_hours`); `--to NAME` restores a specific retained build rather than the most recent. |
| `lock verify [NAME]` | D1: verify `index.lock` against current config/corpus; report any drift (model version, config hash, content hashes). Exit 1 on drift. |
| `tenants [NAME]` | Count a collection's points by `tenant` payload tag, or every collection's when no name is given. Read-only. This is what to run **before** `index backfill-tenant`: with `security.tenant_scoped_retrieval` on, a point carrying no tag is invisible to every query, so the untagged count is the size of the blackout the flag would cause ([security.md](security.md) §3). `--untagged PATH` additionally writes the untagged points to a JSONL file grouped by document (`document_id`, `source_uri`, chunk count), which is the evidence to read before asserting who owns them — chunks with no `document_id` group into one row with an empty id, because they cannot be attributed per document at all. |
| `backfill-tenant NAME` | Write the `tenant` payload tag onto a collection's points through the adapters' metadata-only update path. `--tenant ID` (required) is the id to write. **Dry run by default** — it reports what it would write and writes nothing; `--apply` performs the write. A point already tagged for another tenant is refused with `CONFLICT`, because untagged points in a collection holding two tenants cannot be attributed to either; two flags are the explicit answers and the refusal names both. `--overwrite` claims the whole collection, relabelling every point in it. `--assume-untagged` claims only the untagged residue, leaving existing tags alone — the right one for a mixed collection — and the report records that the ownership was asserted rather than derived (`assumed` in `--json`). Passing both is `VALIDATION_FAILED`: they assert different things about the same collection. `--batch-size N` sets how many points travel per update call (default 512). Re-running is safe: only untagged points are selected, so a run that died halfway resumes by being run again. |

## `fasterrag documents <subcommand>`

Per-document operations on a collection (same path as `DELETE /v1/collections/{name}/documents/{document_id}`). The collection is the global `--collection`, defaulting to `vector_db.collection.default_name`.

| Subcommand | Description |
|---|---|
| `delete DOCUMENT_ID` | Delete one document: every chunk carrying that `document_id`, its dedup record in the journal (so the same file can be ingested again), its entry in `index.lock`, and the configured semantic cache. `DOCUMENT_ID` is the `d_…` id `GET /v1/ingest/{job_id}/documents` lists for each source of a job; the CLI has no document listing of its own yet (TASK-0469). `--tenant ID` restricts the delete to chunks whose `tenant` payload tag is that id; without it every chunk of the document is removed, whoever it is tagged for, because the CLI acts as the operator. **Idempotent:** a document the journal has no record of exits `0` and says so, since the vector database is still asked to delete and cannot report whether anything matched; a collection that does not exist exits `1` with `NOT_FOUND`. A server already running with `cache.backend: memory` holds its cache in another process and is not invalidated by this command — its entries expire on `cache.ttl`; use the REST route to clear them at once. |

## `fasterrag provision <tool>`

Config-driven provisioning for `qdrant`, `langfuse`, `grafana` (idempotent; **doctor-gated**; zero code changes). Prints the running URL on success (Langfuse: `http://<host>:3000`).

| Flag | Description |
|---|---|
| `--status` | Show provisioning/health state instead of provisioning. |
| `--down` | Stop the managed containers for the tool (data volumes preserved). |

## `fasterrag status`

One-screen system status: API reachability, worker pools, queue depths, DLQ depth, vector DB health per adapter, circuit-breaker states, cache hit rates, active jobs.

## `fasterrag doctor`

D10 preflight diagnostics. Checks: Docker present and running · required ports free (API, dashboard, 6333 **and** 6334 for Qdrant, Langfuse stack ports) · disk space · RAM/GPU availability · the named Docker storage volume exists (`vector_db.mode: docker`; omitted rather than passed when the daemon is down, since a check that could not run must never report a pass) · vector DB reachable in the configured mode (docker / external local / remote host:port) · API keys valid (non-destructive provider ping) · config schema valid. **Every failed check prints a concrete fix-it instruction.** `doctor` must pass before any auto-provisioning runs.

| Flag | Description |
|---|---|
| `--fix` | Apply the repairs that are safe to apply, then **run every check again** and report what is true afterwards. See below. |
| `--json` | Machine-readable report (same schema as `GET /v1/admin/doctor`; `--fix` adds a `fixes` object alongside `passed` and `checks`). |

### `doctor --fix`

`--fix` never takes a destructive or surprising action. A repair qualifies only if it is idempotent, cannot destroy state, and cannot surprise, which today is exactly two:

| Failed check | Repair |
|---|---|
| `vector_db_volume` | `docker volume create <vector_db.docker.volume>`. Converges — an existing volume is returned untouched, so an index can never be reinitialised. |
| `vector_db` | `docker start` the **stopped** managed container, then wait until the backend answers a real request. Only ever a container carrying fasterRag's own `fasterrag.managed=true` label: a container of that name without the label belongs to somebody else and is left alone. |

Everything else is reported as needing a human, with the reason and the concrete command already printed by the check's own fix-it line. `--fix` will not free a port (that means killing whatever holds it), free disk (that means deleting files fasterRag did not write), write `.env`, rewrite `config.yaml`, or **create** a container — creating one pulls an image and publishes ports, which is `fasterrag provision qdrant`'s job.

Every attempted repair is verified by re-running the whole report afterwards, and each attempt is classified from that second report, never from the repair returning without an error:

| Status | Meaning |
|---|---|
| `fixed` | Repaired, and the re-check now passes. |
| `not fixed` | Attempted, and the check still fails (or the repair itself errored). |
| `needs human` | No safe automatic repair; the reason is printed. |

**Exit codes are unchanged**: `0` when every check passes after the repairs, `4` when any still fails. `--fix` cannot launder a failed preflight into a green one, so it stays safe to wire into a setup script that branches on `4`.

## `fasterrag ci-status`

Reads the GitHub Actions API and **fails when any workflow's last conclusion on a branch is not green**. It exists because `main` was red for 81 days once (TASK-0261) and for three days again in November 2026 (TASK-0313) with nobody noticing: the pre-commit hook passes locally, so every commit looks green to the person making it, and until this command nothing in the repository read the Actions API.

It is a gate, not a report. Every way it can fail to learn something exits non-zero:

| Situation | Exit |
|---|---|
| Every workflow's last completed run concluded `success` | `0` |
| Any workflow's last completed run is `failure`, `cancelled`, or `timed_out` | `1` |
| Any workflow's last run concluded `success` with **every job skipped** | `1` |
| Any workflow has **no** completed run on the branch | `1` |
| No workflows were selected at all (empty `.github/workflows`, or everything excluded) | `1` |
| The API could not be read — no network, rate limit, 404 — and nothing else was red | `3` |
| No `github.com` `origin` remote, or `--token-env` names a variable that is unset | `2` |

A `cancelled` run is not a pass: `concurrency: cancel-in-progress` left 21 commits on `main` with no completed verdict, and no verdict is not a green one. Nor is a run whose every job was skipped — GitHub concludes such a run `success`, but a workflow that ran nothing found nothing.

### `skipped` is not `fine`

The report names every step a failing job **skipped**, because that is where the damage hides. On run `42522906097` (`6be46f9`, 2026-11-08) the `quality` job failed at `Typecheck` — and the thirteen steps after it, nine of them named gates, were all reported `skipped`:

```console
$ fasterrag ci-status --workflow ci.yml
balrajsingh165/FasterRag @ main
  RED ci.yml: failure (#163 6be46f9e) -- quality/Typecheck -- 13 later step(s) never ran
      https://github.com/balrajsingh165/FasterRag/actions/runs/42522906097
      never ran: quality/Docs match the repository
      never ran: quality/Denials match the tree, not just the ledger
      ...
```

A reader told only "red on `Typecheck`" will fix the typecheck and expect green. In fact nine documentation gates, the unit suite, both coverage gates and the doctor smoke test never ran, so none of them had the chance to object to anything else in that commit. Runner bookkeeping (`Set up job`, `Complete job`, `Post Run …`) is excluded from the count; a step skipped by its own `if:` inside a job that *passed* is a decision, not a gap, and is not reported.

**The listing is never filtered by trigger**, and that is the point. On 2026-11-07 `main` was simultaneously green on the `ci` *push* run and red on the `security` and `nightly` *scheduled* runs of the same commit — so a check that read only the trigger a human looks at would reproduce the bug instead of catching it.

| Flag | Description |
|---|---|
| `--branch NAME` | Branch whose runs are read. Default `main`. |
| `--workflow FILE` | A workflow file to check. Repeatable. Defaults to every `*.yml`/`*.yaml` file in `.github/workflows`, discovered from the working directory, so a fourth workflow is gated the day it is added. |
| `--exclude FILE` | A workflow file to leave out. Repeatable. |
| `--token-env NAME` | Read a bearer token from this environment variable. Anonymous when omitted. Nothing is ever read from `GITHUB_TOKEN` or a `.netrc` unless that variable is the one named. |

```console
$ fasterrag ci-status
balrajsingh165/FasterRag @ main
  ok  ci.yml: success (#164 882a07ea)
  ok  nightly.yml: success (#100 6be46f9e)
  ok  security.yml: success (#23 6be46f9e)
3 workflow(s) green on main
```

### Why `ci.yml` excludes itself in CI

`ci.yml` runs this command against the other workflows (`--exclude ci.yml`) rather than all three, because a workflow that gates on its own previous conclusion has no working state. If it treats its own last red run as a failure, the push that fixes the cause still reads that red run and fails, and the branch can never return to green. If it excuses it, it excuses every red, because the latest completed run of `ci.yml` on `main` is always an ancestor of the commit being tested. There is no middle setting: self-checking is either deadlocked or vacuous.

Excluding it loses nothing. A workflow's own push run is attached to the commit, shown in the PR merge box, and emailed on failure — it is visible by construction. The invisible ones are exactly the workflows whose latest run came from a `schedule` nobody opens, and those are the ones the step checks.

The anonymous GitHub API budget is 60 requests an hour **per address**, shared with every other job on a runner, so the CI step passes `--token-env GITHUB_TOKEN` with the workflow's own `actions: read` token. Locally, anonymous is fine: two requests per workflow.

## `fasterrag estimate <path|url> [...]`

D9 preflight cost estimator — BEFORE ingestion: document/token counts and projected embedding cost per configured provider. Requires `cost.estimator: true` (the default); with it off the command exits `2` with a `VALIDATION_FAILED` problem naming the setting, because a disabled estimator reporting zeroes would read as a corpus that costs nothing.

With `chunking.contextual_enrichment: true` the report also carries the **enrichment** cost, separately from embedding rather than blended into it — enrichment is a *generation* charge on a different model at different rates, and one combined number would hide which knob to turn. It is one call per chunk, each sending the whole parent document, so the prompt cost scales with document tokens × chunk count. That figure is quoted **uncached**: prompt caching is what makes enrichment affordable, but its discount depends on the provider and the cache window, so quoting a number fasterRag cannot verify would understate a real bill. An over-estimate an operator can reason about is the safer error.

Wall-clock time is deliberately **not** projected — throughput has not been measured on reference hardware, and an unmeasured projection would be a claim without a measurement ([benchmarks.md](benchmarks.md)).

| Flag | Description |
|---|---|
| `--recursive` | Descend into subdirectories, exactly as `ingest --recursive` does. Directory arguments are expanded the same way with or without it, so the estimate prices the file set the ingestion it precedes would read. |
| `--provider NAME` | Compare against a specific provider instead of the configured one. |
| `--all-providers` | Cost table across all configured/known providers. |

## `fasterrag traces <subcommand>`

D8 trace inspection. A trace id is a 32-character hex string nobody retains between the query and the investigation, so listing is what makes stored traces reachable at all.

| Subcommand | Description |
|---|---|
| `list` | Recent trace ids, newest first, one per line. `--limit N` caps the count (default `20`). |
| `show <trace_id>` | One trace in full: id, query, collection, creation time, candidate count, and each span with its duration in milliseconds and its attributes. `trace_id` is positional and required. |

Both subcommands take the global flags (`--config`, `--collection`, `--set`, `--json`, `--quiet`, `--verbose`); there are no others. Under `--json`, `list` writes `{"traces": [<id>, ...]}` — an empty array when there are none — and `show` writes the whole trace document.

**Exit codes**: `0` for a successful `list` *including* an empty one, and for a `show` that found its trace; `1` when `show` names a trace that is not stored (expired past `traces.retention_days`, or never written); `2` when the configuration itself fails to load. An empty `list` is a success because "no traces yet" is a true answer, not a failure — in plain output it says so, and adds `(traces.store is false)` when storage is switched off, so an empty result can never be mistaken for a broken store.

## `fasterrag replay --trace <id>`

D8 time-travel replay: re-execute a past query under a candidate config and show a side-by-side diff of retrieval sets and answers.

| Flag | Description |
|---|---|
| `--candidate candidate.yaml` | Candidate config to replay under. Defaults to the current config, which is the determinism check: an unchanged config must reproduce the retrieval set exactly. |
| `--diff-only` | Only show what changed (added/removed/reordered chunks, answer diff) — omits the full answer text. |

Replay never writes: it stores no trace of its own and never populates the semantic cache, so investigating an incident cannot alter the evidence. The retrieval set is what replay guarantees is reproducible; answer *wording* is not, because a provider at non-zero temperature is not deterministic — which is why the diff reports citations alongside the text.

## `fasterrag backup <destination>`

Capture the artifact inventory of [disaster-recovery.md](disaster-recovery.md) §1 — backend-native collection snapshots plus the journal, lockfiles, traces, and `config.yaml` — into a new timestamped **set** under the destination. Runs accumulate a history; they do not overwrite one another.

`.env` is never captured. Secrets belong in the operator's secret manager, and a backup routine that swept them into a tarball would put every credential wherever backups are stored. Its absence from a restore is expected, not an omission.

| Flag | Description |
|---|---|
| `--collections NAME...` | Limit to these collections; every one by default. |
| `--retain N` | Backup sets to keep (default 14). Older sets are pruned **together with the backend snapshots they reference**, so retention reclaims space inside the vector database rather than orphaning it. Values below 1 are refused. |

Cadence is deliberately not fasterRag's job — schedule it with cron, a systemd timer, or Task Scheduler ([disaster-recovery.md](disaster-recovery.md) §1).

## `fasterrag restore <source>`

Restore from a backup. `source` is either one set, or the destination holding several — in which case the newest is used, because during an incident the path an operator has to hand is the one they backed up to.

A collection whose restored vector count differs from what the manifest recorded is **reported, not raised**: a partial restore an operator can see beats one that aborts halfway with no report of how far it got.

| Flag | Description |
|---|---|
| `--collections NAME...` | Limit to these collections; every one in the manifest by default. |
| `--collections-only` | Skip the control files. The §4 shortcut for restoring one corrupted collection into an otherwise healthy deployment. |

## `fasterrag benchmark`

Run the benchmark suite from [performance.md](performance.md) and print/append ledger-formatted results ([benchmarks.md](benchmarks.md)).

| Flag | Description |
|---|---|
| `--suite ingest\|query\|eval\|all` | Which suite (default `all`). |
| `--dataset NAME` | Dataset name recorded in the ledger entry. |
| `--query TEXT` | The question the `query` suite asks. Required for that suite; it has no default, because a benchmark that invents its own query measures nothing a reader can reproduce. |
| `--iterations N` | Calls per repetition (default `20`). |
| `--ledger` | Emit a ready-to-commit benchmark ledger entry (includes hardware fingerprint + commit hash). |
| `--sources PATH [...]` | Corpus the `ingest` suite measures against. Directories are expanded to the files inside them; an expansion that finds nothing is refused rather than measured, because a ledger entry for a corpus of zero documents is a published number backed by nothing. |

## `fasterrag export --out <archive>`

D11 portability: export documents, chunks, metadata, and the index manifest to a portable archive (vendor-neutral format).

| Flag | Description |
|---|---|
| `--include-vectors` | Include raw vectors (enables direct vector copy on import where dimensions match). |

## `fasterrag import <archive>`

D11: import a previously exported archive. The archive is verified whole — checksums, manifest counts, referential integrity, vector width — before anything is written, so a refused archive leaves the target untouched (exit 2). A write that fails partway drops the collection the import created; one that already existed is left alone and named in the error, because re-running the same import completes it. The restored collection gets a fresh `index.lock`, and the archive's own lock is kept beside it as `<collection>.source.lock.json`. See [archive-format.md](archive-format.md) §Import semantics.

| Flag | Description |
|---|---|
| `--reembed` | Re-embed with the currently configured model instead of copying vectors. Always legal, because it derives the vectors it stores. |
| `--target-collection NAME` | Import into a specific collection. |

Without `--reembed` the archived vectors are copied, and that is refused unless the target's embedding identity matches the manifest — from the target collection's own lockfile when it has one, otherwise from `embeddings.model` and `embeddings.dimensions`. Copying vectors produced by another model builds a collection that answers every query confidently and wrongly, with nothing in the index to say why, so the refusal names each mismatched field rather than summarising.

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
| `--changed` | List only settings that differ from their default — the quickest way to see what a deployment customised. |

```console
$ fasterrag config show --changed
* chunking.chunk_size                              512                      default=768
* chunking.token_counter                           'model'                  default='auto'
```

## `fasterrag autopilot run`

D6 eval-driven auto-tuning (requires `autopilot.enabled: true`). Scores against the golden set `--dataset` names — it generates nothing and refuses to run without one — searches the query-time parameters it can score without a rebuild — hybrid weights, `rrf_k` and rerank settings (`top_k` is deliberately excluded: measuring recall@k retrieves exactly k, so a top_k candidate cannot change the score; chunk size and overlap need the separate, priced index-time sweep and have no CLI surface yet, TASK-0145), and writes a **suggested config diff with measured deltas** (e.g. recall@10 before/after) to stdout and `autopilot-suggestion.yaml`. **Never applies changes** — a human reviews and applies the diff.

| Flag | Description |
|---|---|
| `--budget-minutes N` | Search time budget. |
| `--dataset PATH` | **Required.** Directory holding the golden set to score against. The command does not generate one — run `autopilot generate-golden-set` first, or point at `tests/eval/datasets/handbook`. |
| `--out PATH` | Where the suggestion is written (default `autopilot-suggestion.yaml`). It is never applied; a human does that. |

## `fasterrag autopilot generate-golden-set <sources...>`

Generate a golden Q&A set from a corpus with the P4 contract ([prompts.md](prompts.md)) and write it as JSONL. This is the prerequisite for both `autopilot run` and the eval regression gate (D7), so it exists as its own command rather than only as a step inside tuning.

Sources are parsed and chunked exactly as ingestion would — no collection is needed, and none is read. A golden set is usually wanted *before* the index exists, which is the point of tuning against one.

| Flag | Description |
|---|---|
| `--out PATH` | Where to write it; defaults to `./golden.jsonl`. An existing file is never overwritten. |
| `--size N` | How many records to aim for. Defaults to `autopilot.golden_set_size`. |
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
