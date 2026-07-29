# disaster-recovery.md — Backups, Restore Drill, RPO/RTO

> **RPO and RTO are TBD-until-measured**: both values are set only after the restore drill below has actually been executed on reference hardware and its timings recorded in the [benchmark ledger](benchmarks.md). The drill is a task in [todo.md](todo.md) and must be ticked with a date — an unexecuted restore procedure is assumed broken.

## 1. What is backed up

| Artifact | Contents | Why it matters | Method |
|---|---|---|---|
| **Vector collections** | dense vectors, payloads/metadata, collection config | The index itself; hours-to-days of embedding compute | Backend-native snapshots via the adapter (Qdrant snapshot API in the reference implementation), per collection |
| **Index manifest + lockfile** (`index.lock`) | config hash, embedding model+version, chunker strategy+version, per-doc content hashes, alias state | Proves what the index *is*; enables drift detection and reproducible rebuild (D1) | File copy (atomic-write source) |
| **`config.yaml`** | all system behavior | Restores exact behavior; contains no secrets so it is safely versioned | Git (it is always committed) + backup copy |
| **Ingestion journal** | job records, checkpoints, DLQ entries, content hashes | Resumability and exactly-once guarantees (D3) | File/DB snapshot |
| **Trace store** | full query traces (retrieved chunks, scores, prompts, responses) | Replay/debugging history (D8); audit trail | File/DB snapshot |
| *(not backed up)* `.env` | secrets | Deliberately excluded from automated backups; operators store secrets in their secret manager. Langfuse's `SALT`/`ENCRYPTION_KEY`/`NEXTAUTH_SECRET` MUST be preserved by the operator — losing them invalidates Langfuse credentials ([observability.md](observability.md)) | operator-owned |

Backup cadence, destination, and retention are operator decisions; the tooling ships with a documented default of daily snapshots retained 14 days.

## 2. Restore drill (write once, EXECUTE for real, tick in todo.md)

The drill restores a complete deployment onto a clean machine from backups only. It must be executed — not desk-checked — before any RPO/RTO is published, and re-executed at every major release.

1. **Provision clean host**: OS + Docker only. Record start time `T0`.
2. **Restore config**: place backed-up `config.yaml`; recreate `.env` from the operator secret store.
3. **Preflight**: `fasterrag doctor` — must pass (fix-it hints resolve environment gaps).
4. **Restore vector DB**: `fasterrag provision qdrant` (or start external instance), then restore collection snapshots via the adapter; verify counts per collection.
5. **Restore control files**: index manifest + `index.lock`, ingestion journal, trace store into their configured paths.
6. **Verify integrity**: `fasterrag index lock verify` — zero drift expected against restored config; journal loads at last checkpoint; `fasterrag status` all green.
7. **Verify behavior**: run the eval harness smoke set — metrics within regression-gate tolerance of pre-disaster ledger values; one known query replayed via `fasterrag replay` matches its stored trace.
8. **Record `T1`** (queries serving) and `T2` (full verification done). RTO = `T1 − T0`; verify data loss window vs snapshot timestamps = achieved RPO.
9. **Record results**: ledger entry (timings, hardware) + tick the drill task in [todo.md](todo.md) with the date. File a bug for every step that needed improvisation — improvisation in a drill is a documentation defect.

## 3. Recovery objectives

| Objective | Definition | Value |
|---|---|---|
| **RPO** (Recovery Point Objective) | Maximum data-loss window: time between last usable backup set and the failure | **TBD-until-measured** (bounded above by backup cadence; proven by the drill) |
| **RTO** (Recovery Time Objective) | `T1 − T0` from the executed drill: clean host → queries serving | **TBD-until-measured** (proven by the drill) |

## 4. Partial-loss shortcuts (cheaper than full restore)

| Loss | Shortcut |
|---|---|
| One collection corrupted | Restore that collection's snapshot; alias flip if a blue/green sibling exists (D2) |
| Index drifted / model mismatch | No restore needed: D2 zero-downtime reindex rebuilds from source documents |
| Journal lost, index intact | Re-run ingest of the source set — dedup makes it a no-op except genuinely missing docs (D3) |
| Trace store lost | Accept loss (bounded by `traces.retention_days`); no impact on serving |
| Langfuse stack lost | Re-provision (`observability.langfuse: true` path); traces resume; historic Langfuse data restores from its own volumes if backed up |
