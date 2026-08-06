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

### Sets, retention, and cadence

Each `fasterrag backup <destination>` writes its own timestamped **set** under the destination, so runs accumulate a history instead of each overwriting the last. `--retain N` keeps the newest N sets (default **14**) and deletes the older ones **together with the backend snapshots they reference** — pruning only the directory would leave the snapshot inside the vector database, referenced by no manifest and consuming disk until somebody found it by hand.

Retention counts **sets, not days**. fasterRag runs when it is invoked and cannot know what cadence a scheduler was configured with, so counting days would mean guessing at a number only the operator holds. At the daily cadence below, 14 sets is 14 days.

**Cadence is not fasterRag's job.** cron, systemd timers, and Task Scheduler already run things on a schedule, do it better, and are what an operator already monitors — a scheduler built into a RAG framework would be a worse one that fails silently. Schedule it yourself:

```cron
# Daily at 02:30, keeping two weeks of sets.
30 2 * * *  cd /srv/fasterrag && fasterrag backup /backups/fasterrag --retain 14 --json >> /var/log/fasterrag-backup.log 2>&1
```

Restore takes either one set or the destination holding several, in which case the newest is used — during an incident the path an operator has to hand is the one they backed up to, and making them list subdirectories first is an avoidable step at the worst possible moment.

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

### Drill execution log

| Date | Steps executed | Result |
|---|---|---|
| 2026-08-01 | 4 (restore vector DB, verify counts), 5 (control files), 7 (behavior verification) | **Passed, partial scope.** A live `apitest` collection was snapshotted, deleted outright, and restored from the backup. The restore reported matching vector counts, and the same query returned the identical answer citing the identical chunk id `c_ca8722f2b113fc1d` — the collection configuration (384 dimensions, cosine, sparse leg) survived because the snapshot is backend-native rather than a re-export of points. Restore wall clock: 4,311 ms for a 2-vector collection, including process startup. |

**What this drill did *not* execute.** Steps 1–3 (clean host, `.env` recreation, `fasterrag doctor`) and step 6's `index lock verify` were not performed: the restore was onto the *same* running host, which makes this the single-collection shortcut of §4 rather than the full clean-host procedure of §2. Step 6 additionally reported `no lockfile` for the restored collection — correctly, since that collection was ingested before lockfile writing was wired in, so there was nothing to back up and nothing to verify against.

**RPO and RTO remain TBD-until-measured.** The 4,311 ms above is a restore *duration* for one tiny collection on a laptop, not an RTO: RTO is `T1 − T0` from a clean host, and no clean host was used. Publishing it as RTO would be exactly the substitution the provable-claims policy exists to prevent.

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
| **Index lost, journal intact** | **Re-ingest is NOT a recovery path.** Clear the journal for that collection first, or restore its snapshot. |
| Trace store lost | Accept loss (bounded by `traces.retention_days`); no impact on serving |
| Langfuse stack lost | Re-provision (`observability.langfuse: true` path); traces resume; historic Langfuse data restores from its own volumes if backed up |

### The re-ingest trap

Dropping a collection and re-running the same ingest **does not rebuild it.** Content hashes are remembered in the journal per collection, not per index, so every document is recognised as already-ingested, the job settles `completed` with `indexed: 0`, and the collection is never recreated. Nothing errors — the counts are the only signal, and `deduplicated` is where the documents went.

That is D3 behaving exactly as designed: deduplication is what makes an at-least-once pipeline produce exactly-once effects, and it cannot distinguish "this document is already indexed" from "this document was indexed into an index that no longer exists". It is still a sharp edge during recovery, when re-ingest is the instinctive thing to reach for.

**Restore from a snapshot** ([§2](#2-restore-drill-write-once-execute-for-real-tick-in-todomd)) is the supported path. If you must rebuild from source instead, clear the journal entries for that collection first so the documents are seen as new.

Observed on 2026-08-03 while verifying the library facade: a run whose worker pool died mid-ingest recorded its documents in the journal before failing, and the retry reported `completed` with `indexed: 0` — correct, and initially indistinguishable from a retrieval bug.
