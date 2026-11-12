# ADR-0009: The Ingestion Journal Stays on Plain Files, Written Properly

- Status: accepted
- Date: 2026-11-12
- Deciders: fasterRag maintainers

## Context and Problem Statement

The ingestion journal (`services/journal.py`) holds job records, per-document outcomes and per-collection content hashes as plain files under `.fasterrag/journal/`. TASK-0437 found four defects in how those files were written:

- A rewrite renamed the live job record to `.prev` *before* writing the new one, so for a moment there was no live record. A reader in that window reported `NOT_FOUND` for a job that existed, and on Windows a reader that opened the file mid-rename got a raw `PermissionError`, which the API served as a `500` (the Windows CI failure of 2026-11-12).
- Nothing called `fsync`, so an "atomic" write was atomic against a process crash but not against power loss.
- Two writers shared one fixed temporary name, and appends from two processes took no lock.
- `find_by_idempotency_key` read every job file (633 ms at 10,000 jobs, BENCH-0004).

The task offered two ways out: move the journal into SQLite in WAL mode behind the existing `Journal` interface, or keep the file layout and fix each defect. Which?

## Decision Drivers

- Operators already have journals on disk. A new backend needs a one-shot migration and a test for it; a fixed file layout needs neither.
- The layout is documented and read by people: [disaster-recovery.md](../disaster-recovery.md) backs it up as files, and an operator clears one collection's dedup history by removing one file.
- Tombstones (TASK-0443) made the dedup file an ordered log whose meaning depends on line order, which a file append preserves by construction.
- `Journal`'s two-faced shape (an `async` method over a `*_blocking` twin, TASK-0399) must hold whatever the storage.
- Windows is a supported platform: it cannot `fsync` a directory, reports `PermissionError` for a file being replaced, and locks byte ranges mandatorily.

## Considered Options

1. **Keep the files and fix them**: write a uniquely named temporary file, `fsync` it, `os.replace` it over the live record, `fsync` the directory where the platform allows; keep the previous good copy without ever removing the live file; an advisory lock around appends; an index for idempotency keys.
2. **SQLite in WAL mode** behind the `Journal` interface, with a migration from the file layout.

## Decision Outcome

**Chosen: option 1**, on the maintainer's decision of 2026-11-12. Every defect TASK-0437 lists has a file-level fix of a few dozen lines, each provable by a test that injects a failure between any two file operations; none needs a migration, and the on-disk layout operators back up does not change.

How each defect is closed (the idempotency lookup lands as a separate commit of the same task and is recorded here when it does):

- **No window without a record.** A rewrite copies the live record to `.prev` (only when it passes its checksum, so a torn record never overwrites the last good copy), then writes the new record to `<name>.<random>.tmp`, `fsync`s it and `os.replace`s it over the live file. The live file is replaced, never renamed away; a reader sees the old record or the new one.
- **Durable.** File contents are `fsync`ed before the rename, and the directory after it on POSIX. Windows cannot open a directory to `fsync` it; NTFS journals the rename's metadata itself, and the code branches on the platform rather than failing there.
- **Busy files on Windows.** Opening and replacing are both retried for up to about 0.4 s on `PermissionError`; a read that still cannot open the record falls back to `.prev`, and with neither it raises `IngestionError` with code `NOT_READY` (503, retryable) — never a raw `OSError`.
- **Full disk.** A rewrite needs room for its new file before the old one is released. When the copy to `.prev` cannot be written for lack of space, the previous copy is deleted to make room and the rewrite proceeds, so a running job can still checkpoint on a disk it filled (failure-modes row 33). The record is then briefly without a fallback copy; it is never without a live one.
- **Unique temporary names** remove the collision two writers had on one fixed `.tmp` name.
- **Appends** take an exclusive advisory lock on a sidecar `<file>.lock` (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows — standard library only). The lock is on a separate file because a Windows byte-range lock is mandatory: locking the log itself would make readers fail. A full or read-only disk raises `IngestionError`, never a raw `OSError` (TASK-0475); a lock held for 30 s raises a retryable `NOT_READY`. A crash mid-append leaves a line without its newline; the next append ends it first and readers skip the fragment with a warning, where before every later read of the log raised a JSON error.

### Consequences

- Good: no migration; the layout, the backup procedure and the tools that read the files are unchanged, apart from `*.lock` sidecars beside the logs, which hold nothing and are safe to delete while no writer runs.
- Good: every guarantee is pinned by a test that injects a crash, a torn write or a full disk between any two file operations (`tests/unit/services/test_journal_crashes.py`), including a Hypothesis stateful test.
- Bad: a rewrite now costs two `fsync`s and an extra small file write (the `.prev` copy) where it cost none. Checkpoints are written once per `ingestion.journal.checkpoint_every` documents, so this is paid per checkpoint, not per document.
- Bad: appends are flushed but not `fsync`ed one by one — that would be an `fsync` per document. They are made durable in a batch instead: writing a checkpoint first `fsync`s the job's document log and the collection's dedup file, so everything a checkpoint covers is on disk before the checkpoint claims it. A power loss can lose log lines written after the last checkpoint; those documents are re-processed on resume, and re-indexing them is idempotent.
- Option 2 stays available. If a single-writer database becomes necessary — many processes writing one journal, or journals too large to scan at all — it can be built behind the same interface, with the migration this decision avoided.
