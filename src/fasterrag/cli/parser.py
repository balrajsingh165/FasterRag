"""The ``fasterrag`` argument parser.

Built with the standard library rather than a CLI framework: the approved stack in
``CLAUDE.md`` does not include one, and argparse covers subcommands, grouped flags, and
generated help without adding a dependency to a package whose install weight is a stated
concern.

The surface mirrors ``docs/cli-reference.md`` exactly. Commands whose service layer has not
shipped yet are absent rather than present-and-broken — a command that accepts its flags and
then fails is worse than one that reports itself unknown, because only the second is
discoverable from ``--help``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any, Final

from fasterrag.config.loader import DEFAULT_CONFIG_PATH
from fasterrag.services.backfill import DEFAULT_BATCH_SIZE
from fasterrag.services.backup import DEFAULT_RETAIN
from fasterrag.services.ci_status import DEFAULT_BRANCH
from fasterrag.services.provision_registry import PROVISIONABLE_TOOLS

__all__ = ["PENDING_COMMANDS", "REMOVED_COMMANDS", "build_parser", "removed_command"]


class _StoreTrueNoDefault(argparse.Action):
    """``store_true`` that leaves the attribute alone when the flag is absent."""

    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs: Any) -> None:
        """Register a zero-argument flag whose default is suppressed."""
        kwargs.pop("nargs", None)
        kwargs.pop("default", None)
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        """Set the destination to ``True``."""
        setattr(namespace, self.dest, True)


# TODO: each entry ships with the task named. Listed here so `fasterrag <name>` explains
# which slice implements it rather than printing a bare "invalid choice".
PENDING_COMMANDS: Final[dict[str, str]] = {}

# CRITICAL: a removed command is refused by name before argparse runs, and is never
# registered. Registering it would list it in `--help` and oblige docs/cli-reference.md to
# carry a heading for a command that does nothing, and argparse would still answer
# `worker --pools cpu` with a bare "unrecognized arguments". Each reason names the open task
# the command would return under, so the message says where to follow it.
REMOVED_COMMANDS: Final[dict[str, str]] = {
    "worker": (
        "ingestion runs inside 'fasterrag serve', and this command only checked the vector "
        "database and waited. A separate worker process is tracked as TASK-0130"
    ),
}


def _add_global_flags(parser: argparse.ArgumentParser, *, root: bool = False) -> None:
    """Attach the flags every command accepts."""
    unset: Any = None if root else argparse.SUPPRESS
    flag: Any = "store_true" if root else _StoreTrueNoDefault

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH) if root else argparse.SUPPRESS,
        help="config file to load",
    )
    parser.add_argument("--collection", default=unset, help="target collection")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[] if root else argparse.SUPPRESS,
        metavar="KEY=VALUE",
        help="override any config.yaml setting, e.g. --set chunking.chunk_size=512 "
        "(repeatable; validated exactly as a file value is)",
    )
    parser.add_argument("--json", dest="as_json", action=flag, help="JSON output")
    parser.add_argument("--quiet", "-q", action=flag, help="errors only")
    parser.add_argument("--verbose", "-v", action=flag, help="debug logging")


def _add_serve(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``serve``."""
    parser = subparsers.add_parser("serve", help="run the API server")
    _add_global_flags(parser)
    parser.add_argument("--reload", action="store_true", help="development auto-reload")
    parser.add_argument("--host", default=None, help="override the configured bind host")
    parser.add_argument("--port", type=int, default=None, help="override the configured port")


def _add_ingest(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``ingest``."""
    parser = subparsers.add_parser("ingest", help="submit sources for ingestion")
    _add_global_flags(parser)
    parser.add_argument("sources", nargs="*", help="paths or URLs to ingest")
    parser.add_argument(
        "--metadata", action="append", default=[], metavar="KEY=VALUE", help="chunk metadata"
    )
    parser.add_argument("--priority-class", default=None, help="tiered-embedding routing class")
    parser.add_argument("--recursive", action="store_true", help="recurse into directories")
    parser.add_argument(
        "--resume",
        default=None,
        metavar="JOB_ID",
        help="run an interrupted job again from its last checkpoint instead of new sources",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="print a progress line as the job's documents settle",
    )
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only")


def _add_query(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``query``."""
    parser = subparsers.add_parser("query", help="run a query")
    _add_global_flags(parser)
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--top-k", type=int, default=None, help="override retrieval.top_k")
    parser.add_argument(
        "--filter", action="append", default=[], metavar="KEY=VALUE", help="metadata filter"
    )
    parser.add_argument("--no-stream", action="store_true", help="wait for the whole answer")
    parser.add_argument("--show-chunks", action="store_true", help="print retrieved chunks")
    parser.add_argument("--show-timings", action="store_true", help="print per-stage latency")


def _add_index(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``index`` and its subcommands."""
    parser = subparsers.add_parser("index", help="manage collections")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)

    listing = actions.add_parser("list", help="collections with vector counts and model")
    _add_global_flags(listing)

    create = actions.add_parser("create", help="create a collection")
    _add_global_flags(create)
    create.add_argument("name", help="collection name")
    create.add_argument("--distance", default=None, help="override the configured distance")
    create.add_argument("--shards", type=int, default=None, help="shard count")
    create.add_argument("--replicas", type=int, default=None, help="replica count")

    delete = actions.add_parser("delete", help="drop a collection")
    _add_global_flags(delete)
    delete.add_argument("name", help="collection name")
    delete.add_argument("--force", action="store_true", help="required if an alias target")

    reembed = actions.add_parser("reembed", help="blue/green re-embed with an eval gate (D2)")
    _add_global_flags(reembed)
    reembed.add_argument("name", help="the served alias to rebuild behind")
    reembed.add_argument("sources", nargs="+", help="paths to ingest into the new build")
    reembed.add_argument("--no-eval-gate", action="store_true", help="swap without the gate (dev)")
    reembed.add_argument("--dataset", default=None, help="eval dataset directory to gate on")
    reembed.add_argument("--recursive", action="store_true", help="recurse into directories")
    reembed.add_argument(
        "--watch",
        action="store_true",
        help="print a progress line as the build's documents settle",
    )

    rollback = actions.add_parser("rollback", help="flip the alias back to a retained build (D2)")
    _add_global_flags(rollback)
    rollback.add_argument("name", help="the served alias")
    rollback.add_argument("--to", default=None, help="a specific retained build to restore")

    tenants = actions.add_parser("tenants", help="count points by tenant payload tag")
    _add_global_flags(tenants)
    tenants.add_argument(
        "name", nargs="?", default=None, help="one collection; every collection if omitted"
    )
    tenants.add_argument(
        "--untagged",
        default=None,
        help="write the untagged points, grouped by document, as JSONL to this path",
    )

    backfill = actions.add_parser(
        "backfill-tenant", help="write the tenant payload tag onto existing points"
    )
    _add_global_flags(backfill)
    backfill.add_argument("name", help="collection, spelled as `index list` prints it")
    backfill.add_argument("--tenant", required=True, help="the tenant id to write")
    backfill.add_argument(
        "--apply", action="store_true", help="write; without it the command only reports"
    )
    backfill.add_argument(
        "--overwrite",
        action="store_true",
        help="relabel points already tagged for another tenant",
    )
    backfill.add_argument(
        "--assume-untagged",
        action="store_true",
        help="in a collection holding another tenant, tag only the untagged points",
    )
    backfill.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="points per update call"
    )

    lock = actions.add_parser("lock", help="index lockfile operations (D1)")
    _add_global_flags(lock)
    lock_actions = lock.add_subparsers(dest="lock_action", required=True)
    verify = lock_actions.add_parser("verify", help="verify a collection against its lockfile")
    _add_global_flags(verify)
    verify.add_argument("name", nargs="?", default=None, help="collection; defaults to configured")


def _add_documents(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``documents delete``."""
    parser = subparsers.add_parser("documents", help="manage the documents in a collection")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)

    delete = actions.add_parser("delete", help="delete one document from a collection")
    _add_global_flags(delete)
    delete.add_argument("document_id", help="the d_… id the ingest job reported")
    delete.add_argument(
        "--tenant",
        default=None,
        help="delete only chunks tagged for this tenant; every chunk of the document if omitted",
    )


def _add_provision(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``provision``."""
    parser = subparsers.add_parser("provision", help="config-driven provisioning")
    _add_global_flags(parser)
    parser.add_argument("tool", choices=PROVISIONABLE_TOOLS, help="tool to provision")
    parser.add_argument("--status", action="store_true", help="report state instead")
    parser.add_argument("--down", action="store_true", help="stop the managed containers")


def _add_estimate(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``estimate``."""
    parser = subparsers.add_parser("estimate", help="preflight cost estimate")
    _add_global_flags(parser)
    parser.add_argument("sources", nargs="+", help="paths or URLs to estimate")
    parser.add_argument("--recursive", action="store_true", help="recurse into directories")
    parser.add_argument("--provider", default=None, help="price against a specific provider")
    parser.add_argument("--all-providers", action="store_true", help="price every known provider")


def _add_replay(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``replay``."""
    parser = subparsers.add_parser("replay", help="re-execute a past query (D8)")
    _add_global_flags(parser)
    parser.add_argument("--trace", required=True, help="the stored trace id to replay")
    parser.add_argument(
        "--candidate",
        default=None,
        metavar="PATH",
        help="candidate config to replay under; defaults to the current one",
    )
    parser.add_argument("--diff-only", action="store_true", help="omit the full answer text")


def _add_traces(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``traces``."""
    parser = subparsers.add_parser("traces", help="inspect stored query traces")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)

    listing = actions.add_parser("list", help="recent trace ids, newest first")
    _add_global_flags(listing)
    listing.add_argument("--limit", type=int, default=20, help="how many to list")

    show = actions.add_parser("show", help="print one stored trace")
    _add_global_flags(show)
    show.add_argument("trace_id", help="the trace id")


def _add_autopilot(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``autopilot run`` and ``autopilot generate-golden-set``."""
    parser = subparsers.add_parser("autopilot", help="eval-driven tuning that only suggests (D6)")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)

    run = actions.add_parser("run", help="search configurations and write a suggested diff")
    _add_global_flags(run)
    run.add_argument("--dataset", default=None, help="eval dataset directory to tune against")
    run.add_argument("--budget-minutes", type=float, default=5.0, help="wall-clock ceiling")
    run.add_argument("--out", default=None, help="where to write the suggestion")

    golden = actions.add_parser(
        "generate-golden-set", help="write a golden Q&A set generated from a corpus (P4)"
    )
    _add_global_flags(golden)
    golden.add_argument("sources", nargs="+", help="corpus files or directories")
    golden.add_argument("--out", default="golden.jsonl", help="where to write the set")
    golden.add_argument(
        "--size",
        type=int,
        default=None,
        help="how many records to aim for; defaults to autopilot.golden_set_size",
    )
    golden.add_argument("--seed", type=int, default=0, help="makes generation reproducible")


def _add_backup(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``backup`` and ``restore``."""
    backup = subparsers.add_parser("backup", help="capture a full deployment backup")
    _add_global_flags(backup)
    backup.add_argument("destination", help="directory to write the backup into")
    backup.add_argument("--collections", nargs="*", default=[], help="limit to these")
    backup.add_argument(
        "--retain",
        type=int,
        default=DEFAULT_RETAIN,
        help="how many backup sets to keep, pruning the oldest and their backend snapshots",
    )

    restore = subparsers.add_parser("restore", help="restore a deployment from a backup")
    _add_global_flags(restore)
    restore.add_argument(
        "source", help="a backup set, or the destination holding several (newest is used)"
    )
    restore.add_argument("--collections", nargs="*", default=[], help="limit to these")
    restore.add_argument("--collections-only", action="store_true", help="skip the control files")


def _add_benchmark(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``benchmark``."""
    parser = subparsers.add_parser("benchmark", help="run the benchmark suites")
    _add_global_flags(parser)
    parser.add_argument(
        "--suite", choices=["ingest", "query", "eval", "all"], default="all", help="which suite"
    )
    parser.add_argument("--dataset", default=None, help="dataset name recorded in the ledger")
    parser.add_argument("--sources", nargs="*", default=[], help="corpus for the ingest suite")
    parser.add_argument("--query", default=None, help="question for the query suite")
    parser.add_argument("--iterations", type=int, default=20, help="calls per repetition")
    parser.add_argument("--ledger", action="store_true", help="emit ledger-formatted entries")


def _add_portability(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``export`` and ``import`` (D11)."""
    export = subparsers.add_parser("export", help="write a collection to a portable archive")
    _add_global_flags(export)
    export.add_argument("--out", required=True, help="archive path; .fragx is appended")
    export.add_argument(
        "--include-vectors",
        action="store_true",
        help="carry the vectors, enabling a vector-copy import into a matching model",
    )

    importer = subparsers.add_parser("import", help="import a portable archive")
    _add_global_flags(importer)
    importer.add_argument("archive", help="the .fragx to import")
    importer.add_argument(
        "--reembed",
        action="store_true",
        help="ignore the archived vectors and embed under the current configuration",
    )
    importer.add_argument("--target-collection", default=None, help="import into this collection")


def _add_config(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``config init``, ``config show``, and ``config validate``."""
    parser = subparsers.add_parser("config", help="configuration tools")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)

    init = actions.add_parser("init", help="write the canonical config.yaml here")
    _add_global_flags(init)
    init.add_argument(
        "--path",
        default=DEFAULT_CONFIG_PATH,
        help="where to write the file; defaults to ./config.yaml",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing file (it is never overwritten without this)",
    )

    validate = actions.add_parser("validate", help="validate config.yaml and referenced env vars")
    _add_global_flags(validate)

    show = actions.add_parser("show", help="list every setting with its value and default")
    _add_global_flags(show)
    show.add_argument(
        "--changed",
        action="store_true",
        help="list only the settings that differ from their default",
    )


def _add_ci_status(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach ``ci-status``, the gate that reads a branch's Actions verdict."""
    ci_status = subparsers.add_parser(
        "ci-status", help="fail when any workflow's last conclusion on a branch is red"
    )
    _add_global_flags(ci_status)
    ci_status.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"the branch whose runs are read (default: {DEFAULT_BRANCH})",
    )
    ci_status.add_argument(
        "--workflow",
        action="append",
        default=[],
        metavar="FILE",
        help="a workflow file to check; repeatable, defaults to every file in .github/workflows",
    )
    ci_status.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="FILE",
        help="a workflow file to leave out; repeatable",
    )
    ci_status.add_argument(
        "--token-env",
        metavar="NAME",
        help="read a bearer token from this environment variable; anonymous when omitted",
    )


def removed_command(argv: Sequence[str]) -> str | None:
    """Return the removed command ``argv`` asks for, or ``None`` when it asks for a live one.

    Only the command position counts, so a collection or a source path may spell a removed
    name: the global flags are parsed away first and the first word left over is the command.
    """
    if not set(REMOVED_COMMANDS) & set(argv):
        return None
    probe = argparse.ArgumentParser(prog="fasterrag", add_help=False)
    _add_global_flags(probe, root=True)
    _, rest = probe.parse_known_args(list(argv))
    return rest[0] if rest and rest[0] in REMOVED_COMMANDS else None


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the whole CLI surface."""
    parser = argparse.ArgumentParser(
        prog="fasterrag",
        description="Backend-only RAG framework. See docs/cli-reference.md.",
    )
    _add_global_flags(parser, root=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_serve(subparsers)
    _add_ingest(subparsers)
    _add_query(subparsers)
    _add_index(subparsers)
    _add_documents(subparsers)
    _add_provision(subparsers)

    status = subparsers.add_parser("status", help="one-screen system status")
    _add_global_flags(status)

    doctor = subparsers.add_parser("doctor", help="preflight diagnostics (D10)")
    _add_global_flags(doctor)
    doctor.add_argument(
        "--fix",
        action="store_true",
        help="apply the repairs that are safe to apply, then run every check again",
    )

    _add_ci_status(subparsers)

    _add_estimate(subparsers)
    _add_autopilot(subparsers)
    _add_backup(subparsers)
    _add_benchmark(subparsers)
    _add_replay(subparsers)
    _add_traces(subparsers)
    _add_portability(subparsers)
    _add_config(subparsers)

    for name, task in PENDING_COMMANDS.items():
        pending = subparsers.add_parser(name, help=f"not implemented yet — ships with {task}")
        _add_global_flags(pending)
        pending.add_argument("rest", nargs="*", help=argparse.SUPPRESS)

    return parser
