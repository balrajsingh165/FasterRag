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

__all__ = ["PENDING_COMMANDS", "build_parser"]


class _StoreTrueNoDefault(argparse.Action):
    """``store_true`` that leaves the attribute alone when the flag is absent.

    ``argparse.SUPPRESS`` suppresses the default but ``store_true`` insists on one, so a
    subparser's ``--json`` would reset a root-level ``--json`` back to ``False``. This sets
    the attribute only when the flag actually appears.
    """

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
PENDING_COMMANDS: Final[dict[str, str]] = {
    "export": "TASK-0079 (portability archives, D11)",
    "import": "TASK-0079 (portability archives, D11)",
}


def _add_global_flags(parser: argparse.ArgumentParser, *, root: bool = False) -> None:
    """Attach the flags every command accepts.

    Repeated on each subparser as well as the root so ``fasterrag --json query ...`` and
    ``fasterrag query --json ...`` both work; operators write them in either position and
    neither reading is wrong.

    # CRITICAL: only the root carries real defaults. A subparser that also defaulted would
    # overwrite whatever the root parsed — argparse applies subparser defaults after the
    # root's values, so `fasterrag --json doctor` would silently come back with json off.
    """
    unset: Any = None if root else argparse.SUPPRESS
    flag: Any = "store_true" if root else _StoreTrueNoDefault

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH) if root else argparse.SUPPRESS,
        help="config file to load",
    )
    parser.add_argument("--collection", default=unset, help="target collection")
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


def _add_worker(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``worker``."""
    parser = subparsers.add_parser("worker", help="run the pipeline worker pools")
    _add_global_flags(parser)
    parser.add_argument("--pools", default="cpu,embed,index", help="pools this process runs")
    parser.add_argument("--cpu-workers", type=int, default=None, help="override workers.cpu")
    parser.add_argument("--embed-workers", type=int, default=None, help="override workers.embed")


def _add_ingest(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``ingest``."""
    parser = subparsers.add_parser("ingest", help="submit sources for ingestion")
    _add_global_flags(parser)
    parser.add_argument("sources", nargs="+", help="paths or URLs to ingest")
    parser.add_argument(
        "--metadata", action="append", default=[], metavar="KEY=VALUE", help="chunk metadata"
    )
    parser.add_argument("--priority-class", default=None, help="tiered-embedding routing class")
    parser.add_argument("--recursive", action="store_true", help="recurse into directories")
    parser.add_argument("--watch", action="store_true", help="print progress until completion")
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
    reembed.add_argument("--watch", action="store_true", help="print progress until completion")

    rollback = actions.add_parser("rollback", help="flip the alias back to a retained build (D2)")
    _add_global_flags(rollback)
    rollback.add_argument("name", help="the served alias")
    rollback.add_argument("--to", default=None, help="a specific retained build to restore")

    lock = actions.add_parser("lock", help="index lockfile operations (D1)")
    _add_global_flags(lock)
    lock_actions = lock.add_subparsers(dest="lock_action", required=True)
    verify = lock_actions.add_parser("verify", help="verify a collection against its lockfile")
    _add_global_flags(verify)
    verify.add_argument("name", nargs="?", default=None, help="collection; defaults to configured")


def _add_provision(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``provision``."""
    parser = subparsers.add_parser("provision", help="config-driven provisioning")
    _add_global_flags(parser)
    parser.add_argument("tool", choices=["qdrant"], help="tool to provision")
    parser.add_argument("--status", action="store_true", help="report state instead")
    parser.add_argument("--down", action="store_true", help="stop the managed containers")


def _add_estimate(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``estimate``."""
    parser = subparsers.add_parser("estimate", help="preflight cost estimate")
    _add_global_flags(parser)
    parser.add_argument("sources", nargs="+", help="paths or URLs to estimate")
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
    """Register ``autopilot run``."""
    parser = subparsers.add_parser("autopilot", help="eval-driven tuning that only suggests (D6)")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)

    run = actions.add_parser("run", help="search configurations and write a suggested diff")
    _add_global_flags(run)
    run.add_argument("--dataset", default=None, help="eval dataset directory to tune against")
    run.add_argument("--budget-minutes", type=float, default=5.0, help="wall-clock ceiling")
    run.add_argument("--out", default=None, help="where to write the suggestion")


def _add_backup(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``backup`` and ``restore``."""
    backup = subparsers.add_parser("backup", help="capture a full deployment backup")
    _add_global_flags(backup)
    backup.add_argument("destination", help="directory to write the backup into")
    backup.add_argument("--collections", nargs="*", default=[], help="limit to these")

    restore = subparsers.add_parser("restore", help="restore a deployment from a backup")
    _add_global_flags(restore)
    restore.add_argument("source", help="a backup directory holding a manifest")
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


def _add_config(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``config validate``."""
    parser = subparsers.add_parser("config", help="configuration tools")
    _add_global_flags(parser)
    actions = parser.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate", help="validate config.yaml and referenced env vars")
    _add_global_flags(validate)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the whole CLI surface."""
    parser = argparse.ArgumentParser(
        prog="fasterrag",
        description="Backend-only RAG framework. See docs/cli-reference.md.",
    )
    _add_global_flags(parser, root=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_serve(subparsers)
    _add_worker(subparsers)
    _add_ingest(subparsers)
    _add_query(subparsers)
    _add_index(subparsers)
    _add_provision(subparsers)

    status = subparsers.add_parser("status", help="one-screen system status")
    _add_global_flags(status)

    doctor = subparsers.add_parser("doctor", help="preflight diagnostics (D10)")
    _add_global_flags(doctor)
    doctor.add_argument("--fix", action="store_true", help="apply safe automatic fixes")

    _add_estimate(subparsers)
    _add_autopilot(subparsers)
    _add_backup(subparsers)
    _add_benchmark(subparsers)
    _add_replay(subparsers)
    _add_traces(subparsers)
    _add_config(subparsers)

    for name, task in PENDING_COMMANDS.items():
        pending = subparsers.add_parser(name, help=f"not implemented yet — ships with {task}")
        _add_global_flags(pending)
        pending.add_argument("rest", nargs="*", help=argparse.SUPPRESS)

    return parser
