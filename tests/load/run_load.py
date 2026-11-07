"""The operator's entry point: validate, probe, generate, audit, report.

    python -m tests.load.run_load --target http://rig:8000 --scenario mixed \
        --users 64 --duration 900 --dataset tests/eval/datasets/handbook \
        --sut-hardware "AMD 7950X, 64 GB, NVMe, Ubuntu 24.04" --isolated

The order of the phases is the whole design, so it is stated rather than inferred:

1. **Parse and validate.** The dataset is loaded here, before anything is started, because
   a query set with no queries and a corpus with no documents are both runs that would
   otherwise finish successfully having sent nothing.
2. **Preflight.** The target is proven reachable, ready, fasterRag, and able to answer this
   scenario's exact requests. This happens *before* the Locust check on purpose: a dead
   target invalidates the run whoever is running it and whatever is installed, and the
   probe costs about a second.
3. **Tool check.** Locust is imported from the active interpreter. A contributor without it
   gets the install command, not a stack trace.
4. **Cold repetition**, discarded from the headline and reported separately, because an
   unprimed cache and an unloaded model belong to cold start (docs/performance.md §3 rule
   3).
5. **Warmed repetitions**, three by default, median reported, every repetition attached
   (rule 5). Each is bracketed by a ``/metrics`` scrape so counter-derived SLIs describe
   that repetition rather than the process's whole lifetime.
6. **Audit**, then report. The exit code is the audit's, so a caller reading ``$?`` and
   nothing else still cannot mistake an empty run for a fast one.

**This command never says a system is fast or passes.** Every SLO target in
``docs/slo.md`` is TBD-until-measured, so there is nothing to grade against; the only
question it answers is whether the run produced a valid sample. ``--json`` writes the
machine-readable report, and ``--ledger-draft`` prints either a benchmark-ledger entry or
an itemised refusal explaining what would have to be true for one to exist.

The API key is read from ``FASTERRAG_LOAD_API_KEY`` and never accepted as a flag: a
command line is visible in shell history and in every process listing on the box.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urljoin

from fasterrag.services.benchmark import commit_hash, fingerprint
from tests.load.harness import (
    API_KEY_VARIABLE,
    ENVIRONMENT_PREFIX,
    MINIMUM_SAMPLES,
    REPETITIONS,
    SCENARIOS,
    ExitCode,
    LoadConfig,
    LoadHarnessError,
    LoadReport,
    Problem,
    RunStats,
    Scenario,
    Series,
    audit,
    derive_slis,
    exit_code_for,
    median_slis,
    parse_exposition,
    parse_locust_stats,
)
from tests.load.preflight import fetch, preflight

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
LOCUSTFILE: Final = Path(__file__).resolve().parent / "locustfile.py"
INSTALL_HINT: Final = "pip install -r tests/load/requirements.txt"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, every flag of which changes what a number means."""
    parser = argparse.ArgumentParser(
        prog="python -m tests.load.run_load",
        description="Run the fasterRag load scenarios and report the docs/slo.md §1 SLIs.",
    )
    parser.add_argument("--target", required=True, help="base URL of the fasterRag instance")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--users", type=int, required=True, help="concurrent users")
    parser.add_argument("--spawn-rate", type=float, default=0.0, help="users spawned per second")
    parser.add_argument("--duration", type=int, required=True, help="seconds per repetition")
    parser.add_argument("--dataset", type=Path, required=True, help="eval dataset directory")
    parser.add_argument(
        "--sut-hardware",
        required=True,
        help="exact spec of the machine under test; it cannot be detected from here",
    )
    parser.add_argument("--warmup", type=int, default=60, help="seconds of discarded cold run")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--stream-weight", type=int, default=1)
    parser.add_argument("--processes", type=int, default=1, help="Locust worker processes")
    parser.add_argument("--min-samples", type=int, default=MINIMUM_SAMPLES)
    parser.add_argument("--max-failure-ratio", type=float, default=0.01)
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="assert the system under test carried no co-tenant load; without it the run "
        "can never become a ledger entry",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--ledger-draft", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> LoadConfig:
    """Turn parsed arguments into the run's configuration."""
    scenario: Scenario = args.scenario
    return LoadConfig(
        target=str(args.target).rstrip("/"),
        scenario=scenario,
        users=int(args.users),
        spawn_rate=float(args.spawn_rate) or float(args.users),
        duration_s=int(args.duration),
        dataset=Path(args.dataset),
        sut_hardware=str(args.sut_hardware),
        warmup_s=int(args.warmup),
        repetitions=int(args.repetitions),
        collection=args.collection,
        tenant=args.tenant,
        stream_weight=int(args.stream_weight),
        isolated=bool(args.isolated),
        min_samples=int(args.min_samples),
        max_failure_ratio=float(args.max_failure_ratio),
    )


def require_locust() -> None:
    """Refuse the run when the load generator is not installed, naming the fix."""
    if importlib.util.find_spec("locust") is None:
        raise LoadHarnessError(
            "locust is not installed in this interpreter. It is a load-only dependency "
            f"and `.[dev]` deliberately does not carry it: {INSTALL_HINT}",
            ExitCode.TOOL_MISSING,
        )


def scrape(config: LoadConfig, api_key: str | None) -> tuple[Series, ...]:
    """Read the target's metrics catalogue, which brackets every repetition."""
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = fetch(urljoin(f"{config.target}/", "/metrics"), headers=headers)
    if response.status != 200:
        raise LoadHarnessError(
            f"the metrics scrape returned {response.status} mid-run, so the counter-derived "
            "SLIs for this repetition cannot be computed",
            ExitCode.PREFLIGHT_FAILED,
        )
    return parse_exposition(response.body.decode("utf-8", "replace"))


def locust_environment(config: LoadConfig, duration_s: int) -> dict[str, str]:
    """Return the environment the Locust process is configured through."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.update(
        {
            f"{ENVIRONMENT_PREFIX}TARGET": config.target,
            f"{ENVIRONMENT_PREFIX}SCENARIO": config.scenario,
            f"{ENVIRONMENT_PREFIX}USERS": str(config.users),
            f"{ENVIRONMENT_PREFIX}SPAWN_RATE": str(config.spawn_rate),
            f"{ENVIRONMENT_PREFIX}DURATION": str(duration_s),
            f"{ENVIRONMENT_PREFIX}DATASET": str(config.dataset),
            f"{ENVIRONMENT_PREFIX}SUT_HARDWARE": config.sut_hardware,
            f"{ENVIRONMENT_PREFIX}STREAM_WEIGHT": str(config.stream_weight),
            f"{ENVIRONMENT_PREFIX}MIN_SAMPLES": str(config.min_samples),
            f"{ENVIRONMENT_PREFIX}MAX_FAILURE_RATIO": str(config.max_failure_ratio),
        }
    )
    if config.collection:
        environment[f"{ENVIRONMENT_PREFIX}COLLECTION"] = config.collection
    if config.tenant:
        environment[f"{ENVIRONMENT_PREFIX}TENANT"] = config.tenant
    return environment


def _user_classes(scenario: Scenario) -> tuple[str, ...]:
    """Return the Locust user classes this scenario drives."""
    if scenario == "query":
        return ("QueryUser",)
    if scenario == "ingest":
        return ("IngestUser",)
    return ("QueryUser", "IngestUser")


def _stats_path(stem: Path) -> Path:
    """Return where Locust actually wrote its JSON, which may gain a .json suffix."""
    for candidate in (stem.with_name(f"{stem.name}.json"), stem):
        if candidate.is_file():
            return candidate
    raise LoadHarnessError(
        f"locust wrote no statistics to {stem} or {stem}.json, so the run produced nothing "
        "to audit",
        ExitCode.VACUOUS,
    )


def run_once(
    config: LoadConfig, label: str, duration_s: int, processes: int, api_key: str | None
) -> RunStats:
    """Run one Locust repetition, bracketed by metrics scrapes, and parse its statistics."""
    with tempfile.TemporaryDirectory(prefix="fasterrag-load-") as directory:
        stem = Path(directory) / "stats"
        command = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(LOCUSTFILE),
            "--headless",
            "--only-summary",
            "--host",
            config.target,
            "--users",
            str(config.users),
            "--spawn-rate",
            str(config.spawn_rate),
            "--run-time",
            f"{duration_s}s",
            "--json-file",
            str(stem),
            *(("--processes", str(processes)) if processes > 1 else ()),
            *_user_classes(config.scenario),
        ]

        before = scrape(config, api_key)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=locust_environment(config, duration_s),
            check=False,
        )
        elapsed = time.perf_counter() - started
        after = scrape(config, api_key)

        if completed.returncode == int(ExitCode.PREFLIGHT_FAILED):
            raise LoadHarnessError(
                f"{label}: locust's own preflight refused the target mid-run",
                ExitCode.PREFLIGHT_FAILED,
            )

        payload = json.loads(_stats_path(stem).read_text(encoding="utf-8"))

    return RunStats(
        label=label,
        duration_s=elapsed,
        endpoints=parse_locust_stats(payload),
        metrics_before=before,
        metrics_after=after,
    )


def execute(config: LoadConfig, processes: int) -> LoadReport:
    """Run the whole procedure and return its report, measured or refused."""
    api_key = os.environ.get(API_KEY_VARIABLE, "").strip() or None
    preflight(config, api_key)
    require_locust()

    cold = (
        run_once(config, "cold", config.warmup_s, processes, api_key)
        if config.warmup_s > 0
        else None
    )
    runs = tuple(
        run_once(config, f"warmed-{index + 1}", config.duration_s, processes, api_key)
        for index in range(max(config.repetitions, 1))
    )

    problems: list[Problem] = []
    per_run = []
    for run in runs:
        problems.extend(audit(run, config))
        per_run.append(derive_slis(run))

    return LoadReport(
        config=config,
        generator=fingerprint().describe(),
        started_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        commit=commit_hash(),
        cold=cold,
        runs=runs,
        slis=median_slis(per_run),
        problems=tuple(problems),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness and return the exit code the run earned."""
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        report = execute(config, int(args.processes))
    except LoadHarnessError as exc:
        sys.stderr.write(f"load run refused [{exc.exit_code.name}]: {exc.detail}\n")
        return int(exc.exit_code)

    sys.stdout.write(f"{report.render()}\n")
    if args.ledger_draft:
        sys.stdout.write(f"\n{report.ledger_draft()}\n")
    if args.json:
        args.json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    return int(exit_code_for(report.problems))


if __name__ == "__main__":
    raise SystemExit(main())
