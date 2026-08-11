"""The commands that stand things up or price them out: ``provision`` and ``estimate``.

``provision`` is doctor-gated by the service layer, not here — putting the gate in the CLI
would let the REST path around it, and a gate one caller can skip is not a gate. Which tools
it accepts is decided there too, by ``services/provision_registry``: the hand-written
``if args.tool == ...`` chain this replaced was one half of the pair of lists that let REST
and CLI disagree about Langfuse and Grafana for nine days (TASK-0251).

``estimate`` runs *before* ingestion for a reason: embedding cost is decided by chunk count
and overlap, not by file size, so the only honest way to price a corpus is to chunk it. The
estimator does exactly that and never embeds.
"""

from __future__ import annotations

import argparse

from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.settings import settings_from
from fasterrag.cli.sources import expand_sources
from fasterrag.errors import ConfigError, FasterRagError, ProvisioningError
from fasterrag.services.estimation import estimate_sources, require_estimator
from fasterrag.services.provision_registry import provision_tool, stop_tool, tool_status

__all__ = ["run_estimate", "run_provision"]


async def run_provision(args: argparse.Namespace, console: Console) -> ExitCode:
    """Provision, stop, or report on a managed dependency.

    ``--status`` and ``--down`` are mutually exclusive in effect; ``--status`` wins, because
    asking for state and asking to change it in one command is a mistake worth refusing to
    guess about.
    """
    try:
        settings = settings_from(args)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    if args.status and args.down:
        console.error("--status and --down cannot be combined")
        return ExitCode.USAGE

    verb = tool_status if args.status else stop_tool if args.down else provision_tool

    try:
        result = await verb(args.tool, settings)
    except ProvisioningError as exc:
        console.problem(exc.code.value, exc.detail, exc.fix)
        console.document({"tool": args.tool, "status": "failed", "detail": exc.detail})
        return ExitCode.PREFLIGHT
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.FAILURE

    console.emit(f"{result.tool}: {result.status}")
    if result.url:
        console.emit(f"  url: {result.url}")
    if result.detail:
        console.emit(f"  {result.detail}")
    console.document(
        {
            "tool": result.tool,
            "status": result.status,
            "url": result.url,
            "detail": result.detail,
        }
    )
    return ExitCode.SUCCESS


async def run_estimate(args: argparse.Namespace, console: Console) -> ExitCode:
    """Report what ingesting a set of sources would cost, before embedding any of it.

    Refuses with a usage exit when ``cost.estimator`` is false, rather than estimating
    anyway: the setting reads as a control over this command, so honouring it here is what
    makes it one.

    Directory arguments are expanded exactly as ``ingest`` expands them. An estimate is only
    a preflight if it prices the same file set the ingestion it precedes would read; taking
    a directory verbatim priced it as one unreadable document, which quoted every corpus at
    zero. The expansion happens after the refusal, because a disabled command has no business
    walking the filesystem.
    """
    try:
        settings = settings_from(args)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    try:
        require_estimator(settings)
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    sources = expand_sources(args.sources, recursive=getattr(args, "recursive", False))

    try:
        estimate = estimate_sources(
            sources, settings, all_providers=bool(args.all_providers or args.provider)
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.FAILURE

    console.emit(f"documents       {estimate.documents} ({estimate.unreadable} unreadable)")
    console.emit(f"chunks          {estimate.chunks}")
    console.emit(f"tokens          {estimate.tokens}")
    console.emit(f"parsed in       {estimate.parse_seconds:.2f}s")

    providers = estimate.providers
    if args.provider:
        providers = [item for item in providers if item.provider == args.provider]
        if not providers:
            console.error(f"no price is recorded for provider {args.provider!r}")

    for item in providers:
        cost = f"${item.cost_usd:.4f}" if item.cost_usd is not None else "unknown"
        console.emit(f"  {item.provider}/{item.model}: {cost}")

    if estimate.enrichment is not None:
        enrichment = estimate.enrichment
        cost = f"${enrichment.cost_usd:.4f}" if enrichment.cost_usd is not None else "unknown"
        console.emit("")
        console.emit(f"enrichment      {enrichment.calls} call(s) to {enrichment.model}")
        console.emit(
            f"  prompt        {enrichment.prompt_tokens} tokens "
            f"(the parent document, once per chunk)"
        )
        console.emit(f"  completion    {enrichment.completion_tokens} tokens")
        console.emit(f"  cost          {cost}")
        console.emit(f"  basis         {enrichment.basis}")

    if estimate.projection_note:
        console.emit("")
        console.emit(f"note            {estimate.projection_note}")

    console.document(estimate.as_dict())
    return ExitCode.SUCCESS
