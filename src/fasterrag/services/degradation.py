"""The degradation ladder's registry: which rungs exist, and what serves each one.

D4 promises that component loss produces an explicitly labelled degraded answer instead of
a silent quality drop. The label travels on the wire — every query response carries ``mode``
and ``degraded`` (``docs/api-reference.md``) — so the set of rungs is a public contract, and
the documentation states which of them are built.

Until this module nothing could decide that statement. ``hybrid_only`` was a bare constant in
``services.querying``, ``extractive`` one in ``services.generation``, and ``cache_only`` — the
rung D4 once specified, deferred by ``docs/adr/ADR-0011-degradation-ladder-scope.md`` —
appeared nowhere in ``src`` at all. The only way to ask the tree whether a rung existed was
to grep for its name, which answers *yes* off a docstring, and a wrong *yes* fails an honest
sentence (TASK-0369).

``_BUILTIN_RUNGS`` replaces the grep, and is modelled on ``adapters/*/factory.py``: a mapping
from the wire value to the implementation that serves it, with ``_PLANNED_RUNGS`` naming the
rungs the documentation mentions and the code does not serve (today only the deferred
``cache_only``). ``scripts/check_unbuilt_claims.py`` reads both with ``ast``, the same way it
reads ``_BUILTIN_ADAPTERS`` (TASK-0394).

A registry you can forget to update is not a registry, so two things hold it to the tree:

* **the ladder dispatches through it.** ``serving()`` is the only producer of a degraded
  ``mode``, it imports the registered implementation before handing the value back, and it
  refuses a rung that is deferred or is not a rung at all. Shipping ``cache_only`` without
  moving it out of ``_PLANNED_RUNGS`` raises at the moment it tries to label its first
  response, so the registry cannot lag the code;
* **the enum is checked against it at import.** A rung named in ``DegradationMode`` and
  registered nowhere makes this module fail to import, so the registry cannot lag the
  vocabulary either.

The fallback *bodies* stay in the services that own them — the fused shortlist is a list of
chunks, an extractive answer is a string, a cached answer is neither — because no single
signature spans them and inventing one would restructure the retrieval path for a registry.

**The wire form is unchanged, deliberately.** ``serving()`` returns ``str``, the three module
constants are still plain strings, and ``mode`` is still typed ``str`` wherever it is carried
or serialized. ``DegradationMode`` is the internal vocabulary the registry is keyed on and
checked against; it never reaches a response body, a metric label, or the semantic cache, so
no stored or published ``mode`` changes shape. ``FULL`` is in the enum because a response must
be labelled when nothing degraded, and out of the registry because it is the top of the ladder
rather than a rung: it has no fallback to register, and no document denies it.
"""

from __future__ import annotations

from enum import StrEnum
from importlib import import_module
from typing import Final

from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["DegradationMode", "serving"]


class DegradationMode(StrEnum):
    """Every ``mode`` a response may report: the undegraded top, then one per rung."""

    FULL = "full"
    HYBRID_ONLY = "hybrid_only"
    CACHE_ONLY = "cache_only"
    EXTRACTIVE = "extractive"


_BUILTIN_RUNGS: Final[dict[str, str]] = {
    "hybrid_only": "fasterrag.services.querying:fused_shortlist",
    "extractive": "fasterrag.services.generation:extractive_answer",
}

# CRITICAL: ADR-0011 narrowed D4 to the two rungs above and deferred `cache_only`. The name
# stays here rather than being deleted, because this set is what lets
# scripts/check_unbuilt_claims.py decide every documented sentence that says the rung is
# not built. If it ever ships, move it into _BUILTIN_RUNGS in the commit that ships it.
_PLANNED_RUNGS: Final[frozenset[str]] = frozenset({"cache_only"})

# CRITICAL: checked at import, so a rung added to DegradationMode and to nothing else cannot
# be imported, let alone served. Without it the enum and the registry drift apart and the gate
# decides rung denials against the stale half — making every rung look shipped, which is the
# exact defect the registry exists to catch.
_UNREGISTERED: Final = sorted(
    mode.value
    for mode in DegradationMode
    if mode is not DegradationMode.FULL and mode.value not in set(_BUILTIN_RUNGS) | _PLANNED_RUNGS
)
if _UNREGISTERED:
    raise FasterRagError(
        f"degradation rungs {', '.join(_UNREGISTERED)} are named in DegradationMode and "
        "registered nowhere; add each to _BUILTIN_RUNGS when it ships, or to _PLANNED_RUNGS "
        "while it is specified and unbuilt",
        code=ErrorCode.INTERNAL,
        retryable=False,
    )


def _refusal(mode: DegradationMode) -> str:
    """Explain why a rung cannot be served: deferred and unbuilt, or not a rung at all."""
    if mode.value in _PLANNED_RUNGS:
        return (
            f"the {mode.value!r} rung was deferred by ADR-0011 and is not built; "
            "register its implementation in _BUILTIN_RUNGS in the commit that ships it, "
            "because the documentation's denial is read off that registry"
        )
    return (
        f"{mode.value!r} is not a rung on the degradation ladder; the rungs are "
        f"{', '.join(sorted(set(_BUILTIN_RUNGS) | _PLANNED_RUNGS))}"
    )


def _resolve(implementation: str, mode: DegradationMode) -> None:
    """Import a rung's registered implementation, proving the reference still resolves."""
    module_name, _, attribute = implementation.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise FasterRagError(
            f"the {mode.value!r} rung registers {implementation!r}, which does not import",
            code=ErrorCode.INTERNAL,
            retryable=False,
        ) from exc
    if not callable(getattr(module, attribute, None)):
        raise FasterRagError(
            f"the {mode.value!r} rung registers {implementation!r}, which is not a callable; "
            "a registered rung must point at the code that serves it",
            code=ErrorCode.INTERNAL,
            retryable=False,
        )


def serving(mode: DegradationMode) -> str:
    """Return the wire ``mode`` for a rung, once a registered implementation serves it."""
    implementation = _BUILTIN_RUNGS.get(mode.value)
    if implementation is None:
        raise FasterRagError(_refusal(mode), code=ErrorCode.INTERNAL, retryable=False)
    _resolve(implementation, mode)
    return mode.value
