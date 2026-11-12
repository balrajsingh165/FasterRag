"""Pin the degradation ladder's registry: the thing that makes a rung denial decidable.

Two properties are under test and nothing else. **A rung cannot be served without
registering itself**, which is what lets `scripts/check_unbuilt_claims.py` read
`_BUILTIN_RUNGS` as the answer to "is this rung built?" (TASK-0394). And **the wire form
did not move**: `mode` is still the same plain string in the same place, because it is
published in `docs/api-reference.md`, stored in the semantic cache, and used as a metric
label, so changing it would be a breaking change wearing a refactor's clothes.
"""

from __future__ import annotations

import json

import pytest

from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services import degradation
from fasterrag.services.degradation import DegradationMode, serving
from fasterrag.services.generation import EXTRACTIVE_MODE, Answer, extractive_answer
from fasterrag.services.querying import FULL_MODE, HYBRID_ONLY_MODE, fused_shortlist


def test_every_mode_but_full_is_registered_as_built_or_as_planned() -> None:
    """The import-time invariant, restated: the enum and the registry cannot drift."""
    registered = set(degradation._BUILTIN_RUNGS) | degradation._PLANNED_RUNGS

    assert {mode.value for mode in DegradationMode} - {"full"} == registered
    assert set(degradation._BUILTIN_RUNGS) & degradation._PLANNED_RUNGS == set()


def test_full_is_not_a_rung() -> None:
    """The top of the ladder registers no fallback and no document denies it."""
    assert "full" not in set(degradation._BUILTIN_RUNGS) | degradation._PLANNED_RUNGS

    with pytest.raises(FasterRagError) as caught:
        serving(DegradationMode.FULL)

    assert caught.value.code is ErrorCode.INTERNAL


def test_a_declared_rung_that_is_not_implemented_cannot_be_served() -> None:
    """`cache_only` is deferred by ADR-0011. Serving it before registering it must be impossible."""
    with pytest.raises(FasterRagError) as caught:
        serving(DegradationMode.CACHE_ONLY)

    assert caught.value.code is ErrorCode.INTERNAL
    assert "_BUILTIN_RUNGS" in caught.value.detail
    assert "ADR-0011" in caught.value.detail


def test_each_built_rung_registers_the_function_that_actually_serves_it() -> None:
    """A reference to code that no longer exists is a registry that has stopped being one."""
    assert degradation._BUILTIN_RUNGS == {
        "hybrid_only": "fasterrag.services.querying:fused_shortlist",
        "extractive": "fasterrag.services.generation:extractive_answer",
    }
    assert callable(fused_shortlist)
    assert callable(extractive_answer)


def test_a_rung_whose_implementation_has_gone_is_refused_rather_than_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution is why the mapping is rung-to-implementation and not a list of names."""
    monkeypatch.setitem(
        degradation._BUILTIN_RUNGS, "extractive", "fasterrag.services.generation:gone"
    )

    with pytest.raises(FasterRagError) as caught:
        serving(DegradationMode.EXTRACTIVE)

    assert "not a callable" in caught.value.detail


def test_a_rung_whose_module_has_gone_is_refused_rather_than_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of resolution: a module that no longer imports is not a rung either."""
    monkeypatch.setitem(degradation._BUILTIN_RUNGS, "extractive", "fasterrag.nowhere:answer")

    with pytest.raises(FasterRagError) as caught:
        serving(DegradationMode.EXTRACTIVE)

    assert "does not import" in caught.value.detail


@pytest.mark.parametrize(
    ("mode", "wire"),
    [(DegradationMode.HYBRID_ONLY, "hybrid_only"), (DegradationMode.EXTRACTIVE, "extractive")],
)
def test_a_built_rung_serves_the_exact_string_the_api_documents(
    mode: DegradationMode, wire: str
) -> None:
    """`docs/api-reference.md` publishes these values; the registry may not restate them."""
    served = serving(mode)

    assert served == wire
    # CRITICAL: `type(...) is str`, not isinstance. A StrEnum member passes isinstance and
    # would still reach a response body, a metric label and the cache as an enum. The whole
    # wire-compatibility argument for this change is that it does not.
    assert type(served) is str


def test_the_mode_constants_are_the_plain_strings_they_have_always_been() -> None:
    """Every caller still imports these three names and gets the same values."""
    assert (FULL_MODE, HYBRID_ONLY_MODE, EXTRACTIVE_MODE) == ("full", "hybrid_only", "extractive")
    assert all(type(value) is str for value in (FULL_MODE, HYBRID_ONLY_MODE, EXTRACTIVE_MODE))


def test_a_degraded_response_body_serializes_exactly_as_before() -> None:
    """The published contract, end to end: `mode` is a JSON string and `degraded` is true."""
    answer = Answer(answer="passages", mode=serving(DegradationMode.EXTRACTIVE))

    body = json.loads(json.dumps(answer.as_dict()))

    assert body["mode"] == "extractive"
    assert body["degraded"] is True
