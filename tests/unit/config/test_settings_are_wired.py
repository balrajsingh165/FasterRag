"""Every configuration setting must be read by something.

`security.max_request_mb` was declared, documented as enforced, and read by nothing — an
8 MB body against a 1 MB limit answered 200 (TASK-0199). A setting that validates but is
never consumed reads as a supported control and silently does nothing, which is the same
failure the CLI dead-flag guard exists to stop.
"""

import inspect
import re
from pathlib import Path

from pydantic import BaseModel

from fasterrag.adapters.llm import base as llm_base
from fasterrag.config.schema import Settings
from fasterrag.core import context as context_module

SRC = Path(__file__).resolve().parents[3] / "src" / "fasterrag"

# Declared, and deliberately not consumed yet. Each names the task that will land it, so an
# entry here is a decision somebody made rather than an omission nobody noticed.
PENDING: dict[str, str] = {
    "reliability.degradation_ladder": "TASK-0165",
}


def leaves() -> list[tuple[str, str]]:
    """Return every ``(dotted_name, attribute)`` in the schema."""
    found: list[tuple[str, str]] = []

    def walk(model: BaseModel, prefix: str = "") -> None:
        for name in type(model).model_fields:
            value = getattr(model, name)
            dotted = f"{prefix}{name}"
            if isinstance(value, BaseModel):
                walk(value, f"{dotted}.")
            else:
                found.append((dotted, name))

    walk(Settings())
    return found


def consumers() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC.rglob("*.py")
        if path.name != "schema.py" and "__pycache__" not in str(path)
    )


def test_the_schema_has_settings_to_check() -> None:
    """A walk that silently found nothing would make the test below vacuous."""
    assert len(leaves()) > 90


def test_every_setting_is_read_by_something() -> None:
    source = consumers()
    dead = [
        dotted
        for dotted, leaf in leaves()
        if dotted not in PENDING
        and not re.search(rf"\.{leaf}\b", source)
        and f'"{leaf}"' not in source
    ]

    assert dead == [], f"declared but never read: {dead}"


def test_every_pending_setting_is_still_declared() -> None:
    """An entry outliving its setting turns the exemption list into stale noise."""
    declared = {dotted for dotted, _ in leaves()}

    assert set(PENDING) <= declared


def test_no_model_property_is_computed_and_never_read() -> None:
    """The variant the flag, setting, and schema-field guards all miss.

    `Completion.truncated` and `AssembledContext.truncated` were each declared, docstringed,
    unit-tested in isolation, and read by nothing — so a model that stopped at its token
    ceiling was served as a complete answer (TASK-0206). A property is a declaration like
    any other, and an unread one is the same defect wearing a different shape.

    # CRITICAL: this matches by attribute *name*, not by type. Two classes sharing a
    # property name mask each other — one being read satisfies the check for both. It
    # catches a name nothing reads anywhere, which is the common case, and cannot catch a
    # name read only on the wrong class.
    """
    source = consumers()
    dead: list[str] = []

    for module in (llm_base, context_module):
        for owner_name, owner in vars(module).items():
            if not inspect.isclass(owner) or owner.__module__ != module.__name__:
                continue
            for name, member in vars(owner).items():
                if not isinstance(member, property) or name.startswith("_"):
                    continue
                if not re.search(rf"\.{name}\b", source):
                    dead.append(f"{owner_name}.{name}")

    assert dead == [], f"computed but never read: {dead}"


def test_every_pending_setting_cites_a_task() -> None:
    todo = (SRC.parents[1] / "docs" / "todo.md").read_text(encoding="utf-8")

    for setting, task in PENDING.items():
        assert task in todo, f"{setting} is pending against {task}, which todo.md does not list"
