"""``docs/python-api.md`` must describe the package surface that exists.

The document is the compatibility contract for ``pip install fasterrag``: what a caller may
import, what each ``FasterRag`` member takes and returns, and which entry-point groups a
plugin registers under. Nothing kept it true. Every list in it was hand-verified and then
left, which is how it came to promise ``priority_class``, ``QueryResult``, a
``CollectionsAPI``, and ``fasterrag._internal`` — none of which exist — while omitting
``index_lock``, ``create_collection``, ``cache``, ``embeddings``, ``journal``,
``vector_db``, and eleven of the seventeen abstract methods a vector-database plugin must
implement — every one of those re-measured against the tree on 2026-11-11 (TASK-0368).

Every claim is checked against the real thing, never string against string: a documented
symbol is resolved with ``importlib`` and ``getattr``, a documented signature is compared
with ``inspect.signature``, and a documented entry-point group is compared with the constant
the factory reads. Drift in either direction fails — a phantom entry sends a caller into an
``AttributeError``, and a missing one hides a stable name from the people it is stable for.

**A member has three states here, not two** (TASK-0368). It is documented and shipped, it is
undocumented, or it is documented *as deliberately absent* — which is what the document says
about ``export_archive`` and ``import_archive``, built everywhere except the facade. The
salvaged version of this file had the third state in a hardcoded list and never consulted it
from the phantom-member check, so it read the document's own honesty as a defect. The absent
ones now live in their own table, are read from it rather than hardcoded here, and are held
to the opposite assertion: they must still be missing, and must still name the task that
closes them — in a column of their own, because prose elsewhere in the row cites other tasks
and a citation that any mention satisfies records nothing.

**The vacuity guards below are floors measured against this document, not aspirations.** The
salvaged file asserted the shape of a rewrite that never landed — twelve member rows had to
exceed twelve, three stable modules had to reach eight — which is a gate failing on the
absence of work rather than on drift. A floor is only worth having if it is one the document
clears today and would stop clearing if a parser broke.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import re
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.facade import FasterRag
from fasterrag.sync import FasterRag as SyncFasterRag

DOC = Path(__file__).resolve().parents[3] / "docs" / "python-api.md"

_TASK_ID = re.compile(r"TASK-\d{4}")

# CRITICAL: floors measured against docs/python-api.md on 2026-11-11, there to catch a
# parser that silently matches nothing, not to demand a bigger document. Each is a little
# under what the document holds, so adding to it never fails and losing a section does.
MINIMUM_MEMBER_ROWS = 18
MINIMUM_SIGNATURE_ROWS = 12
MINIMUM_IMPORT_MODULES = 10
MINIMUM_IMPORTED_NAMES = 24
MINIMUM_RESULT_TYPES = 8
MINIMUM_STABLE_MODULES = 3
MINIMUM_PUBLIC_EXPORTS = 15
MINIMUM_PLUGIN_GROUPS = 3

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
_DOTTED_PATH = re.compile(r"^fasterrag(?:\.[A-Za-z_]\w*)+$")
_TYPE_NAME = re.compile(r"\b[A-Z]\w*")
_RESULT_TYPE = re.compile(r"^- `(\w+)` \(`([\w.]+)`\) — (.*)$", re.M)
_PLUGIN_GROUP = re.compile(
    r"^- `([\w.]+)` → subclass `(\w+)` \(`from ([\w.]+) import (\w+)`\): (.*)$", re.M
)
_FROZEN_EXCEPT = re.compile(r"^Every one of them is frozen except (.+?), which", re.M)
_TOML_GROUP = re.compile(r'entry-points\."([\w.]+)"')
_IMPORT_LINE = re.compile(r"from (fasterrag[\w.]*) import ((?:\w+(?:, )?)+)")
_IMPORT_BLOCK = re.compile(r"from (fasterrag[\w.]*) import \(([^)]*)\)")


@dataclasses.dataclass(frozen=True)
class Parameter:
    """One parameter in the shape the document and ``inspect`` can both be reduced to."""

    name: str
    keyword_only: bool
    has_default: bool
    annotation: str


def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def backticked(text: str) -> set[str]:
    return set(_BACKTICKED.findall(text))


def table(text: str, first_header: str) -> list[list[str]]:
    """Return the body rows of the table whose header row starts with ``first_header``.

    An escaped bar inside a cell is a literal one — a union type in a signature — rather
    than a column separator, so the split ignores it.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header} |"))
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)]
        rows.append(cells[1:-1])
    return rows


def split_top_level(text: str) -> list[str]:
    """Split on commas that are not nested inside brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def documented_signature(cell: str) -> tuple[list[Parameter], str]:
    """Parse ``(a: int, *, b: str | None = None) -> Ret`` from a signature cell."""
    depth = 0
    end = 0
    for index, char in enumerate(cell):
        depth += (char == "(") - (char == ")")
        end = index
        if depth == 0:
            break
    inner, rest = cell[1:end], cell[end + 1 :].strip()
    assert rest.startswith("->"), f"signature cell has no return type: {cell}"
    parameters: list[Parameter] = []
    keyword_only = False
    for piece in split_top_level(inner):
        if piece == "*":
            keyword_only = True
            continue
        head, _, default = piece.partition("=")
        name, _, annotation = head.partition(":")
        parameters.append(
            Parameter(name.strip(), keyword_only, bool(default), annotation.replace(" ", ""))
        )
    return parameters, rest[2:].replace(" ", "")


def annotation_text(annotation: object) -> str:
    """Render an annotation the way the document writes it, whitespace removed."""
    if annotation is inspect.Parameter.empty:
        return ""
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return text.replace(" ", "")


def real_signature(cls: type, name: str) -> tuple[list[Parameter], str]:
    """Reduce a class member's signature to the document's shape, ``self``/``cls`` dropped."""
    attribute = inspect.getattr_static(cls, name)
    bound = isinstance(attribute, classmethod)
    signature = inspect.signature(getattr(cls, name) if bound else attribute)
    parameters = [
        Parameter(
            parameter.name,
            parameter.kind is inspect.Parameter.KEYWORD_ONLY,
            parameter.default is not inspect.Parameter.empty,
            annotation_text(parameter.annotation),
        )
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]
    returns = annotation_text(signature.return_annotation).replace("Self", cls.__name__)
    return parameters, returns


def class_members(cls: type) -> dict[str, Any]:
    """Public names the class body defines."""
    return {name: value for name, value in vars(cls).items() if not name.startswith("_")}


def instance_members(instance: object) -> set[str]:
    """Public names ``__init__`` sets on an instance, invisible to class introspection."""
    return {name for name in vars(instance) if not name.startswith("_")}


def is_method(value: object) -> bool:
    return inspect.isfunction(value) or isinstance(value, classmethod)


def async_instance() -> FasterRag:
    return FasterRag.from_settings(Settings())


def sync_instance() -> SyncFasterRag:
    return SyncFasterRag.from_settings(Settings())


def member_rows() -> list[tuple[list[str], str]]:
    """Return ``(names, signature cell)`` for every row of the ``FasterRag`` table."""
    rows: list[tuple[list[str], str]] = []
    for cells in table(doc_text(), "Member"):
        names = [token.removeprefix("FasterRag.") for token in _BACKTICKED.findall(cells[0])]
        rows.append((names, cells[1].strip("`")))
    return rows


def absent_members() -> dict[str, str]:
    """Return ``name -> closing task`` from the deliberately-absent table.

    Read from the document rather than listed here, so the gate cannot disagree with the
    page a caller reads about what is missing on purpose. The task has its own column
    because prose elsewhere in the row cites other tasks, and a citation that any mention
    satisfies records nothing.
    """
    found: dict[str, str] = {}
    for cells in table(doc_text(), "Absent member"):
        for name in _BACKTICKED.findall(cells[0]):
            found[name] = cells[1].strip()
    return found


def signature_rows() -> list[tuple[str, str]]:
    """Return ``(name, cell)`` for the rows whose cell is a signature rather than a kind.

    A row may name two members sharing one description — the context-manager pair — and
    those never carry a signature, so refusing to unpack them is the parse bug, not a
    document defect.
    """
    return [
        (names[0], cell)
        for names, cell in member_rows()
        if cell.startswith("(") and len(names) == 1
    ]


def documented_members() -> set[str]:
    return {name for names, _ in member_rows() for name in names}


def documented_result_types() -> dict[str, tuple[str, set[str]]]:
    """Return ``name -> (module, documented fields)`` from the result-types list."""
    found: dict[str, tuple[str, set[str]]] = {}
    for name, module, rest in _RESULT_TYPE.findall(doc_text()):
        fields = {token for token in _BACKTICKED.findall(rest) if _IDENTIFIER.match(token)}
        found[name] = (module, fields)
    return found


def documented_mutable_result_types() -> set[str]:
    """Return the result types the document excepts from its frozen claim."""
    (sentence,) = _FROZEN_EXCEPT.findall(doc_text())
    return set(_BACKTICKED.findall(sentence))


def documented_imports() -> dict[str, set[str]]:
    """Return ``module -> names`` for every ``from fasterrag... import`` the document shows."""
    text = doc_text()
    found: dict[str, set[str]] = {}
    for module, names in _IMPORT_LINE.findall(text) + _IMPORT_BLOCK.findall(text):
        found.setdefault(module, set()).update(
            name.strip() for name in names.split(",") if name.strip()
        )
    return found


def documented_names() -> set[str]:
    """Every name the document shows: in backticks, or in a ``from ... import`` it prints."""
    names = backticked(doc_text())
    for imported in documented_imports().values():
        names |= imported
    return names


def documented_public_modules() -> list[str]:
    """Return the modules the stability table places under the SemVer-stable public API."""
    for surface, guarantee in table(doc_text(), "Surface"):
        if "SemVer-stable public API" in guarantee:
            return [
                token for token in _BACKTICKED.findall(surface) if token.startswith("fasterrag")
            ]
    raise AssertionError("the stability table has no SemVer-stable public API row")


def documented_plugin_groups() -> dict[str, tuple[str, str, set[str]]]:
    """Return ``group -> (base class, module it is imported from, abstract methods)``."""
    found: dict[str, tuple[str, str, set[str]]] = {}
    for group, base, module, imported, rest in _PLUGIN_GROUP.findall(doc_text()):
        assert base == imported, f"{group} names {base} but imports {imported}"
        found[group] = (base, module, set(_BACKTICKED.findall(rest)))
    return found


def real_plugin_groups() -> dict[str, ModuleType]:
    """Return ``entry-point group -> factory module`` by walking the adapter packages."""
    adapters = importlib.import_module("fasterrag.adapters")
    found: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(adapters.__path__):
        if not info.ispkg or find_spec(f"fasterrag.adapters.{info.name}.factory") is None:
            continue
        factory = importlib.import_module(f"fasterrag.adapters.{info.name}.factory")
        found[factory.ENTRY_POINT_GROUP] = factory
    return found


def resolve(dotted: str) -> object:
    """Resolve ``fasterrag.a.b`` as a module, else as attribute ``b`` of module ``fasterrag.a``."""
    try:
        if find_spec(dotted) is not None:
            return importlib.import_module(dotted)
    except ModuleNotFoundError:
        pass
    module, _, attribute = dotted.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


def test_the_member_table_was_parsed() -> None:
    """A table pattern matching nothing would report perfect parity with an empty set."""
    assert len(member_rows()) >= MINIMUM_MEMBER_ROWS
    assert len(signature_rows()) >= MINIMUM_SIGNATURE_ROWS


def test_the_import_statements_were_parsed() -> None:
    """Likewise for the code blocks: silence here would make the import check vacuous."""
    imports = documented_imports()

    assert len(imports) >= MINIMUM_IMPORT_MODULES
    assert sum(len(names) for names in imports.values()) >= MINIMUM_IMPORTED_NAMES


def test_the_stability_table_names_the_public_modules() -> None:
    assert len(documented_public_modules()) >= MINIMUM_STABLE_MODULES


def test_every_documented_facade_member_exists() -> None:
    """A row for a member that does not exist sends a caller into an ``AttributeError``."""
    surface = set(class_members(FasterRag)) | instance_members(async_instance())
    phantom = [
        name
        for names, _ in member_rows()
        for name in names
        if name not in surface and name not in vars(FasterRag)
    ]

    assert phantom == [], f"rows in python-api.md with no facade member: {phantom}"


def test_every_documented_signature_matches_the_facade() -> None:
    """Names, keyword-only-ness, defaults, annotations, and return type — the whole column."""
    mismatched: list[str] = []
    for name, cell in signature_rows():
        documented = documented_signature(cell)
        real = real_signature(FasterRag, name)
        if documented != real:
            mismatched.append(f"{name}: documented {documented}, real {real}")

    assert mismatched == [], "\n".join(mismatched)


def test_every_documented_property_is_a_property() -> None:
    members = class_members(FasterRag)
    wrong = [
        name
        for names, cell in member_rows()
        if cell in {"property", "properties"}
        for name in names
        if not isinstance(members.get(name), property)
    ]

    assert wrong == [], f"documented as properties but not properties: {wrong}"


def test_every_public_facade_member_has_a_row() -> None:
    """The table is the contract; a public member with no row is stable for nobody."""
    documented = documented_members()
    missing = sorted(name for name in class_members(FasterRag) if name not in documented)

    assert missing == [], f"public facade members with no row in python-api.md: {missing}"


def test_every_public_instance_attribute_is_documented() -> None:
    """``settings`` is set in ``__init__``, so class introspection alone would miss it."""
    names = documented_names()
    missing = sorted(name for name in instance_members(async_instance()) if name not in names)

    assert missing == [], f"public instance attributes python-api.md never names: {missing}"


def test_the_absent_members_are_still_absent_and_still_cite_a_task() -> None:
    """When an archive method lands, this fails: move it into the table and delist it."""
    absent = absent_members()

    assert absent, "the deliberately-absent table parsed to nothing"
    for name, task in absent.items():
        assert not hasattr(FasterRag, name), f"{name} exists now; give it a row and delist it"
        assert not hasattr(SyncFasterRag, name), f"{name} exists on the sync facade now"
        assert _TASK_ID.fullmatch(task), f"{name} is absent on purpose but names no task"


def test_no_member_is_both_promised_and_recorded_as_absent() -> None:
    """The two tables are the two answers; a name in both of them answers neither."""
    both = sorted(documented_members() & set(absent_members()))

    assert both == [], f"documented as callable and as deliberately absent: {both}"


def test_the_sync_facade_mirrors_every_async_method() -> None:
    """python-api.md promises a blocking counterpart per method, same name and parameters."""
    async_members = class_members(FasterRag)
    sync_members = class_members(SyncFasterRag)
    assert {"__aenter__", "__aexit__"} <= set(vars(FasterRag))
    assert {"__enter__", "__exit__"} <= set(vars(SyncFasterRag))

    missing = sorted(
        name
        for name, value in async_members.items()
        if is_method(value) and name not in sync_members
    )
    assert missing == [], f"async methods with no blocking counterpart: {missing}"

    differing = [
        name
        for name, value in async_members.items()
        if is_method(value)
        and real_signature(FasterRag, name)[0] != real_signature(SyncFasterRag, name)[0]
    ]
    assert differing == [], f"blocking counterparts whose parameters differ: {differing}"


def test_the_sync_facade_adds_nothing_the_document_does_not_cover() -> None:
    """A sync-only member would be undocumented, since the table describes the async one."""
    async_surface = set(class_members(FasterRag)) | instance_members(async_instance())
    sync_surface = set(class_members(SyncFasterRag)) | instance_members(sync_instance())
    extra = sorted(sync_surface - async_surface)

    assert extra == [], f"sync facade members the async facade lacks: {extra}"


def test_the_result_types_were_parsed() -> None:
    assert len(documented_result_types()) >= MINIMUM_RESULT_TYPES


def test_every_documented_result_type_lists_exactly_its_fields() -> None:
    """The fields are the contract, and the data-model rule is never to invent one."""
    mismatched: list[str] = []
    for name, (module, documented) in documented_result_types().items():
        cls = getattr(importlib.import_module(module), name)
        assert dataclasses.is_dataclass(cls), f"{module}.{name} is not a dataclass"
        real = {field.name for field in dataclasses.fields(cls)} | {
            attribute for attribute, value in vars(cls).items() if isinstance(value, property)
        }
        if documented != real:
            mismatched.append(
                f"{name}: documented but absent {sorted(documented - real)}, "
                f"real but undocumented {sorted(real - documented)}"
            )

    assert mismatched == [], "\n".join(mismatched)


def test_the_mutable_result_types_are_the_ones_the_document_excepts() -> None:
    """Immutability is what a caller caches an answer on, so it is part of the contract.

    Named as an exception rather than asserted wholesale, because the exception is the
    interesting half: a type that quietly stopped being frozen is a shared object a caller
    was told it could keep.
    """
    documented = documented_mutable_result_types()
    real = {
        name
        for name, (module, _) in documented_result_types().items()
        if not getattr(importlib.import_module(module), name).__dataclass_params__.frozen
    }

    assert documented == real, f"documented mutable {sorted(documented)}, real {sorted(real)}"


def test_every_facade_return_type_is_a_documented_result_type() -> None:
    """A method may only return what the result-types list describes, or the facade itself."""
    described = set(documented_result_types()) | {FasterRag.__name__, "None"}
    unexplained = sorted(
        {
            type_name
            for names, cell in member_rows()
            if cell.startswith("(")
            for type_name in _TYPE_NAME.findall(documented_signature(cell)[1])
            if type_name not in described and type_name not in {"AsyncIterator", "list"}
        }
    )

    assert unexplained == [], f"return types with no result-types entry: {unexplained}"


def test_every_documented_import_resolves_to_a_public_name() -> None:
    """Resolved for real, and required to be in ``__all__`` — the document's own rule."""
    problems: list[str] = []
    for module_name, names in documented_imports().items():
        module = importlib.import_module(module_name)
        for name in sorted(names):
            if not hasattr(module, name):
                problems.append(f"from {module_name} import {name}: no such attribute")
            elif name not in module.__all__:
                problems.append(f"from {module_name} import {name}: not in __all__")

    assert problems == [], "\n".join(problems)


def test_every_public_export_is_documented() -> None:
    """Every name in a stable module's ``__all__`` must appear in the document."""
    names = documented_names()
    exported = 0
    missing: list[str] = []
    for module_name in documented_public_modules():
        module = importlib.import_module(module_name)
        for name in module.__all__:
            exported += 1
            if name not in names:
                missing.append(f"{module_name}.{name}")

    assert exported >= MINIMUM_PUBLIC_EXPORTS, "too few public exports for this to mean much"
    assert missing == [], f"public exports python-api.md never names: {missing}"


def test_every_documented_module_path_exists() -> None:
    """Every ``fasterrag.x.y`` in backticks is a module or an attribute of one."""
    groups = set(real_plugin_groups())
    broken: list[str] = []
    for token in sorted(backticked(doc_text())):
        if not _DOTTED_PATH.match(token) or token in groups:
            continue
        try:
            resolve(token)
        except (ImportError, AttributeError) as exc:
            broken.append(f"{token}: {exc}")

    assert broken == [], "\n".join(broken)


def test_the_factories_declare_entry_point_groups() -> None:
    assert len(real_plugin_groups()) >= MINIMUM_PLUGIN_GROUPS


def test_the_documented_plugin_groups_are_the_ones_the_factories_read() -> None:
    documented = documented_plugin_groups()
    real = real_plugin_groups()

    assert set(documented) == set(real), (
        f"documented {sorted(documented)}, factories read {sorted(real)}"
    )
    assert set(_TOML_GROUP.findall(doc_text())) <= set(real)


def test_every_plugin_base_class_lists_exactly_its_abstract_methods() -> None:
    """A plugin author implements what this list says; a missing name is a broken plugin."""
    factories = real_plugin_groups()
    problems: list[str] = []
    for group, (base_name, module_name, documented) in documented_plugin_groups().items():
        base = getattr(importlib.import_module(module_name), base_name)
        assert inspect.isabstract(base), f"{base_name} is not abstract"
        assert base in vars(factories[group]).values(), f"{group}'s factory never names {base_name}"
        real = set(base.__abstractmethods__)
        if documented != real:
            problems.append(
                f"{base_name}: documented but not abstract {sorted(documented - real)}, "
                f"abstract but undocumented {sorted(real - documented)}"
            )

    assert problems == [], "\n".join(problems)


def test_documented_calls_await_exactly_the_coroutine_functions() -> None:
    """``report = evaluate(...)`` without ``await`` hands a caller a coroutine, not a report."""
    text = doc_text()
    wrong: list[str] = []
    for module_name, names in documented_imports().items():
        module = importlib.import_module(module_name)
        for name in names:
            target = getattr(module, name)
            if not inspect.isfunction(target):
                continue
            awaited = re.search(rf"await {name}\(", text) is not None
            called = re.search(rf"(?<!await )(?<![\w.]){name}\(", text) is not None
            if awaited and not inspect.iscoroutinefunction(target):
                wrong.append(f"{name} is awaited but is not a coroutine function")
            if called and inspect.iscoroutinefunction(target):
                wrong.append(f"{name} is called without await but is a coroutine function")

    assert wrong == [], "\n".join(wrong)


def test_documented_calls_use_keywords_the_callable_accepts() -> None:
    """``RecursiveChunker(chunk_size=768, overlap=64)`` must be a call that works."""
    text = doc_text()
    wrong: list[str] = []
    for module_name, names in documented_imports().items():
        module = importlib.import_module(module_name)
        for name in names:
            target = getattr(module, name)
            if not callable(target):
                continue
            accepted = set(inspect.signature(target).parameters)
            for arguments in re.findall(rf"(?<![\w.]){name}\(([^()]*)\)", text):
                for keyword in re.findall(r"(\w+)=", arguments):
                    if keyword not in accepted:
                        wrong.append(f"{name}({keyword}=...) is not a parameter it accepts")

    assert wrong == [], "\n".join(wrong)
