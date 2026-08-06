"""Every API request field must be read by the handler that accepts it.

`IngestRequest.priority_class` was declared, documented in `api-reference.md` as "used by
tiered embedding routing (D9)", and read by nothing — the API twin of the CLI's
`--priority-class`, which had the same defect. `ReplayRequest.diff_only` was declared,
undocumented, and ignored. A field a caller sets and a handler drops is indistinguishable
from one that worked.
"""

import inspect
import re
from pathlib import Path

from pydantic import BaseModel

from fasterrag.api import schemas

API = Path(__file__).resolve().parents[3] / "src" / "fasterrag" / "api"

# Fields a handler never touches by name because FastAPI or a service consumes the whole
# model. Each is genuinely read, just not through `.field` in this package.
INDIRECT: frozenset[str] = frozenset()


def declared() -> list[str]:
    """Return every ``Model.field`` the API schemas define."""
    found: list[str] = []
    for name, obj in vars(schemas).items():
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
            found.extend(f"{name}.{field}" for field in obj.model_fields)
    return found


def handlers() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in API.rglob("*.py")
        if path.name != "schemas.py" and "__pycache__" not in str(path)
    )


def test_the_schemas_define_fields_to_check() -> None:
    """A reflection that found nothing would make the test below vacuous."""
    assert len(declared()) > 20


def test_every_request_field_is_read_by_a_handler() -> None:
    source = handlers()
    dead = [
        item
        for item in declared()
        if item not in INDIRECT and not re.search(rf"\.{item.split('.')[1]}\b", source)
    ]

    assert dead == [], f"declared but never read: {dead}"
