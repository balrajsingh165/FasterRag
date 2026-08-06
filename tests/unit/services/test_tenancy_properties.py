"""Tenant scoping invariants, generated rather than reasoned about.

Isolation here is a property of the *identifier*: a tenant's collections are prefixed, and a
tenant may address only names carrying its own prefix. That makes the whole guarantee rest on
one question — can two different tenants ever produce the same backend name?

They could. `scoped_name("_b", "a")` and `scoped_name("b", "a_")` both produced `a___b`, so
two tenants addressed one collection and `visible_to` returned true for both — read and
write, since the name is what the adapter upserts into (TASK-0211). The tenant pattern
forbade the separator *inside* an id but allowed one to end with its first character, which
reconstructs it across the boundary with the collection name.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from fasterrag.errors import FasterRagError
from fasterrag.services.tenancy import (
    SEPARATOR,
    scoped_name,
    unscoped_name,
    visible_to,
)


def _accepted(tenant: str) -> bool:
    """Return whether the validator permits this id, by asking it."""
    try:
        scoped_name("probe", tenant)
    except FasterRagError:
        return False
    return True


# CRITICAL: filtered by calling the validator, never by restating its rules. A strategy that
# hardcoded "no trailing underscore" would encode the fix, so weakening the rule would widen
# what the system accepts while leaving what the tests generate unchanged — and the
# injectivity property below would go on passing over inputs that can no longer collide.
# Verified: reverting the fix makes this strategy widen and the collision reappear.
TENANT = st.from_regex(r"\A[a-zA-Z0-9][a-zA-Z0-9._-]{0,12}\Z").filter(_accepted)

# Unrestricted apart from length. A collection name is caller-supplied, and the collision
# this suite exists to rule out was reached through the *name*, not the tenant.
NAME = st.text(min_size=1, max_size=16)


@given(name=NAME, tenant=TENANT)
def test_scoping_round_trips(name: str, tenant: str) -> None:
    """A tenant named it `docs` and every response must call it `docs`."""
    assert unscoped_name(scoped_name(name, tenant), tenant) == name


@given(name=NAME, tenant=TENANT)
def test_a_tenant_can_address_what_it_creates(name: str, tenant: str) -> None:
    assert visible_to(scoped_name(name, tenant), tenant)


@given(name=NAME, first=TENANT, second=TENANT)
def test_two_tenants_never_share_a_backend_collection(name: str, first: str, second: str) -> None:
    """The whole isolation guarantee: distinct tenants, distinct backend names."""
    assume(first != second)

    assert scoped_name(name, first) != scoped_name(name, second)


@given(first=NAME, second=NAME, one=TENANT, two=TENANT)
def test_no_pair_of_tenants_and_names_collides(first: str, second: str, one: str, two: str) -> None:
    """The general form, and the one that found the bug.

    Equal backend names must mean equal owners *and* equal collections — otherwise one
    tenant reads and writes another's vectors while both believe the name is theirs.
    """
    assume((one, first) != (two, second))

    assert scoped_name(first, one) != scoped_name(second, two)


@given(
    stem=st.from_regex(r"\A[a-zA-Z0-9]{1,8}\Z"),
    tail=st.text(min_size=1, max_size=8),
)
def test_a_tenant_extending_another_cannot_reach_its_collections(stem: str, tail: str) -> None:
    """Aimed at the shape random search will not find.

    The collision needs four values to line up — a tenant, a longer tenant sharing its
    prefix, and two names that make the concatenations meet. Uniform generation reaches
    that about never, so the pair is constructed instead: `(stem, SEP[0] + tail)` against
    `(stem + SEP[0], tail)`. Both were `stem + SEP + SEP[0] + tail` before the fix.

    Either the longer tenant is refused, or the two names differ. Both are acceptable
    outcomes; sharing one backend name is not.
    """
    longer = f"{stem}{SEPARATOR[0]}"
    try:
        extended = scoped_name(tail, longer)
    except FasterRagError:
        return

    assert extended != scoped_name(f"{SEPARATOR[0]}{tail}", stem)


@given(name=NAME, owner=TENANT, other=TENANT)
def test_a_tenant_cannot_see_another_tenants_collection(name: str, owner: str, other: str) -> None:
    assume(owner != other)

    assert not visible_to(scoped_name(name, owner), other)


@given(name=NAME)
def test_an_untenanted_deployment_is_untouched(name: str) -> None:
    """Enabling multi-tenancy is the only thing that may ever move a collection."""
    assert scoped_name(name, None) == name
    assert unscoped_name(name, None) == name
    assert visible_to(name, None)


@given(name=NAME, tenant=TENANT)
def test_a_tenant_does_not_see_untenanted_collections(name: str, tenant: str) -> None:
    """Those predate tenancy and belong to nobody who is asking."""
    assume(not name.startswith(f"{tenant}{SEPARATOR}"))

    assert not visible_to(name, tenant)


@given(tenant=st.from_regex(r"\A[a-zA-Z0-9][a-zA-Z0-9._-]{0,8}\Z"))
def test_an_id_ending_in_the_separators_first_character_is_refused(tenant: str) -> None:
    """The fix. Such an id reconstructs the separator against a name that starts with it."""
    assume(SEPARATOR not in tenant)

    with pytest.raises(FasterRagError):
        scoped_name("x", f"{tenant}{SEPARATOR[0]}")


@given(tenant=st.text(min_size=1, max_size=12))
def test_an_id_containing_the_separator_is_refused(tenant: str) -> None:
    with pytest.raises(FasterRagError):
        scoped_name("x", f"{tenant}{SEPARATOR}{tenant}")
