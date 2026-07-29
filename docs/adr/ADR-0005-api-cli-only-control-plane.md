# ADR-0005: Programmatic-Only Control Plane (API + CLI + Library); Dashboard Is Observability-Only

- Status: accepted
- Date: 2026-07-29
- Deciders: fasterRag maintainers

## Context and Problem Statement

The requirements contain an apparent contradiction: "the system is operated only through a terminal/API — no GUI" and "ship a self-hosted web dashboard." How are both satisfied, and where is the line?

## Decision Drivers

- Auditability and automation: every control action must be scriptable, loggable, and reproducible.
- A control GUI doubles the attack surface and the testing matrix for the exact operations that must be safest.
- Operators still need rich visual inspection: latencies, costs, cache stats, full LLM I/O history.

## Considered Options

1. **Control plane = REST API + CLI + Python library only; a separate, optional, read-only observability dashboard**
2. Full web admin UI (control + observability)
3. No visual surface at all

## Decision Outcome

**Chosen: option 1.** The three programmatic surfaces (REST API, CLI, importable Python package) share one service layer, so behavior is identical everywhere. The dashboard is a separate, optional, self-hosted GUI that renders metrics, traces, and history but **contains zero mutating routes by construction** — enforced by a route-table test (FMEA row 37). This reconciliation is stated as an explicit assumption in [scope.md](../scope.md) and [observability.md](../observability.md).

### Consequences

- Good: every control action is scriptable and auditable; the dashboard cannot become a privilege-escalation or CSRF surface for control operations.
- Good: the contradiction is resolved by definition, documented as an assumption, and testable (zero mutating routes).
- Bad: operators who expect click-to-configure must use CLI/API/config instead; mitigated by `fasterrag doctor`'s fix-it guidance and one-toggle provisioning.
- Constraint: any future PR adding a control capability to the dashboard is rejected on principle; revisiting this requires superseding this ADR.
