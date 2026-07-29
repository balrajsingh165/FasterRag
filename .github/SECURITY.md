# Security Policy

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report privately via GitHub's private vulnerability reporting: **Security tab → "Report a vulnerability"** on this repository. Include reproduction steps, affected version/commit, and impact assessment. You will receive an acknowledgement, and a fix or mitigation plan will be coordinated with you before any public disclosure.

## Scope

In scope:

- The fasterRag codebase (once the build phase ships), the published `fasterrag` package, and the documented provisioning flows (Qdrant/Langfuse/Grafana).
- Secret-handling violations: anything that causes credentials to be logged, persisted outside `.env`, or returned by an API/dashboard surface.
- Isolation failures: cross-tenant data access, semantic-cache leakage across tenants, dashboard gaining any control capability.

Out of scope:

- Vulnerabilities in third-party backends themselves (Qdrant, Langfuse, Grafana, model providers) — report those upstream; we will still track pinned-version bumps here.
- Deployments that ignore the documented hardening guidance in [docs/security.md](../docs/security.md) and [docs/deployment.md](../docs/deployment.md) (e.g. exposing Qdrant or the dashboard publicly).

## Supported versions

Pre-1.0: only the latest tagged release line receives security fixes. The dependency policy (hash-locked installs, secret scanning in CI, non-root containers, SBOM per release) is documented in [docs/security.md](../docs/security.md).
