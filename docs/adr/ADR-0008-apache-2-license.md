# ADR-0008: fasterRag Is Licensed Under Apache-2.0

- Status: accepted
- Date: 2026-09-13
- Deciders: fasterRag maintainers

## Context and Problem Statement

`pyproject.toml` declared `GPL-3.0-or-later` from the first commit, and nothing recorded that as a choice. It read as a default nobody had revisited, and a license is the one part of a package that cannot be revisited after the fact: **a PyPI version number can never be reused**, so the first upload fixes the license grant for that version permanently, and anyone who takes the GPL grant keeps it. Relicensing later needs every contributor's agreement.

The stated adoption thesis ([scope.md](../scope.md)) is teams embedding fasterRag inside their own products. GPL's copyleft is precisely what a commercial legal review blocks on: a product that links a GPL library must itself be released under the GPL. The default therefore contradicted the goal, and the contradiction had to be resolved before the first release rather than discovered by the first adopter.

## Decision Drivers

- The adoption funnel the documentation courts is closed-source products embedding the library. The license must permit that.
- The first PyPI upload is irreversible for that version. The decision has to be deliberate and written down before it.
- The projects fasterRag is positioned against — LangChain, LlamaIndex, Qdrant, Milvus — are Apache-2.0. Adopters already know what that grant means.
- Users of an AI-adjacent library are exposed to patent risk; a license with an explicit patent grant protects them and, reciprocally, the project.
- There is one contributor at the time of the decision, so relicensing now costs nothing; every commit from a second contributor onward raises that cost.
- No dependency constrains the choice: the stack (FastAPI, Pydantic, httpx, the Qdrant client, psycopg, the provider SDKs) is MIT, BSD, or Apache licensed, with no copyleft dependency that would force GPL.

## Considered Options

1. **Keep GPL-3.0-or-later.** Strong copyleft; any product embedding fasterRag must be open-sourced under the GPL.
2. **MIT.** Maximally permissive; no patent grant.
3. **Apache-2.0.** Permissive, with an explicit patent grant and a `NOTICE` mechanism for attribution.

## Decision Outcome

**Option 3, Apache-2.0.** It permits the embedding the project exists for, it is the license the comparable ecosystem uses so adopters need no review, and it carries the patent grant MIT lacks. GPL was rejected because it defeats the adoption thesis; MIT was rejected only on the patent grant, which costs nothing to have and something to lack.

The decision was made by the maintainer on 2026-09-13 and applied in the same change: the canonical Apache-2.0 text replaces `LICENSE`, a `NOTICE` file carries the copyright line, and `pyproject.toml` declares `Apache-2.0` in both the `license` field and the trove classifier so the wheel's metadata agrees with the file it ships.

### Consequences

- Closed-source products may embed fasterRag with attribution and without releasing their own source.
- Contributions are accepted under the same license (inbound = outbound); a contributor license agreement is not required, and [CONTRIBUTING.md](../CONTRIBUTING.md) says so.
- `NOTICE` must travel with every distribution, which the sdist and wheel includes do.
- The first PyPI release (TASK-0087) is no longer blocked on this decision; blocker B1 is retired.
- Changing the license again would need the agreement of every contributor at that time. This record is what makes that a deliberate future decision rather than another default.

## Links

- [release.md](../release.md) — the release procedure that lists the license decision as irreversible and orders it first
- [security.md](../security.md) §supply chain — the dependency-license verification that confirmed no copyleft constraint
- TASK-0164 in [todo.md](../todo.md)
