# release.md — Release Procedure for the `fasterrag` Package

The package releases to PyPI under the name **`fasterrag`** (as declared in `pyproject.toml`). This document is the complete, ordered procedure — preconditions, mechanics, publication, post-release, rollback — and the **confidence threshold that authorizes it**. The maintainer executes releases; nothing here is automated end-to-end on purpose, because two steps (license, version) are irreversible.

> **Standing rule for release notes and announcements**: the provable-claims policy applies to them exactly as to docs — no performance or superiority claim without a [benchmarks.md](benchmarks.md) ledger entry. Until the isolated-hardware baseline lands (TASK-0084), announcements describe capabilities and goals, never speed.

## 1. The release gate: confidence threshold

A release is authorized only when **all three** hold:

1. **Release-confidence score ≥ 85/100**, re-derived in [release-readiness.md](release-readiness.md) **within 7 days** of the intended tag, using that document's method (gates re-run, claims probed, docs-vs-code swept). A stale score does not count — a confidence number that is not re-derived is exactly the unbacked claim this repository forbids. *(Current: 62/100 as of 2026-08-10 → not releasable.)*
2. **Zero open items in the release-gating checklist** (release-readiness.md §3.1). Items may close by building **or** by an honest narrowing recorded in the docs and the ledger — "cut the surface" is a legitimate close; "leave it ambiguous" is not.
3. **CI fully green on the exact tag commit** — every job, including integration, models, Windows, and the doc gates.

Threshold by release type: **beta (`0.x.0-betaN`) ≥ 85** as above. **First stable (`1.0.0`) ≥ 90**, additionally requiring the measurement program complete: citable ledger baselines (TASK-0084), SLO targets set from them, chaos recovery times (TASK-0136), and the clean-host DR drill with published RPO/RTO (TASK-0085). The name is "fasterRag"; a stable release with an empty ledger would be the contradiction the whole doctrine exists to prevent.

## 2. Pre-release checklist (the current §3.1 items, resolved in this order)

**Irreversible decisions first** — these can never be amended after the first upload, because **a PyPI version number can never be reused**, even after deletion:

- [ ] License ADR (TASK-0164, B1): keep GPL-3.0-or-later or move permissive; record the reasoning as ADR-0008/0009. The uploaded artifact fixes the license grant for that version forever.
- [ ] Identity encoding (TASK-0210): length-prefixed digest or keep NUL-joined + policing; either way `IDENTITY_VERSION` is frozen by the tag — after it, changing ids forces every adopter through a reindex.
- [ ] Version number (TASK-0020): `0.1.0-beta.1` unless decided otherwise.

**Then the engineering closes** (build or narrow, each with its task): cost governor (TASK-0242) · D7 gate into CI (TASK-0244) · REST provisioning parity (TASK-0251) · coverage reconciliation (TASK-0252) · LLM breaker consult-or-remove (TASK-0245) · disk-full typing (TASK-0234) · reranker default decision (TASK-0175, B6) · ladder-scope ADR (TASK-0165, ships or formally defers TASK-0159) · the batched maintainer reviews of the adapter surface (TASK-0126/0133/0182/0191 — after the tag they are public API).

## 3. Release mechanics (execute in order; stop on any failure)

1. **Re-derive confidence** per §1. Record the score in release-readiness.md; proceed only at threshold.
2. **Freeze**: no feature commits on `main` between the readiness run and the tag; fixes found during the freeze restart step 1.
3. **Stamp**: set `__version__` in `src/fasterrag/__init__.py` (hatch reads it) · convert CHANGELOG `[Unreleased]` into `## [0.1.0-beta.1] — YYYY-MM-DD` (keep an empty Unreleased above) · update README's status line and python-api.md's "not yet published" rows · single-line commit `chore(release): stamp 0.1.0-beta.1`.
4. **Full verification**: [verification.md](verification.md) tiers 0–3 and 6 on the stamped commit; tier 4/5 walkthrough at least once on Linux **and** Windows. CI green on the commit.
5. **Build + inspect**: `python -m build` → `twine check dist/*` → install the wheel into a clean venv → `fasterrag config init && fasterrag doctor` → confirm the packaged `config.yaml` is byte-identical to the repo's.
6. **SBOM** (closes the TASK-0087 remainder): generate CycloneDX from `uv.lock` (e.g. `uv export` → converter), attach to the release artifacts.
7. **Tag**: annotated, single-line message — `git tag -a v0.1.0-beta.1 -m "fasterrag 0.1.0-beta.1"` → push the tag → CI green **on the tag**.
8. **Publish to TestPyPI first**: `twine upload --repository testpypi dist/*` → `pip install -i https://test.pypi.org/simple/ fasterrag==0.1.0b1` in a clean venv → smoke (`config init`, `doctor`, one ingest+query against a local Qdrant). Name collisions or metadata problems surface here, where they are cheap.
9. **Publish to PyPI**: `twine upload dist/*`. This is the irreversible moment — the version and its license grant are now permanent.
10. **GitHub release**: from the tag, body = the CHANGELOG section verbatim (no additional claims), SBOM + `dist/*` attached.
11. **Post-release verification**: on a machine that has never seen the repo — `pip install "fasterrag[huggingface]"` on Linux and Windows, then quickstart steps 0–6 from [how-to-use.md](how-to-use.md). Any failure here is a `0.1.0-beta.2` driver, filed in todo.md.
12. **Close the loop**: tick TASK-0020/TASK-0087 in todo.md with the date · re-derive release-readiness against the *next* milestone · start the new Unreleased section.

## 4. Rollback and bad-release handling

A published version is **yanked, never deleted** (`pip` will refuse it for new resolutions but existing pins keep working; deletion breaks pinned users and the number stays burned either way). Procedure: yank on PyPI → fix on `main` (git-revert playbook in [deployment.md](deployment.md)) → release the next patch/beta number → GitHub release note updated to point at it. Never re-upload different bytes under any version, and never rush the fix past §1 — a bad release does not lower the gate.

## 5. Cadence and support

Pre-1.0: only the latest release line receives fixes ([.github/SECURITY.md](../.github/SECURITY.md)). Betas ship when the gate passes, not on a calendar. Every release re-runs this document top to bottom; if a step was wrong or missing, fixing this file is part of the release.
