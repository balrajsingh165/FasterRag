"""The release pipeline cannot publish from a rehearsal, a branch, or a mismatched tag.

A PyPI version can never be reused, and nothing in this repository can run a GitHub
workflow before the day it matters. So the properties that keep `.github/workflows/release.yml`
from publishing by accident are pinned here, where every commit runs them: the decision
`.github/scripts/release.py` makes, and the shape of the workflow that acts on it.
"""

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"

PUBLISHING = {"publish-pypi", "publish-image", "github-release", "post-publish"}
GUARD = (
    "needs.verify.outputs.publish == 'true'",
    "startsWith(github.ref, 'refs/tags/v')",
    "(github.event_name == 'push' || inputs.dry_run == false)",
)

_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<target>\S+)(?P<rest>.*)$")
_PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
_VERSION_COMMENT = re.compile(r"^\s+# v\d+\.\d+\.\d+$")


def load_tool() -> ModuleType:
    """Import `.github/scripts/release.py`, which lives outside every package."""
    spec = importlib.util.spec_from_file_location(
        "release_tool", ROOT / ".github" / "scripts" / "release.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def release() -> dict[Any, Any]:
    """The release workflow as a document."""
    loaded: Any = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    """Return the `on:` mapping, which YAML 1.1 reads as the boolean key True."""
    found: Any = workflow.get("on", workflow.get(True))
    assert isinstance(found, dict)
    return found


@pytest.mark.parametrize(
    ("event", "ref", "dry_run", "version", "publish"),
    [
        ("push", "refs/tags/v0.1.0-beta.1", "", "0.1.0b1", "true"),
        ("push", "refs/tags/v1.0.0", "", "1.0.0", "true"),
        ("workflow_dispatch", "refs/tags/v0.1.0-beta.1", "false", "0.1.0b1", "true"),
        ("workflow_dispatch", "refs/tags/v0.1.0-beta.1", "true", "0.1.0b1", "false"),
        ("workflow_dispatch", "refs/heads/main", "true", "0.1.0.dev0", "false"),
    ],
)
def test_only_a_matching_tag_that_is_not_a_dry_run_publishes(
    event: str, ref: str, dry_run: str, version: str, publish: str
) -> None:
    """The whole decision: a tag that is the tree's version, pushed or dispatched for real."""
    assert tool.decide(event, ref, dry_run, version)["publish"] == publish


@pytest.mark.parametrize(
    ("event", "ref", "dry_run", "version", "reason"),
    [
        ("push", "refs/tags/v0.1.0-beta.2", "", "0.1.0b1", "__version__ is 0.1.0b1"),
        ("push", "refs/tags/v0.1.0-beta.1", "", "0.1.0.dev0", "__version__ is 0.1.0.dev0"),
        ("workflow_dispatch", "refs/tags/v0.1.0-beta.2", "true", "0.1.0b1", "__version__ is"),
        ("push", "refs/tags/v0.1.0b1", "", "0.1.0b1", "not a release tag"),
        ("push", "refs/tags/vnext", "", "0.1.0b1", "not a release tag"),
        ("push", "refs/heads/main", "", "0.1.0b1", "only a v* tag"),
        ("workflow_dispatch", "refs/heads/main", "false", "0.1.0b1", "dry_run is off"),
        ("workflow_dispatch", "refs/heads/main", "", "0.1.0b1", "must be true or false"),
        ("schedule", "refs/heads/main", "", "0.1.0b1", "does not run on"),
        ("pull_request", "refs/pull/7/merge", "false", "0.1.0b1", "does not run on"),
    ],
)
def test_everything_else_is_refused_before_a_later_job_starts(
    event: str, ref: str, dry_run: str, version: str, reason: str
) -> None:
    """A refusal is an exit, not a quiet dry run: the person who tagged must be told."""
    with pytest.raises(tool.RefusedError, match=re.escape(reason)):
        tool.decide(event, ref, dry_run, version)


def test_a_rehearsal_on_the_development_tree_is_labelled_with_its_own_version() -> None:
    """There is no tag to take a label from, and the files are named for `__version__`."""
    outputs = tool.decide("workflow_dispatch", "refs/heads/main", "true", "0.1.0.dev0")

    assert outputs == {
        "version": "0.1.0.dev0",
        "label": "0.1.0.dev0",
        "publish": "false",
        "prerelease": "true",
    }


def test_a_final_version_is_not_marked_as_a_prerelease() -> None:
    assert tool.decide("push", "refs/tags/v1.0.0", "", "1.0.0")["prerelease"] == "false"
    assert tool.decide("push", "refs/tags/v1.0.0-rc.1", "", "1.0.0rc1")["prerelease"] == "true"


def test_the_changelog_section_stops_at_the_next_version_heading() -> None:
    changelog = (
        "# Changelog\n\n## [Unreleased]\n\n"
        "## [0.2.0] — 2027-01-01\n\n- new\n\n## [0.1.0]\n\n- old\n"
    )

    assert tool.changelog_section(changelog, "0.2.0") == "- new\n"
    assert tool.changelog_section(changelog, "0.1.0") == "- old\n"
    with pytest.raises(tool.RefusedError, match=re.escape("no `## [0.3.0]` section")):
        tool.changelog_section(changelog, "0.3.0")


def test_a_section_that_fits_is_the_release_body_verbatim() -> None:
    assert tool.release_body("- a thing\n", "POINTER") == "- a thing\n"


def test_a_section_too_long_for_github_is_cut_at_a_paragraph_and_points_at_the_rest() -> None:
    """GitHub refuses a body over 125,000 characters; the first release's section is longer."""
    section = "### Added\n\n" + "- an entry.\n\n" * 20_000

    body = tool.release_body(section, "POINTER")

    assert len(section) > tool.BODY_LIMIT >= len(body)
    assert body.endswith("- an entry.\n\nPOINTER\n")
    assert section.startswith(body.removesuffix("POINTER\n"))


def test_a_cut_never_leaves_a_heading_with_nothing_under_it() -> None:
    section = "- kept\n\n### Changed\n\n" + "x" * (tool.BODY_LIMIT + 1)

    assert tool.release_body(section, "POINTER") == "- kept\n\nPOINTER\n"


def test_the_changelog_this_tree_would_release_produces_a_body_github_accepts() -> None:
    """Run against the real file, because its length is the reason the cut exists."""
    changelog = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = next(found.group("version") for found in tool._HEADING.finditer(changelog))

    body = tool.release_body(tool.changelog_section(changelog, heading), "POINTER")

    assert 0 < len(body) <= tool.BODY_LIMIT


def test_the_sbom_is_stamped_with_the_version_it_describes() -> None:
    """The exporter writes the project's own component without one, since the version is dynamic."""
    document: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "metadata": {"component": {"name": "fasterrag"}},
        "components": [{"name": "httpx", "version": "0.28.1"}],
    }

    assert tool.stamp_sbom(document, "0.1.0b1") == []
    assert document["metadata"]["component"]["version"] == "0.1.0b1"


def test_an_sbom_that_lists_nothing_or_pins_nothing_is_refused() -> None:
    empty = {"bomFormat": "CycloneDX", "metadata": {"component": {"name": "fasterrag"}}}
    unpinned = {**empty, "components": [{"name": "httpx"}]}

    assert tool.stamp_sbom(empty, "1.0.0") == ["the document lists no components"]
    assert tool.stamp_sbom(unpinned, "1.0.0") == ["components without a version: ['httpx']"]
    assert len(tool.stamp_sbom({"bomFormat": "SPDX"}, "1.0.0")) == 3


def test_the_workflow_runs_on_a_version_tag_or_by_hand_and_nothing_else() -> None:
    """A branch push, a schedule or a pull request must never start a release."""
    on = triggers(release())

    assert set(on) == {"push", "workflow_dispatch"}
    assert on["push"] == {"tags": ["v*"]}


def test_a_run_started_by_hand_is_a_dry_run_unless_somebody_says_otherwise() -> None:
    dry_run = triggers(release())["workflow_dispatch"]["inputs"]["dry_run"]

    assert dry_run["type"] == "boolean"
    assert dry_run["default"] is True


def test_every_publishing_job_carries_the_whole_guard() -> None:
    """Three redundant conditions: the script's decision, the ref, and the trigger."""
    jobs = release()["jobs"]

    assert set(jobs) >= PUBLISHING
    for name in PUBLISHING:
        condition = jobs[name]["if"]
        assert condition.startswith("${{ ")
        assert condition.removeprefix("${{ ").removesuffix(" }}") == " && ".join(GUARD), name
        assert "verify" in jobs[name]["needs"], name


def test_no_job_outside_the_guard_can_write_anything() -> None:
    """A dry run executes every unguarded job, so none of them may hold a write scope."""
    workflow = release()

    assert workflow["permissions"] == {}
    for name, job in workflow["jobs"].items():
        scopes = job.get("permissions")
        assert scopes is not None, f"{name} inherits permissions instead of stating them"
        if name not in PUBLISHING:
            assert set(scopes.values()) == {"read"}, name
            assert "environment" not in job, name


def test_nothing_is_uploaded_before_every_check_has_passed() -> None:
    """The PyPI job is the irreversible one; the other publishing jobs wait for it."""
    jobs = release()["jobs"]

    assert set(jobs["publish-pypi"]["needs"]) == {
        "verify",
        "gates",
        "build",
        "sbom",
        "smoke",
        "image",
    }
    for name in PUBLISHING - {"publish-pypi"}:
        assert "publish-pypi" in jobs[name]["needs"], name


def test_pypi_is_reached_by_trusted_publishing_behind_the_protected_environment() -> None:
    """No stored token: an OIDC exchange, in the one job that may ask for it."""
    jobs = release()["jobs"]
    publish = jobs["publish-pypi"]

    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"] == {"id-token": "write"}
    assert [name for name, job in jobs.items() if "id-token" in job["permissions"]] == [
        "publish-pypi"
    ]
    assert "secrets." not in RELEASE.read_text(encoding="utf-8")
    assert all("password" not in step.get("with", {}) for step in publish["steps"])


def test_the_gate_set_is_the_one_ci_runs_not_a_copy_of_it() -> None:
    gates = release()["jobs"]["gates"]
    ci: Any = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))

    assert gates["uses"] == "./.github/workflows/ci.yml"
    assert "workflow_call" in triggers(ci)


def test_ci_does_not_wait_for_a_release_run_on_main_that_can_never_exist() -> None:
    """`ci-status` fails a workflow with no completed run on the branch; release has none."""
    ci: Any = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    command = ci["jobs"]["branch-verdict"]["steps"][-1]["run"]

    assert "--exclude release.yml" in command
    assert "--exclude ci.yml" in command


def unpinned_actions(path: Path) -> list[str]:
    """Return every `uses:` line that is not a local path or a SHA with its version beside it."""
    problems = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        found = _USES.match(line)
        if found is None or found.group("target").startswith("./"):
            continue
        pinned = _PINNED.match(found.group("target")) and _VERSION_COMMENT.match(
            found.group("rest")
        )
        if not pinned:
            problems.append(f"{path.name}:{number}: {line.strip()}")
    return problems


def test_every_action_the_release_runs_is_pinned_to_a_commit() -> None:
    """A tag on somebody else's repository can be moved; a SHA cannot."""
    assert unpinned_actions(RELEASE) == []


def test_every_workflow_and_composite_action_pins_its_actions_to_a_commit() -> None:
    """One floating tag anywhere in the chain release -> ci.yml -> composite actions runs code."""
    files = [*WORKFLOWS.glob("*.yml"), *(ROOT / ".github" / "actions").glob("*/action.yml")]

    assert len(files) >= 6
    assert [problem for path in files for problem in unpinned_actions(path)] == []
