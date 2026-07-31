import check_commit_message as rule
import pytest


@pytest.mark.parametrize(
    "subject",
    [
        "feat: add hybrid retrieval",
        "fix: reject a baseline from another embedding model",
        "docs: record the eval harness in the changelog",
        "test: cover the degradation ladder",
        "refactor: move the client slot to the adapter base",
        "perf: batch the sparse encoder",
        "build: pin the qdrant image",
        "ci: run the contract suite on every pull request",
        "chore: tidy the scratch directory",
        "style: reformat the config schema",
        "revert: undo the sparse layout change",
    ],
)
def test_every_conventional_type_is_accepted(subject: str) -> None:
    assert rule.check_message(f"{subject}\n") == []


def test_a_scope_is_accepted() -> None:
    assert rule.check_message("feat(retrieval): add reciprocal rank fusion\n") == []


def test_a_breaking_change_marker_is_accepted() -> None:
    assert rule.check_message("feat!: change the adapter contract\n") == []
    assert rule.check_message("feat(adapters)!: change the contract\n") == []


def test_a_message_without_a_type_is_rejected() -> None:
    violations = rule.check_message("add hybrid retrieval\n")

    assert any("Conventional Commits type" in violation for violation in violations)


def test_an_unknown_type_is_rejected() -> None:
    assert rule.check_message("feature: add hybrid retrieval\n")


def test_a_type_without_a_description_is_rejected() -> None:
    assert rule.check_message("feat:\n")
    assert rule.check_message("feat\n")


def test_a_multi_line_message_is_rejected() -> None:
    violations = rule.check_message("feat: add the loader\n\nWith an explanation.\n")

    assert any("single line" in violation for violation in violations)


def test_trailers_are_rejected() -> None:
    assert rule.check_message("feat: add the loader\n\nSigned-off-by: Someone <a@b.c>\n")


def test_ai_attribution_is_rejected() -> None:
    assert rule.check_message("feat: add the loader\n\nCo-Authored-By: Someone <a@b.c>\n")
    assert rule.check_message("feat: add the loader (Generated with Claude Code)\n")


def test_an_over_long_subject_is_rejected() -> None:
    violations = rule.check_message(f"feat: {'x' * 120}\n")

    assert any("above the" in violation for violation in violations)


def test_an_empty_message_is_rejected() -> None:
    assert rule.check_message("\n\n# comment only\n")


def test_git_comment_lines_are_ignored() -> None:
    message = "feat: add the loader\n# Please enter the commit message for your changes.\n#\n"

    assert rule.check_message(message) == []


def test_model_identifiers_are_not_treated_as_attribution() -> None:
    assert rule.check_message("docs: document claude-opus-5 as an anthropic model id\n") == []
