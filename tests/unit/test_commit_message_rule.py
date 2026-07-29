import check_commit_message as rule


def test_single_line_message_passes() -> None:
    assert rule.check_message("Add config loader with fail-fast validation\n") == []


def test_multi_line_message_is_rejected() -> None:
    violations = rule.check_message("Add config loader\n\nWith a longer explanation.\n")
    assert any("single line" in violation for violation in violations)


def test_trailers_are_rejected() -> None:
    violations = rule.check_message("Add config loader\n\nSigned-off-by: Someone <a@b.c>\n")
    assert violations


def test_ai_attribution_is_rejected() -> None:
    assert rule.check_message("Add config loader\n\nCo-Authored-By: Someone <a@b.c>\n")
    assert rule.check_message("Add config loader (Generated with Claude Code)\n")


def test_empty_message_is_rejected() -> None:
    assert rule.check_message("\n\n# comment only\n")


def test_git_comment_lines_are_ignored() -> None:
    message = "Add config loader\n# Please enter the commit message for your changes.\n#\n"
    assert rule.check_message(message) == []


def test_model_identifiers_are_not_treated_as_attribution() -> None:
    assert rule.check_message("Document claude-opus-5 as an anthropic model id\n") == []
