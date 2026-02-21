"""Tests for prompt template loading."""

from pathlib import Path

import pytest


def test_load_prompt_success(prompts_dir: Path) -> None:
    """Test loading a valid prompt template."""
    from cheddar.prompts import load_prompt

    prompt = load_prompt("inject-bugs", prompts_dir=prompts_dir)
    assert "Inject bugs" in prompt
    assert "{{repo_path}}" in prompt


def test_load_prompt_with_substitution(prompts_dir: Path) -> None:
    """Test loading a prompt with variable substitution."""
    from cheddar.prompts import load_prompt

    prompt = load_prompt(
        "inject-bugs",
        prompts_dir=prompts_dir,
        variables={"repo_path": "/tmp/sandbox-123/repo"},
    )
    assert "/tmp/sandbox-123/repo" in prompt
    assert "{{repo_path}}" not in prompt


def test_load_prompt_missing_file(prompts_dir: Path) -> None:
    """Test loading a non-existent prompt template raises error."""
    from cheddar.prompts import PromptNotFoundError, load_prompt

    with pytest.raises(PromptNotFoundError) as exc_info:
        load_prompt("nonexistent-prompt", prompts_dir=prompts_dir)
    assert "nonexistent-prompt" in str(exc_info.value)


def test_load_prompt_missing_variable(prompts_dir: Path) -> None:
    """Test loading prompt without required variable keeps placeholder."""
    from cheddar.prompts import load_prompt

    # If variable not provided, placeholder should remain
    prompt = load_prompt("inject-bugs", prompts_dir=prompts_dir, variables={})
    assert "{{repo_path}}" in prompt


def test_load_prompt_extra_variables(prompts_dir: Path) -> None:
    """Test loading prompt with extra variables (should be ignored)."""
    from cheddar.prompts import load_prompt

    prompt = load_prompt(
        "inject-bugs",
        prompts_dir=prompts_dir,
        variables={
            "repo_path": "/tmp/test",
            "extra_var": "should be ignored",
        },
    )
    assert "/tmp/test" in prompt
    assert "extra_var" not in prompt
    assert "should be ignored" not in prompt


def test_get_available_prompts(prompts_dir: Path) -> None:
    """Test listing available prompt templates."""
    from cheddar.prompts import get_available_prompts

    prompts = get_available_prompts(prompts_dir=prompts_dir)
    assert "inject-bugs" in prompts
    assert "review-bugs" in prompts


def test_get_available_prompts_empty_dir(tmp_path: Path) -> None:
    """Test listing prompts in empty directory returns empty list."""
    from cheddar.prompts import get_available_prompts

    empty_prompts = tmp_path / "empty-prompts"
    empty_prompts.mkdir()
    prompts = get_available_prompts(prompts_dir=empty_prompts)
    assert prompts == []


def test_load_prompt_default_prompts_dir() -> None:
    """Test loading prompt uses default prompts directory."""
    from cheddar.prompts import get_prompts_dir

    # Test that get_prompts_dir returns a sensible default
    default_dir = get_prompts_dir()
    assert default_dir.name == "prompts"


def test_load_prompt_pair_returns_system_and_user(prompts_dir: Path) -> None:
    """Prompt pair returns two messages with correct roles."""
    from cheddar.prompts import load_prompt_pair

    messages = load_prompt_pair("extract-manifest", prompts_dir=prompts_dir)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "code analysis assistant" in messages[0]["content"]


def test_load_prompt_pair_substitutes_variables(prompts_dir: Path) -> None:
    """Variables are substituted in both system and user prompts."""
    from cheddar.prompts import load_prompt_pair

    messages = load_prompt_pair(
        "extract-manifest",
        prompts_dir=prompts_dir,
        variables={
            "expected_bug_count": "5",
            "agent_bugs_json": '{"bugs": []}',
            "git_diff": "--- a/foo.py\n+++ b/foo.py",
        },
    )
    assert "{{expected_bug_count}}" not in messages[1]["content"]
    assert "5" in messages[1]["content"]


def test_load_prompt_pair_missing_system(prompts_dir: Path) -> None:
    """Raises PromptNotFoundError when system file is missing."""
    from cheddar.prompts import PromptNotFoundError, load_prompt_pair

    with pytest.raises(PromptNotFoundError, match="nonexistent.system"):
        load_prompt_pair("nonexistent", prompts_dir=prompts_dir)


def test_load_prompt_pair_missing_user(prompts_dir: Path) -> None:
    """Raises PromptNotFoundError when user file is missing."""
    from cheddar.prompts import PromptNotFoundError, load_prompt_pair

    # Create only system file
    (prompts_dir / "orphan.system.md").write_text("system only\n")
    with pytest.raises(PromptNotFoundError, match="orphan.user"):
        load_prompt_pair("orphan", prompts_dir=prompts_dir)
