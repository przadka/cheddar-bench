"""Shared test fixtures for cheddar-bench tests."""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository structure for testing."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    # Create a minimal Python file
    (repo_path / "main.py").write_text("def hello():\n    return 'Hello, World!'\n")

    # Initialize as git repo (needed for agent compatibility)
    (repo_path / ".git").mkdir()

    return repo_path


@pytest.fixture
def sample_bug_data() -> dict[str, Any]:
    """Sample bug data for testing."""
    return {
        "file": "main.py",
        "line": 10,
        "type": "off-by-one",
        "description": "Loop iterates one extra time",
        "original": "for i in range(n):",
        "injected": "for i in range(n + 1):",
    }


@pytest.fixture
def sample_bugs_json() -> dict[str, Any]:
    """Sample bugs.json content for testing."""
    return {
        "bug_count": 2,
        "bugs": [
            {
                "file": "main.py",
                "line": 10,
                "type": "off-by-one",
                "description": "Loop iterates one extra time",
                "original": "for i in range(n):",
                "injected": "for i in range(n + 1):",
            },
            {
                "file": "utils.py",
                "line": 25,
                "type": "null-handling",
                "description": "Missing null check",
                "original": "return data.value",
                "injected": "return data.value  # removed null check",
            },
        ],
    }


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with sample templates."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()

    # Create sample prompt templates
    (prompts / "inject-bugs.md").write_text(
        "Inject bugs into the repository at {{repo_path}}.\n"
        "Output JSON with the following structure:\n"
        "```json\n"
        '{"loc": 0, "bug_count": 0, "bugs": []}\n'
        "```\n"
    )

    (prompts / "review-bugs.md").write_text(
        "Review the code at {{repo_path}} for bugs.\nOutput JSON with issues.\n"
    )

    # Prompt pairs for system/user split
    (prompts / "extract-manifest.system.md").write_text(
        "You are a code analysis assistant that reformats bug manifests.\n"
    )
    (prompts / "extract-manifest.user.md").write_text(
        "Reformat this manifest with {{expected_bug_count}} bugs.\n"
        "```json\n{{agent_bugs_json}}\n```\n"
        "```diff\n{{git_diff}}\n```\n"
    )
    (prompts / "match-all-bugs-report.system.md").write_text(
        "You match injected bugs to reviewer findings.\n"
    )
    (prompts / "match-all-bugs-report.user.md").write_text(
        "Injected bugs:\n{{bugs_json}}\nFindings:\n{{findings_json}}\n"
    )

    return prompts


@pytest.fixture
def challenges_dir(tmp_path: Path) -> Path:
    """Create a temporary challenges directory."""
    challenges = tmp_path / "challenges"
    challenges.mkdir()
    return challenges
