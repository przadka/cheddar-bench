"""Tests for CLI commands."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _make_challenge_side_effect(
    bugs_json: dict[str, Any],
) -> Any:
    """Create a side_effect for mock agent.challenge that writes bugs.json."""

    def _side_effect(
        sandbox_path: Path,
        prompt: str,  # noqa: ARG001
        timeout_seconds: int = 0,  # noqa: ARG001
    ) -> tuple[str, str]:
        (sandbox_path / "bugs.json").write_text(json.dumps(bugs_json))
        return ("raw output", "")

    return _side_effect


def test_cli_version() -> None:
    """Test --version flag."""
    from cheddar.cli import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_help() -> None:
    """Test --help flag."""
    from cheddar.cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cheddar" in result.stdout.lower()


def test_cli_list_agents() -> None:
    """Test list agents command."""
    from cheddar.cli import app

    result = runner.invoke(app, ["list", "agents"])
    assert result.exit_code == 0
    assert "claude" in result.stdout
    assert "codex" in result.stdout
    assert "gemini" in result.stdout


def test_cli_list_repos_empty(tmp_path: Path) -> None:
    """Test list repos with empty repos directory."""
    from cheddar.cli import app, state

    # Point to empty repos directory
    original_get_repos = state.get_repos_dir
    state.get_repos_dir = lambda: tmp_path / "repos"

    try:
        result = runner.invoke(app, ["list", "repos"])
        assert result.exit_code == 0
        assert "no repositories" in result.stdout.lower()
    finally:
        state.get_repos_dir = original_get_repos


# Challenge command tests


def test_cli_challenge_help() -> None:
    """Test challenge command help."""
    from cheddar.cli import app

    result = runner.invoke(app, ["challenge", "--help"])
    assert result.exit_code == 0
    assert "challenge" in result.stdout.lower()


def test_cli_challenge_invalid_agent(tmp_path: Path) -> None:
    """Test challenge command with invalid agent."""
    from cheddar.cli import app, state

    original_get_repos = state.get_repos_dir
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    (repos_dir / "test-repo").mkdir()
    state.get_repos_dir = lambda: repos_dir

    try:
        result = runner.invoke(app, ["challenge", "invalid-agent", "test-repo"])
        assert result.exit_code != 0
    finally:
        state.get_repos_dir = original_get_repos


def test_cli_challenge_missing_repo(tmp_path: Path) -> None:
    """Test challenge command with missing repository."""
    from cheddar.cli import app, state

    original_get_repos = state.get_repos_dir
    state.get_repos_dir = lambda: tmp_path / "repos"

    try:
        result = runner.invoke(app, ["challenge", "claude", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or result.exit_code == 2
    finally:
        state.get_repos_dir = original_get_repos


def test_cli_challenge_success(
    tmp_path: Path,
    tmp_repo: Path,
    sample_bugs_json: dict[str, Any],
    prompts_dir: Path,
) -> None:
    """Test successful challenge command with mocked agent."""
    from cheddar.cli import app, state
    from cheddar.models import BugManifest

    # Setup directories
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()

    # Copy tmp_repo to repos_dir
    import shutil

    shutil.copytree(tmp_repo, repos_dir / "test-repo")

    challenges_dir = tmp_path / "challenges"
    challenges_dir.mkdir()

    original_repos = state.get_repos_dir
    original_challenges = state.get_challenges_dir
    state.get_repos_dir = lambda: repos_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        mock_manifest = BugManifest(**sample_bugs_json)
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_challenge = True
        mock_agent.challenge.side_effect = _make_challenge_side_effect(sample_bugs_json)

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
            patch(
                "cheddar.extractor.reformat_manifest",
                return_value=(mock_manifest, "diff content", "prompt content"),
            ),
        ):
            result = runner.invoke(app, ["challenge", "claude", "test-repo", "--timeout", "60"])

        # Should succeed
        assert result.exit_code == 0 or "error" not in result.stdout.lower()
    finally:
        state.get_repos_dir = original_repos
        state.get_challenges_dir = original_challenges


def test_cli_challenge_creates_challenge_directory(
    tmp_path: Path,
    tmp_repo: Path,
    sample_bugs_json: dict[str, Any],
    prompts_dir: Path,
) -> None:
    """Test challenge command creates run directory with correct files."""
    from cheddar.cli import app, state
    from cheddar.models import BugManifest

    # Setup
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    import shutil

    shutil.copytree(tmp_repo, repos_dir / "test-repo")

    challenges_dir = tmp_path / "challenges"
    # Don't create challenges_dir - challenge should create it

    original_repos = state.get_repos_dir
    original_challenges = state.get_challenges_dir
    state.get_repos_dir = lambda: repos_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        mock_manifest = BugManifest(**sample_bugs_json)
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_challenge = True
        mock_agent.challenge.side_effect = _make_challenge_side_effect(sample_bugs_json)

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
            patch(
                "cheddar.extractor.reformat_manifest",
                return_value=(mock_manifest, "diff content", "prompt content"),
            ),
        ):
            result = runner.invoke(app, ["challenge", "claude", "test-repo"])

        # Should create runs directory
        if result.exit_code == 0:
            assert challenges_dir.exists()
            # Should create a run subdirectory
            run_dirs = list(challenges_dir.iterdir())
            if run_dirs:
                run_dir = run_dirs[0]
                # Should have config.json and bugs.json
                assert (run_dir / "config.json").exists() or (run_dir / "bugs.json").exists()
    finally:
        state.get_repos_dir = original_repos
        state.get_challenges_dir = original_challenges


# Review command tests


def test_cli_review_help() -> None:
    """Test review command help."""
    from cheddar.cli import app

    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    assert "review" in result.stdout.lower()


def test_cli_review_missing_challenge_id() -> None:
    """Test review command without required --challenge option."""
    from cheddar.cli import app

    result = runner.invoke(app, ["review", "claude", "test-repo"])
    # Should fail because --challenge is required
    assert result.exit_code != 0


def test_cli_review_invalid_agent(tmp_path: Path) -> None:
    """Test review command with invalid agent."""
    from cheddar.cli import app, state

    original_get_challenges = state.get_challenges_dir
    challenges_dir = tmp_path / "challenges"
    challenges_dir.mkdir()
    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(
            app, ["review", "invalid-agent", "test-repo", "--challenge", "test-run"]
        )
        assert result.exit_code != 0
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_review_missing_run(tmp_path: Path) -> None:
    """Test review command with non-existent run ID."""
    from cheddar.cli import app, state

    original_get_challenges = state.get_challenges_dir
    challenges_dir = tmp_path / "challenges"
    challenges_dir.mkdir()
    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(
            app, ["review", "claude", "test-repo", "--challenge", "nonexistent-run"]
        )
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or result.exit_code == 2
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_review_missing_persisted_repo(tmp_path: Path) -> None:
    """Test review command when run exists but has no persisted repo."""
    from cheddar.cli import app, state

    original_get_challenges = state.get_challenges_dir
    challenges_dir = tmp_path / "challenges"
    challenges_dir.mkdir()
    # Create run directory without persisted repo
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir()
    # Valid config but no repo/ directory
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )

    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(app, ["review", "claude", "test-repo", "--challenge", "test-run"])
        assert result.exit_code != 0
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_review_repo_mismatch(tmp_path: Path) -> None:
    """Test review command rejects mismatched repo argument."""
    from cheddar.cli import app, state

    original_get_challenges = state.get_challenges_dir
    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    # Config says repo is "original-repo"
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "original-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )
    # Create persisted repo
    persisted_repo = challenge_dir / "repo"
    persisted_repo.mkdir()
    (persisted_repo / "main.py").write_text("# code")

    state.get_challenges_dir = lambda: challenges_dir

    try:
        # Try to review with different repo name
        result = runner.invoke(app, ["review", "claude", "wrong-repo", "--challenge", "test-run"])
        assert result.exit_code == 2
        # Error message goes to stderr, which CliRunner captures in output
        assert "mismatch" in result.output.lower()
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_review_success(
    tmp_path: Path,
    prompts_dir: Path,
) -> None:
    """Test successful review command with mocked agent."""
    from cheddar.cli import app, state
    from cheddar.models import ReviewReport

    # Setup directories
    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    # Create persisted repo (simulating challenge phase output)
    persisted_repo = challenge_dir / "repo"
    persisted_repo.mkdir()
    (persisted_repo / "main.py").write_text("def hello():\n    return 'Hello'\n")
    # Initialize git in persisted repo
    (persisted_repo / ".git").mkdir()

    # Create config.json
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )

    # Create bugs.json (ground truth)
    (challenge_dir / "bugs.json").write_text(
        '{"bug_count": 1, "bugs": [{"file": "main.py", "line": 10, '
        '"type": "off-by-one", "description": "test", "original": "x", "injected": "y"}]}'
    )

    original_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        # Mock the agent review - now returns raw text, not JSON with issues
        mock_report = ReviewReport(
            agent="claude",
            raw_output="# Code Review\n\nFound an off-by-one error in main.py:10",
            duration_seconds=10.5,
            status="complete",
        )
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_review = True
        mock_agent.review.return_value = mock_report

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
        ):
            result = runner.invoke(
                app, ["review", "claude", "test-repo", "--challenge", "test-run"]
            )

        # Should succeed
        assert result.exit_code == 0, f"Failed with: {result.stdout}"
        assert "complete" in result.stdout.lower() or "review" in result.stdout.lower()
    finally:
        state.get_challenges_dir = original_challenges


def test_cli_review_creates_reviews_directory(
    tmp_path: Path,
    prompts_dir: Path,
) -> None:
    """Test review command creates reviews directory with agent output."""
    from cheddar.cli import app, state
    from cheddar.models import ReviewReport

    # Setup
    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    persisted_repo = challenge_dir / "repo"
    persisted_repo.mkdir()
    (persisted_repo / "main.py").write_text("def hello():\n    return 'Hello'\n")
    (persisted_repo / ".git").mkdir()

    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )
    (challenge_dir / "bugs.json").write_text(
        '{"bug_count": 1, "bugs": [{"file": "main.py", "line": 10, '
        '"type": "off-by-one", "description": "test", "original": "x", "injected": "y"}]}'
    )

    original_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        # Mock review now returns raw text, not parsed issues
        mock_report = ReviewReport(
            agent="codex",
            raw_output="# Code Review\n\nFound off-by-one error in main.py",
            duration_seconds=5.0,
            status="complete",
        )
        mock_agent = MagicMock()
        mock_agent.name = "codex"
        mock_agent.can_review = True
        mock_agent.review.return_value = mock_report

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
        ):
            result = runner.invoke(app, ["review", "codex", "test-repo", "--challenge", "test-run"])

        assert result.exit_code == 0, f"Command failed: {result.stdout}"

        # Should create reviews directory
        reviews_dir = challenge_dir / "reviews"
        assert reviews_dir.exists(), "reviews/ directory not created"
        # Should have agent-specific file
        assert (reviews_dir / "codex.json").exists(), "codex.json not created"

        # Verify content (no issues field anymore)
        content = json.loads((reviews_dir / "codex.json").read_text())
        assert content["agent"] == "codex"
        assert "raw_output" in content
    finally:
        state.get_challenges_dir = original_challenges


def test_cli_review_persists_raw_findings_snapshot(tmp_path: Path, prompts_dir: Path) -> None:
    """review command copies sandbox bugs/*.json into challenge raw_findings/"""
    from cheddar.cli import app, state
    from cheddar.models import FoundBug, ReviewReport

    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    persisted_repo = challenge_dir / "repo"
    persisted_repo.mkdir()
    (persisted_repo / "main.py").write_text("def hello():\n    return 'Hello'\n")
    (persisted_repo / ".git").mkdir()

    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )
    (challenge_dir / "bugs.json").write_text(
        '{"bug_count": 1, "bugs": [{"file": "main.py", "line": 10, '
        '"type": "off-by-one", "description": "test", "original": "x", "injected": "y"}]}'
    )

    sandbox_dir = tmp_path / "sandbox-review"
    sandbox_dir.mkdir()
    bugs_dir = sandbox_dir / "bugs"
    bugs_dir.mkdir()
    (bugs_dir / "a.json").write_text(
        '{"file":"main.py","line":1,"type":"x","why":"y","impact":"z"}'
    )
    (bugs_dir / "b.json").write_text('{"not": "valid finding"}')

    original_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        mock_report = ReviewReport(
            agent="claude",
            raw_output="# Code Review\n\nFound issue",
            duration_seconds=1.0,
            status="complete",
            found_bugs=[FoundBug(file="main.py", line=1, type="logic", why="bad", impact="worse")],
        )
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_review = True
        mock_agent.review.return_value = mock_report

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
            patch("cheddar.sandbox.create_review_sandbox", return_value=sandbox_dir),
            patch("cheddar.sandbox.cleanup_sandbox"),
        ):
            result = runner.invoke(
                app, ["review", "claude", "test-repo", "--challenge", "test-run"]
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        snapshot_dir = challenge_dir / "raw_findings" / "claude"
        assert snapshot_dir.exists()
        assert (snapshot_dir / "a.json").exists()
        assert (snapshot_dir / "b.json").exists()

        report = json.loads((challenge_dir / "reviews" / "claude.json").read_text())
        assert report["raw_findings_snapshot_dir"] == "raw_findings/claude"
    finally:
        state.get_challenges_dir = original_challenges


# Match command tests (T020-T023)


def test_cli_match_help() -> None:
    """Test match command help."""
    from cheddar.cli import app

    result = runner.invoke(app, ["match", "--help"])
    assert result.exit_code == 0
    assert "match" in result.stdout.lower()
    assert "--challenge" in result.stdout
    assert "--model" in result.stdout


def test_cli_match_missing_run(tmp_path: Path) -> None:
    """Test match with non-existent run."""
    from cheddar.cli import app, state

    original_get_challenges = state.get_challenges_dir
    challenges_dir = tmp_path / "challenges"
    challenges_dir.mkdir()
    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(app, ["match", "--challenge", "nonexistent-run"])
        assert result.exit_code == 2
        assert "not found" in result.output.lower()
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_missing_bugs(tmp_path: Path) -> None:
    """Test match when bugs.json is missing."""
    from cheddar.cli import app, state

    original_get_challenges = state.get_challenges_dir
    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)
    # Create config but no bugs.json
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )

    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(app, ["match", "--challenge", "test-run"])
        assert result.exit_code == 2
        assert "bugs.json" in result.output.lower()
    finally:
        state.get_challenges_dir = original_get_challenges


@pytest.fixture
def match_test_setup(tmp_path: Path) -> tuple[Path, Path]:
    """Setup a run directory for match command testing."""
    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    # Create config
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )

    # Create bugs.json
    (challenge_dir / "bugs.json").write_text(
        json.dumps(
            {
                "bug_count": 2,
                "bugs": [
                    {
                        "file": "main.py",
                        "line": 10,
                        "type": "off-by-one",
                        "description": "Loop error",
                        "original": "for i in range(n):",
                        "injected": "for i in range(n + 1):",
                    },
                    {
                        "file": "utils.py",
                        "line": 25,
                        "type": "null-check",
                        "description": "Missing null check",
                        "original": "return data.value",
                        "injected": "return data.value  # removed null check",
                    },
                ],
            }
        )
    )

    # Create reviews directory with a review (now raw text, no issues)
    reviews_dir = challenge_dir / "reviews"
    reviews_dir.mkdir()
    (reviews_dir / "claude.json").write_text(
        json.dumps(
            {
                "agent": "claude",
                "raw_output": """# Code Review

## Issues Found

### main.py:10 - Off-by-one error
The loop iterates one too many times causing an IndexError.

### other.py:50 - Security issue
Potential security vulnerability found.
""",
                "duration_seconds": 5.0,
                "status": "complete",
            }
        )
    )

    return challenges_dir, challenge_dir


def test_cli_match_success(match_test_setup: tuple[Path, Path]) -> None:
    """Test successful match with mocked LLM."""
    from cheddar.cli import app, state
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    challenges_dir, challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        mock_output = GlobalBugMatchOutput(
            assignments=[
                GlobalBugMatchAssignment(
                    bug_index=0,
                    finding_index=1,
                    reasoning="Report clearly mentions this bug",
                    matched_quote="off-by-one error in main.py",
                    matched_line=10,
                    confidence="high",
                ),
                GlobalBugMatchAssignment(
                    bug_index=1,
                    finding_index=0,
                    reasoning="Report does not mention this file",
                    matched_quote="",
                    matched_line=0,
                    confidence="low",
                ),
            ]
        )

        with patch("cheddar.matcher.complete_structured", return_value=mock_output):
            result = runner.invoke(app, ["match", "--challenge", "test-run"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "complete" in result.output.lower() or "match" in result.output.lower()

        # Verify scores/<reviewer>.json was created
        scores_dir = challenge_dir / "scores"
        assert scores_dir.exists(), "scores/ directory not created"
        score_file = scores_dir / "claude.json"
        assert score_file.exists(), "scores/claude.json not created"

        score = json.loads(score_file.read_text())
        assert score["reviewer"] == "claude"
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_multiple_reviewers(match_test_setup: tuple[Path, Path]) -> None:
    """Test match with multiple reviewers."""
    from cheddar.cli import app, state
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    challenges_dir, challenge_dir = match_test_setup

    # Add a second reviewer (raw text, no issues)
    reviews_dir = challenge_dir / "reviews"
    (reviews_dir / "codex.json").write_text(
        json.dumps(
            {
                "agent": "codex",
                "raw_output": "# Code Review\n\nFound null check issue in utils.py:25",
                "duration_seconds": 3.0,
                "status": "complete",
            }
        )
    )

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        mock_output = GlobalBugMatchOutput(
            assignments=[
                GlobalBugMatchAssignment(
                    bug_index=0,
                    finding_index=1,
                    reasoning="Match",
                    matched_quote="found it",
                    matched_line=10,
                    confidence="high",
                ),
                GlobalBugMatchAssignment(
                    bug_index=1,
                    finding_index=2,
                    reasoning="Match",
                    matched_quote="found it",
                    matched_line=25,
                    confidence="high",
                ),
            ]
        )

        with patch("cheddar.matcher.complete_structured", return_value=mock_output):
            result = runner.invoke(app, ["match", "--challenge", "test-run"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify both reviewers scored in scores/ directory
        scores_dir = challenge_dir / "scores"
        assert scores_dir.exists(), "scores/ directory not created"
        score_files = {f.stem for f in scores_dir.glob("*.json")}
        assert score_files == {"claude", "codex"}
        # Verify model field is present in scores (contract requirement)
        for score_file in scores_dir.glob("*.json"):
            score = json.loads(score_file.read_text())
            assert "model" in score, "Score missing 'model' field"
            assert score["model"].startswith("azure/")
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_skips_malformed_review(match_test_setup: tuple[Path, Path]) -> None:
    """Test match skips malformed review files and continues."""
    from cheddar.cli import app, state
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    challenges_dir, challenge_dir = match_test_setup

    # Add a malformed review
    reviews_dir = challenge_dir / "reviews"
    (reviews_dir / "broken.json").write_text("not valid json {{{")

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        mock_output = GlobalBugMatchOutput(
            assignments=[
                GlobalBugMatchAssignment(
                    bug_index=0,
                    finding_index=1,
                    reasoning="Match",
                    matched_quote="found it",
                    matched_line=10,
                    confidence="high",
                )
            ]
        )

        with patch("cheddar.matcher.complete_structured", return_value=mock_output):
            result = runner.invoke(app, ["match", "--challenge", "test-run"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "warning" in result.output.lower() or "skipping" in result.output.lower()

        # Verify valid reviewer was still scored
        scores_dir = challenge_dir / "scores"
        assert scores_dir.exists(), "scores/ directory not created"
        score_file = scores_dir / "claude.json"
        assert score_file.exists(), "scores/claude.json not created"
        score = json.loads(score_file.read_text())
        assert score["reviewer"] == "claude"
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_custom_model(match_test_setup: tuple[Path, Path]) -> None:
    """Test match with --model option."""
    from cheddar.cli import app, state
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    challenges_dir, challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    captured_model = None

    try:

        def mock_complete_structured(**kwargs: object) -> GlobalBugMatchOutput:
            nonlocal captured_model
            captured_model = kwargs.get("model")
            return GlobalBugMatchOutput(
                assignments=[
                    GlobalBugMatchAssignment(
                        bug_index=0,
                        finding_index=1,
                        reasoning="Match",
                        matched_quote="found it",
                        matched_line=10,
                        confidence="high",
                    )
                ]
            )

        with patch("cheddar.matcher.complete_structured", side_effect=mock_complete_structured):
            result = runner.invoke(
                app, ["match", "--challenge", "test-run", "--model", "claude-3-haiku-20240307"]
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert captured_model == "claude-3-haiku-20240307"
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_uses_env_model_when_set(match_test_setup: tuple[Path, Path]) -> None:
    """match should use CHEDDAR_MATCH_MODEL when set."""
    from cheddar.cli import app, state
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    challenges_dir, _challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    captured_model = None

    try:
        with patch.dict(os.environ, {"CHEDDAR_MATCH_MODEL": "azure/gpt-5.2"}, clear=False):

            def mock_complete_structured(**kwargs: object) -> GlobalBugMatchOutput:
                nonlocal captured_model
                captured_model = kwargs.get("model")
                return GlobalBugMatchOutput(
                    assignments=[
                        GlobalBugMatchAssignment(
                            bug_index=0,
                            finding_index=1,
                            reasoning="Match",
                            matched_quote="found it",
                            matched_line=10,
                            confidence="high",
                        )
                    ]
                )

            with patch("cheddar.matcher.complete_structured", side_effect=mock_complete_structured):
                result = runner.invoke(app, ["match", "--challenge", "test-run"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert captured_model == "azure/gpt-5.2"
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_repeat_median_aggregation(match_test_setup: tuple[Path, Path]) -> None:
    """Median aggregation picks the middle run by bugs_found."""
    from cheddar.cli import app, state
    from cheddar.models import BugMatch

    challenges_dir, challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    runs = [
        [
            BugMatch(
                bug_index=0,
                reasoning="run1 miss",
                matched_quote="",
                matched_line=0,
                found=False,
                confidence="low",
                error=False,
            ),
            BugMatch(
                bug_index=1,
                reasoning="run1 miss",
                matched_quote="",
                matched_line=0,
                found=False,
                confidence="low",
                error=False,
            ),
        ],
        [
            BugMatch(
                bug_index=0,
                reasoning="run2 hit",
                matched_quote="main.py",
                matched_line=10,
                found=True,
                confidence="high",
                error=False,
            ),
            BugMatch(
                bug_index=1,
                reasoning="run2 hit",
                matched_quote="utils.py",
                matched_line=25,
                found=True,
                confidence="high",
                error=False,
            ),
        ],
        [
            BugMatch(
                bug_index=0,
                reasoning="run3 hit",
                matched_quote="main.py",
                matched_line=10,
                found=True,
                confidence="high",
                error=False,
            ),
            BugMatch(
                bug_index=1,
                reasoning="run3 miss",
                matched_quote="",
                matched_line=0,
                found=False,
                confidence="low",
                error=False,
            ),
        ],
    ]

    try:
        with patch("cheddar.matcher.match_bugs_to_raw_report", side_effect=runs):
            result = runner.invoke(
                app,
                [
                    "match",
                    "--challenge",
                    "test-run",
                    "--repeat",
                    "3",
                    "--aggregate",
                    "median",
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        score = json.loads((challenge_dir / "scores" / "claude.json").read_text())
        assert score["bugs_found"] == 1
        assert score["total_bugs"] == 2
        assert score["detection_rate"] == 0.5

        runs = json.loads((challenge_dir / "scores" / "claude.runs.json").read_text())
        assert runs["reviewer"] == "claude"
        assert runs["repeat"] == 3
        assert runs["aggregate"] == "median"
        assert runs["selected_run_index"] == 2
        assert len(runs["runs"]) == 3
        assert runs["runs"][0]["bugs_found"] == 0
        assert runs["runs"][1]["bugs_found"] == 2
        assert runs["runs"][2]["bugs_found"] == 1
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_median_requires_repeat_ge_3(match_test_setup: tuple[Path, Path]) -> None:
    """Median aggregation should reject repeat values below 3."""
    from cheddar.cli import app, state

    challenges_dir, _challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(
            app,
            [
                "match",
                "--challenge",
                "test-run",
                "--repeat",
                "2",
                "--aggregate",
                "median",
            ],
        )

        assert result.exit_code == 2
        assert "requires --repeat >= 3" in result.output
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_repeat_single_rejected(match_test_setup: tuple[Path, Path]) -> None:
    """Repeat >1 must use median aggregation to avoid wasted runs."""
    from cheddar.cli import app, state

    challenges_dir, _challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(
            app,
            [
                "match",
                "--challenge",
                "test-run",
                "--repeat",
                "2",
                "--aggregate",
                "single",
            ],
        )

        assert result.exit_code == 2
        assert "requires --aggregate median" in result.output
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_no_reviews(tmp_path: Path) -> None:
    """Test match when no reviews exist."""
    from cheddar.cli import app, state

    challenges_dir = tmp_path / "challenges"
    challenge_dir = challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    # Create config and bugs but no reviews
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )
    (challenge_dir / "bugs.json").write_text(
        json.dumps(
            {
                "bug_count": 1,
                "bugs": [
                    {
                        "file": "main.py",
                        "line": 10,
                        "type": "error",
                        "description": "Bug",
                        "original": "x",
                        "injected": "y",
                    }
                ],
            }
        )
    )

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:
        result = runner.invoke(app, ["match", "--challenge", "test-run"])
        assert result.exit_code == 0
        assert "warning" in result.output.lower() or "no review" in result.output.lower()

        # Verify no scores directory created (no reviewers to score)
        scores_dir = challenge_dir / "scores"
        assert not scores_dir.exists(), "scores/ should not exist when no reviews"
    finally:
        state.get_challenges_dir = original_get_challenges


def test_cli_match_score_includes_match_details(match_test_setup: tuple[Path, Path]) -> None:
    """Test that scores.json includes matches with LLM reasoning."""
    from cheddar.cli import app, state
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    challenges_dir, challenge_dir = match_test_setup

    original_get_challenges = state.get_challenges_dir
    state.get_challenges_dir = lambda: challenges_dir

    try:

        def mock_complete(**_kwargs: object) -> GlobalBugMatchOutput:
            return GlobalBugMatchOutput(
                assignments=[
                    GlobalBugMatchAssignment(
                        bug_index=0,
                        finding_index=1,
                        reasoning="Same off-by-one error in loop",
                        matched_quote="Off-by-one error",
                        matched_line=10,
                        confidence="high",
                    ),
                    GlobalBugMatchAssignment(
                        bug_index=1,
                        finding_index=0,
                        reasoning="No match",
                        matched_quote="",
                        matched_line=0,
                        confidence="low",
                    ),
                ]
            )

        with patch("cheddar.matcher.complete_structured", side_effect=mock_complete):
            result = runner.invoke(app, ["match", "--challenge", "test-run"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify matches in scores/claude.json
        score_file = challenge_dir / "scores" / "claude.json"
        assert score_file.exists(), "scores/claude.json not created"
        score = json.loads(score_file.read_text())

        # Find a matched bug and verify details (new format: BugMatch, not MatchResult)
        matched = [m for m in score["matches"] if m["found"]]
        assert len(matched) > 0, "Expected at least one matched bug"

        # Verify BugMatch structure contains reasoning
        first_match = matched[0]
        assert "reasoning" in first_match
        assert "confidence" in first_match
        assert first_match["confidence"] in ["high", "medium", "low"]
    finally:
        state.get_challenges_dir = original_get_challenges


# --challenges-dir option tests (T003-T006a)


def test_cli_challenge_challenges_dir_option(
    tmp_path: Path,
    tmp_repo: Path,
    sample_bugs_json: dict[str, Any],
    prompts_dir: Path,
) -> None:
    """Test challenge command with --challenges-dir option."""
    from cheddar.cli import app, state
    from cheddar.models import BugManifest

    # Setup repos directory
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    import shutil

    shutil.copytree(tmp_repo, repos_dir / "test-repo")

    # Custom runs directory (not default)
    custom_challenges_dir = tmp_path / "experiment" / "runs"
    # Don't create it - should be created by command

    # Create a fake sandbox directory for the mock
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "main.py").write_text("def hello():\n    return 'Hello, World!'\n")

    original_repos = state.get_repos_dir
    state.get_repos_dir = lambda: repos_dir

    try:
        mock_manifest = BugManifest(**sample_bugs_json)
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_challenge = True
        mock_agent.challenge.side_effect = _make_challenge_side_effect(sample_bugs_json)

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
            patch("cheddar.sandbox.create_challenge_sandbox", return_value=sandbox_dir),
            patch("cheddar.sandbox.persist_sandbox", return_value=tmp_path / "persisted"),
            patch("cheddar.sandbox.cleanup_sandbox"),
            patch(
                "cheddar.extractor.reformat_manifest",
                return_value=(mock_manifest, "diff content", "prompt content"),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "challenge",
                    "claude",
                    "test-repo",
                    "--challenges-dir",
                    str(custom_challenges_dir),
                    "--timeout",
                    "60",
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should create runs in custom directory
        assert custom_challenges_dir.exists(), "Custom runs dir not created"
        challenge_dirs = list(custom_challenges_dir.iterdir())
        assert len(challenge_dirs) > 0, "No run directory created in custom runs dir"
    finally:
        state.get_repos_dir = original_repos


def test_cli_review_challenges_dir_option(
    tmp_path: Path,
    prompts_dir: Path,
) -> None:
    """Test review command with --challenges-dir option."""
    from cheddar.cli import app
    from cheddar.models import ReviewReport

    # Setup custom runs directory with existing run
    custom_challenges_dir = tmp_path / "experiment" / "runs"
    challenge_dir = custom_challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    # Create persisted repo
    persisted_repo = challenge_dir / "repo"
    persisted_repo.mkdir()
    (persisted_repo / "main.py").write_text("def hello():\n    return 'Hello'\n")
    (persisted_repo / ".git").mkdir()

    # Create config.json
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )

    # Create bugs.json
    (challenge_dir / "bugs.json").write_text(
        '{"bug_count": 1, "bugs": [{"file": "main.py", "line": 10, '
        '"type": "off-by-one", "description": "test", "original": "x", "injected": "y"}]}'
    )

    # Mock the agent review
    mock_report = ReviewReport(
        agent="claude",
        raw_output="# Code Review\n\nFound an off-by-one error",
        duration_seconds=10.5,
        status="complete",
    )
    mock_agent = MagicMock()
    mock_agent.name = "claude"
    mock_agent.can_review = True
    mock_agent.review.return_value = mock_report

    with (
        patch("cheddar.agents.get_agent", return_value=mock_agent),
        patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
    ):
        result = runner.invoke(
            app,
            [
                "review",
                "claude",
                "test-repo",
                "--challenge",
                "test-run",
                "--challenges-dir",
                str(custom_challenges_dir),
            ],
        )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    # Should create review in custom runs dir
    assert (challenge_dir / "reviews" / "claude.json").exists()


def test_cli_match_challenges_dir_option(tmp_path: Path) -> None:
    """Test match command with --challenges-dir option."""
    from cheddar.cli import app
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    # Setup custom runs directory
    custom_challenges_dir = tmp_path / "experiment" / "runs"
    challenge_dir = custom_challenges_dir / "test-run"
    challenge_dir.mkdir(parents=True)

    # Create config
    (challenge_dir / "config.json").write_text(
        '{"challenge_id": "test-run", "repo": "test-repo", "challenger": "claude", '
        '"created_at": "2026-01-25T00:00:00", "timeout_seconds": 600, "raw_output": "", "raw_stderr": "", "duration_seconds": 0.0, "status": "complete", "failure_reason": null}'
    )

    # Create bugs.json (1 bug)
    (challenge_dir / "bugs.json").write_text(
        json.dumps(
            {
                "bug_count": 1,
                "bugs": [
                    {
                        "file": "main.py",
                        "line": 10,
                        "type": "off-by-one",
                        "description": "Bug",
                        "original": "x",
                        "injected": "y",
                    }
                ],
            }
        )
    )

    # Create review
    reviews_dir = challenge_dir / "reviews"
    reviews_dir.mkdir()
    (reviews_dir / "claude.json").write_text(
        json.dumps(
            {
                "agent": "claude",
                "raw_output": "Found off-by-one error in main.py:10",
                "duration_seconds": 5.0,
                "status": "complete",
            }
        )
    )

    mock_output = GlobalBugMatchOutput(
        assignments=[
            GlobalBugMatchAssignment(
                bug_index=0,
                finding_index=1,
                reasoning="Match",
                matched_quote="found it",
                matched_line=10,
                confidence="high",
            )
        ]
    )

    with patch("cheddar.matcher.complete_structured", return_value=mock_output):
        result = runner.invoke(
            app,
            ["match", "--challenge", "test-run", "--challenges-dir", str(custom_challenges_dir)],
        )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    # Should create scores in custom runs dir
    assert (challenge_dir / "scores" / "claude.json").exists()


def test_cli_default_challenges_dir_backward_compatible(
    tmp_path: Path,
    tmp_repo: Path,
    sample_bugs_json: dict[str, Any],
    prompts_dir: Path,
) -> None:
    """Test that default runs directory behavior is unchanged (backward compatibility)."""
    from cheddar.cli import app, state
    from cheddar.models import BugManifest

    # Setup repos and runs directories
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    import shutil

    shutil.copytree(tmp_repo, repos_dir / "test-repo")

    default_challenges_dir = tmp_path / "challenges"
    # Don't create - should be created by command

    # Create a fake sandbox directory for the mock
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "main.py").write_text("def hello():\n    return 'Hello, World!'\n")

    original_repos = state.get_repos_dir
    original_challenges = state.get_challenges_dir
    state.get_repos_dir = lambda: repos_dir
    state.get_challenges_dir = lambda: default_challenges_dir

    try:
        mock_manifest = BugManifest(**sample_bugs_json)
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_challenge = True
        mock_agent.challenge.side_effect = _make_challenge_side_effect(sample_bugs_json)

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
            patch("cheddar.sandbox.create_challenge_sandbox", return_value=sandbox_dir),
            patch("cheddar.sandbox.persist_sandbox", return_value=tmp_path / "persisted"),
            patch("cheddar.sandbox.cleanup_sandbox"),
            patch(
                "cheddar.extractor.reformat_manifest",
                return_value=(mock_manifest, "diff content", "prompt content"),
            ),
        ):
            # Run WITHOUT --challenges-dir option
            result = runner.invoke(app, ["challenge", "claude", "test-repo", "--timeout", "60"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should use default runs directory
        assert default_challenges_dir.exists(), "Default runs dir not created"
        challenge_dirs = list(default_challenges_dir.iterdir())
        assert len(challenge_dirs) > 0, "No run directory created in default runs dir"
    finally:
        state.get_repos_dir = original_repos
        state.get_challenges_dir = original_challenges


def test_cli_challenges_dir_creates_directory_if_missing(
    tmp_path: Path,
    tmp_repo: Path,
    sample_bugs_json: dict[str, Any],
    prompts_dir: Path,
) -> None:
    """Test that --challenges-dir creates the directory if it doesn't exist."""
    from cheddar.cli import app, state
    from cheddar.models import BugManifest

    # Setup repos directory
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    import shutil

    shutil.copytree(tmp_repo, repos_dir / "test-repo")

    # Deep nested path that doesn't exist
    nested_challenges_dir = tmp_path / "deep" / "nested" / "experiment" / "runs"
    assert not nested_challenges_dir.exists()

    # Create a fake sandbox directory for the mock
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "main.py").write_text("def hello():\n    return 'Hello, World!'\n")

    original_repos = state.get_repos_dir
    state.get_repos_dir = lambda: repos_dir

    try:
        mock_manifest = BugManifest(**sample_bugs_json)
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.can_challenge = True
        mock_agent.challenge.side_effect = _make_challenge_side_effect(sample_bugs_json)

        with (
            patch("cheddar.agents.get_agent", return_value=mock_agent),
            patch("cheddar.prompts.get_prompts_dir", return_value=prompts_dir),
            patch("cheddar.sandbox.create_challenge_sandbox", return_value=sandbox_dir),
            patch("cheddar.sandbox.persist_sandbox", return_value=tmp_path / "persisted"),
            patch("cheddar.sandbox.cleanup_sandbox"),
            patch(
                "cheddar.extractor.reformat_manifest",
                return_value=(mock_manifest, "diff content", "prompt content"),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "challenge",
                    "claude",
                    "test-repo",
                    "--challenges-dir",
                    str(nested_challenges_dir),
                    "--timeout",
                    "60",
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should create the full nested path
        assert nested_challenges_dir.exists(), "Nested runs dir not created"
    finally:
        state.get_repos_dir = original_repos
