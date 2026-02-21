"""Tests for agent implementations."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Base agent tests


def test_base_agent_name() -> None:
    """Test that base agent requires name property implementation."""
    from cheddar.agents.base import BaseAgent

    # BaseAgent is abstract, so we can't instantiate it directly
    # This test verifies the interface exists
    assert hasattr(BaseAgent, "name")
    assert hasattr(BaseAgent, "can_challenge")
    assert hasattr(BaseAgent, "can_review")


def test_base_agent_run_cli_success(tmp_path: Path) -> None:
    """Test _run_cli executes command successfully."""
    from cheddar.agents.base import BaseAgent

    # Create a concrete implementation for testing
    class TestAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "test"

        def _build_challenge_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return ["echo", "test"]

        def _build_review_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return ["echo", "test"]

    agent = TestAgent()
    result = agent._run_cli(["echo", "hello"], cwd=tmp_path, timeout=10)

    assert result.returncode == 0
    assert "hello" in result.stdout


def test_base_agent_run_cli_timeout(tmp_path: Path) -> None:
    """Test _run_cli handles timeout."""
    import subprocess

    from cheddar.agents.base import BaseAgent

    class TestAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "test"

        def _build_challenge_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return ["sleep", "10"]

        def _build_review_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return ["sleep", "10"]

    agent = TestAgent()
    with pytest.raises(subprocess.TimeoutExpired):
        agent._run_cli(["sleep", "10"], cwd=tmp_path, timeout=1)


def test_get_challengers_and_reviewers_use_class_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability listing should not instantiate agent classes."""
    import cheddar.agents as agents_module
    from cheddar.agents import get_challengers, get_reviewers
    from cheddar.agents.base import BaseAgent

    class ChallengeOnlyAgent(BaseAgent):
        CAN_CHALLENGE = True
        CAN_REVIEW = False

        def __init__(self) -> None:
            raise AssertionError("Agent should not be instantiated")

        @property
        def name(self) -> str:
            return "challenge-only"

        def _build_challenge_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return []

        def _build_review_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return []

    class ReviewOnlyAgent(BaseAgent):
        CAN_CHALLENGE = False
        CAN_REVIEW = True

        def __init__(self) -> None:
            raise AssertionError("Agent should not be instantiated")

        @property
        def name(self) -> str:
            return "review-only"

        def _build_challenge_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return []

        def _build_review_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return []

    class DisabledAgent(BaseAgent):
        CAN_CHALLENGE = False
        CAN_REVIEW = False

        def __init__(self) -> None:
            raise AssertionError("Agent should not be instantiated")

        @property
        def name(self) -> str:
            return "disabled"

        def _build_challenge_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return []

        def _build_review_cmd(self, prompt: str) -> list[str]:  # noqa: ARG002
            return []

    test_agents: dict[str, type[BaseAgent]] = {
        "challenge": ChallengeOnlyAgent,
        "review": ReviewOnlyAgent,
        "disabled": DisabledAgent,
    }
    monkeypatch.setattr(agents_module, "AGENTS", test_agents)

    assert get_challengers() == ["challenge"]
    assert get_reviewers() == ["review"]


# Claude agent tests


def test_claude_agent_name() -> None:
    """Test ClaudeAgent has correct name."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()
    assert agent.name == "claude"


def test_claude_agent_can_challenge() -> None:
    """Test ClaudeAgent can challenge."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()
    assert agent.can_challenge is True


def test_claude_agent_can_review() -> None:
    """Test ClaudeAgent can review."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()
    assert agent.can_review is True


def test_claude_agent_build_challenge_cmd() -> None:
    """Test ClaudeAgent builds correct challenge command."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()
    cmd = agent._build_challenge_cmd("test prompt")

    assert "claude" in cmd[0].lower() or cmd[0] == "claude"
    assert any("test prompt" in arg for arg in cmd)


def test_claude_agent_build_review_cmd() -> None:
    """Test ClaudeAgent builds correct review command."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()
    cmd = agent._build_review_cmd("test prompt")

    assert "claude" in cmd[0].lower() or cmd[0] == "claude"
    assert any("test prompt" in arg for arg in cmd)


def test_claude_agent_challenge_success(tmp_path: Path) -> None:
    """Test ClaudeAgent challenge returns stdout and stderr."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    # Mock subprocess to return output (manifest extraction is done separately now)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Agent completed bug injection successfully."
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        raw_output, raw_stderr = agent.challenge(
            sandbox_path=tmp_path,
            prompt="Inject bugs",
            timeout_seconds=60,
        )

    assert raw_output == "Agent completed bug injection successfully."
    assert raw_stderr == ""


def test_claude_agent_challenge_timeout(tmp_path: Path) -> None:
    """Test ClaudeAgent challenge handles timeout."""
    import subprocess

    from cheddar.agents.claude import ClaudeAgent
    from cheddar.errors import AgentTimeoutError

    agent = ClaudeAgent()

    with (
        patch.object(
            agent, "_run_cli", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=60)
        ),
        pytest.raises(AgentTimeoutError) as exc_info,
    ):
        agent.challenge(
            sandbox_path=tmp_path,
            prompt="Inject bugs",
            timeout_seconds=60,
        )
    assert "claude" in exc_info.value.agent


def test_claude_agent_challenge_returns_raw_output(tmp_path: Path) -> None:
    """Test ClaudeAgent challenge returns raw output without parsing.

    Manifest extraction is now done separately via LLM, so challenge()
    just returns the raw stdout/stderr from the agent.
    """
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "I modified the files as requested."
    mock_result.stderr = "some debug info"

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        raw_output, raw_stderr = agent.challenge(
            sandbox_path=tmp_path,
            prompt="Inject bugs",
            timeout_seconds=60,
        )

    assert raw_output == "I modified the files as requested."
    assert raw_stderr == "some debug info"


def test_claude_agent_review_success(tmp_path: Path) -> None:
    """Test ClaudeAgent review with mocked subprocess."""
    from cheddar.agents.claude import ClaudeAgent
    from cheddar.models import ReviewReport

    agent = ClaudeAgent()

    # Review now returns raw text, no JSON parsing
    # Must be > 500 chars to pass validation (MIN_REVIEW_LENGTH)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        "# Code Review\n\n"
        "## Critical Issues\n\n"
        "Found an off-by-one error in main.py:10. The loop iterates one too many times, "
        "causing an index out of bounds error when accessing the array. This affects the "
        "data processing pipeline and could lead to crashes in production.\n\n"
        "## Secondary Issues\n\n"
        "The error handling in utils.py:45 swallows exceptions silently. The bare except "
        "clause should be narrowed to catch only expected exceptions like ValueError.\n\n"
        "## Recommendations\n\n"
        "1. Change the condition from `i <= len` to `i < len` in the main loop.\n"
        "2. Replace bare except with specific exception types in the utility module.\n"
        "3. Add unit tests for boundary conditions in the array processing function.\n"
    )
    mock_result.stderr = ""

    bugs_dir = tmp_path / "bugs"
    bugs_dir.mkdir()
    (bugs_dir / "bug-1.json").write_text(
        """
{
  "file": "main.py",
  "line": 10,
  "type": "off-by-one",
  "why": "Loop condition allows one extra iteration",
  "impact": "Can trigger index-out-of-bounds"
}
""".strip()
    )

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert isinstance(report, ReviewReport)
    assert report.agent == "claude"
    assert "off-by-one" in report.raw_output
    assert report.status == "complete"


# Gemini agent tests


def test_gemini_agent_name() -> None:
    """Test GeminiAgent has correct name."""
    from cheddar.agents.gemini import GeminiAgent

    agent = GeminiAgent()
    assert agent.name == "gemini"


def test_gemini_agent_can_challenge() -> None:
    """Test GeminiAgent can challenge."""
    from cheddar.agents.gemini import GeminiAgent

    agent = GeminiAgent()
    assert agent.can_challenge is True


def test_gemini_agent_can_review() -> None:
    """Test GeminiAgent can review."""
    from cheddar.agents.gemini import GeminiAgent

    agent = GeminiAgent()
    assert agent.can_review is True


def test_gemini_agent_uses_isolated_config() -> None:
    """Test GeminiAgent writes settings to a temp dir, not ~/.gemini."""
    from cheddar.agents.gemini import GeminiAgent

    agent = GeminiAgent()

    # Settings written to isolated temp dir
    settings_file = agent._gemini_home / ".gemini" / "settings.json"
    assert settings_file.exists()

    # HOME is set to the temp dir in env
    env = agent._get_env()
    assert env["HOME"] == str(agent._gemini_home)
    assert Path.home() not in agent._gemini_home.parents


def test_gemini_agent_challenge_success(tmp_path: Path) -> None:
    """Test GeminiAgent challenge returns stdout and stderr."""
    from cheddar.agents.gemini import GeminiAgent

    agent = GeminiAgent()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Gemini completed bug injection."
    mock_result.stderr = ""

    with patch.object(agent, "_run_cli", return_value=mock_result):
        raw_output, raw_stderr = agent.challenge(
            sandbox_path=tmp_path,
            prompt="Inject bugs",
            timeout_seconds=60,
        )

    assert raw_output == "Gemini completed bug injection."
    assert raw_stderr == ""


def test_gemini_agent_review_success(tmp_path: Path) -> None:
    """Test GeminiAgent review with mocked subprocess."""
    from cheddar.agents.gemini import GeminiAgent
    from cheddar.models import ReviewReport

    agent = GeminiAgent()

    # Review now returns raw text, no JSON parsing
    # Must be > 500 chars to pass validation (MIN_REVIEW_LENGTH)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        "# Code Review\n\n"
        "## Critical Issues\n\n"
        "Found an off-by-one error in main.py:10. The loop iterates one too many times, "
        "causing an index out of bounds error when accessing the array. This affects the "
        "data processing pipeline and could lead to crashes in production.\n\n"
        "## Secondary Issues\n\n"
        "The error handling in utils.py:45 swallows exceptions silently. The bare except "
        "clause should be narrowed to catch only expected exceptions like ValueError.\n\n"
        "## Recommendations\n\n"
        "1. Change the condition from `i <= len` to `i < len` in the main loop.\n"
        "2. Replace bare except with specific exception types in the utility module.\n"
        "3. Add unit tests for boundary conditions in the array processing function.\n"
    )
    mock_result.stderr = ""

    with patch.object(agent, "_run_cli", return_value=mock_result):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert isinstance(report, ReviewReport)
    assert report.agent == "gemini"
    assert "off-by-one" in report.raw_output


# Codex agent tests


def test_codex_agent_name() -> None:
    """Test CodexAgent has correct name."""
    from cheddar.agents.codex import CodexAgent

    agent = CodexAgent()
    assert agent.name == "codex"


def test_codex_agent_can_challenge() -> None:
    """Test CodexAgent can challenge."""
    from cheddar.agents.codex import CodexAgent

    agent = CodexAgent()
    assert agent.can_challenge is True


def test_codex_agent_can_review() -> None:
    """Test CodexAgent can review."""
    from cheddar.agents.codex import CodexAgent

    agent = CodexAgent()
    assert agent.can_review is True


def test_codex_agent_challenge_success(tmp_path: Path) -> None:
    """Test CodexAgent challenge returns stdout and stderr."""
    from cheddar.agents.codex import CodexAgent

    agent = CodexAgent()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        '{"type":"thread.started","thread_id":"test-123"}\n'
        '{"type":"turn.started"}\n'
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
        '"text":"Codex completed bug injection."}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}\n'
    )
    mock_result.stderr = ""

    with patch.object(agent, "_run_cli", return_value=mock_result):
        raw_output, raw_stderr = agent.challenge(
            sandbox_path=tmp_path,
            prompt="Inject bugs",
            timeout_seconds=60,
        )

    assert raw_output == "Codex completed bug injection."
    assert raw_stderr == ""


def test_codex_agent_review_success(tmp_path: Path) -> None:
    """Test CodexAgent review with mocked subprocess."""
    import json as _json

    from cheddar.agents.codex import CodexAgent
    from cheddar.models import ReviewReport

    agent = CodexAgent()

    # Codex --json JSONL format; review text must be > 500 chars
    review_text = (
        "# Code Review\n\n"
        "## Critical Issues\n\n"
        "Found an off-by-one error in main.py:10. The loop iterates one too many times, "
        "causing an index out of bounds error when accessing the array. This affects the "
        "data processing pipeline and could lead to crashes in production.\n\n"
        "## Secondary Issues\n\n"
        "The error handling in utils.py:45 swallows exceptions silently. The bare except "
        "clause should be narrowed to catch only expected exceptions like ValueError.\n\n"
        "## Recommendations\n\n"
        "1. Change the condition from `i <= len` to `i < len` in the main loop.\n"
        "2. Replace bare except with specific exception types in the utility module.\n"
        "3. Add unit tests for boundary conditions in the array processing function.\n"
    )
    item_event = _json.dumps(
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": review_text},
        }
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        '{"type":"thread.started","thread_id":"test-456"}\n'
        '{"type":"turn.started"}\n'
        f"{item_event}\n"
        '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}\n'
    )
    mock_result.stderr = ""

    with patch.object(agent, "_run_cli", return_value=mock_result):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert isinstance(report, ReviewReport)
    assert report.agent == "codex"
    assert "off-by-one" in report.raw_output


# Failed review detection tests


def test_review_fails_on_auth_prompt(tmp_path: Path) -> None:
    """Test that OAuth authorization prompts are detected as failed reviews."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    # Simulated OAuth prompt (like what Gemini outputs when not authenticated)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        "Please visit the following URL to authorize the application:\n\n"
        "https://accounts.google.com/o/oauth2/v2/auth?redirect_uri=...\n\n"
        "Enter the authorization code:"
    )
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert report.status == "failed"


def test_review_fails_on_short_output(tmp_path: Path) -> None:
    """Test that very short outputs are detected as failed reviews."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "OK"  # Way too short to be a valid review
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert report.status == "failed"


def test_review_strips_ansi_codes_for_validation(tmp_path: Path) -> None:
    """Test that ANSI escape codes are stripped before validation."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    # Auth prompt with ANSI escape codes (like real Gemini output)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        "\x1b[?1049h\x1b[2J\x1b[H\x1b[?1006l"
        "Please visit the following URL to authorize the application:\n\n"
        "https://accounts.google.com/o/oauth2/v2/auth\n\n"
        "Enter the authorization code:"
    )
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert report.status == "failed"


# Review timeout and error handling tests


def test_claude_agent_review_timeout(tmp_path: Path) -> None:
    """Test ClaudeAgent review raises AgentTimeoutError on timeout."""
    import subprocess

    from cheddar.agents.claude import ClaudeAgent
    from cheddar.errors import AgentTimeoutError

    agent = ClaudeAgent()

    with (
        patch.object(
            agent, "_run_cli", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=60)
        ),
        pytest.raises(AgentTimeoutError) as exc_info,
    ):
        agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )
    assert "claude" in exc_info.value.agent
    assert "timed out" in exc_info.value.message.lower()


def test_claude_agent_challenge_cli_error(tmp_path: Path) -> None:
    """Test ClaudeAgent challenge raises AgentError on non-zero returncode."""
    from cheddar.agents.claude import ClaudeAgent
    from cheddar.errors import AgentError

    agent = ClaudeAgent()

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "CLI error: command not found"

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
        pytest.raises(AgentError) as exc_info,
    ):
        agent.challenge(
            sandbox_path=tmp_path,
            prompt="Inject bugs",
            timeout_seconds=60,
        )
    assert "claude" in exc_info.value.agent
    assert "CLI exited with code 1" in exc_info.value.message


def test_claude_agent_review_cli_error(tmp_path: Path) -> None:
    """Test ClaudeAgent review raises AgentError on non-zero returncode."""
    from cheddar.agents.claude import ClaudeAgent
    from cheddar.errors import AgentError

    agent = ClaudeAgent()

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "CLI error: permission denied"

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
        pytest.raises(AgentError) as exc_info,
    ):
        agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )
    assert "claude" in exc_info.value.agent
    assert "CLI exited with code 1" in exc_info.value.message


# Mutation detection tests


def test_review_detects_no_mutation(tmp_path: Path) -> None:
    """Test that review reports no mutation when repo unchanged."""
    from cheddar.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        "# Code Review\n\n"
        "## Issues Found\n\n"
        "Found a potential null pointer dereference in src/main.py at line 42. "
        "The variable 'user' could be None when accessed.\n\n"
        "## Recommendations\n\n"
        "Add a null check before accessing user.name."
    )
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert report.mutated_files is None


def test_review_detects_mutation(tmp_path: Path) -> None:
    """Test that review detects when agent mutates the repo."""
    import subprocess

    from cheddar.agents.claude import ClaudeAgent

    # Initialize a git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

    # Create and commit a file (--no-verify to avoid global hooks interfering)
    (tmp_path / "test.py").write_text("print('hello')")
    subprocess.run(["git", "add", "test.py"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "initial"], cwd=tmp_path, capture_output=True
    )

    agent = ClaudeAgent()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = (
        "# Code Review\n\n"
        "## Issues Found\n\n"
        "Found a potential null pointer dereference in src/main.py at line 42. "
        "The variable 'user' could be None when accessed.\n\n"
        "## Recommendations\n\n"
        "Add a null check before accessing user.name."
    )
    mock_result.stderr = ""

    def run_cli_with_mutation(*_args: object, **_kwargs: object) -> MagicMock:
        # Simulate agent modifying a file
        (tmp_path / "test.py").write_text("print('modified')")
        return mock_result

    with (
        patch.object(agent, "_run_cli", side_effect=run_cli_with_mutation),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(
            sandbox_path=tmp_path,
            prompt="Review code",
            timeout_seconds=60,
        )

    assert report.mutated_files is not None
    assert "test.py" in report.mutated_files


def test_review_accepts_raw_findings_with_short_stdout(tmp_path: Path) -> None:
    """Short stdout is accepted when raw bugs/*.json findings exist."""
    from cheddar.agents.claude import ClaudeAgent

    bugs_dir = tmp_path / "bugs"
    bugs_dir.mkdir()
    (bugs_dir / "bug-1.json").write_text(
        """
{
  "file": "src/main.py",
  "line": 42,
  "type": "logic error",
  "why": "Condition is inverted",
  "impact": "Requests fail unexpectedly"
}
""".strip()
    )

    agent = ClaudeAgent()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "done"
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(sandbox_path=tmp_path, prompt="Review code", timeout_seconds=60)

    assert report.status == "complete"
    assert report.raw_findings_total == 1


def test_review_counts_raw_findings_without_structured_validation(tmp_path: Path) -> None:
    """Review report keeps raw finding counts without validating schema."""
    from cheddar.agents.claude import ClaudeAgent

    bugs_dir = tmp_path / "bugs"
    bugs_dir.mkdir()
    (bugs_dir / "valid.json").write_text(
        """
{
  "file": "src/main.py",
  "line": 42,
  "type": "logic error",
  "why": "Condition is inverted",
  "impact": "Requests fail unexpectedly"
}
""".strip()
    )
    # Missing required fields should still count as raw findings.
    (bugs_dir / "invalid.json").write_text('{"file": "src/main.py", "line": 1}')

    agent = ClaudeAgent()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "done"
    mock_result.stderr = ""

    with (
        patch.object(agent, "_run_cli", return_value=mock_result),
        patch.object(agent, "_parse_output", side_effect=lambda x: x),
    ):
        report = agent.review(sandbox_path=tmp_path, prompt="Review code", timeout_seconds=60)

    assert report.status == "complete"
    assert report.raw_findings_total == 2


# Claude stream-json parser tests


def test_extract_text_from_stream_json_multi_turn() -> None:
    """Test extraction collects all assistant text blocks across turns."""
    import json

    from cheddar.agents.claude import _extract_text_from_stream_json

    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Found bug 1 in parse.ts."}]},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": "Read", "input": {"file": "x.py"}}]
                },
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Found bug 2 in util.ts."}]},
            }
        ),
        json.dumps({"type": "result", "result": "Summary only."}),
    ]
    raw = "\n".join(lines)
    result = _extract_text_from_stream_json(raw)
    assert "Found bug 1 in parse.ts." in result
    assert "Found bug 2 in util.ts." in result


def test_extract_text_from_stream_json_fallback_on_timeout() -> None:
    """Test fallback to assistant text blocks when no result message (timeout)."""
    import json

    from cheddar.agents.claude import _extract_text_from_stream_json

    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Found bug 1."}]},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Found bug 2."}]},
            }
        ),
        # No result message — simulates timeout
    ]
    raw = "\n".join(lines)
    result = _extract_text_from_stream_json(raw)
    assert "Found bug 1." in result
    assert "Found bug 2." in result


def test_extract_text_from_stream_json_empty_input() -> None:
    """Test parser handles empty input gracefully."""
    from cheddar.agents.claude import _extract_text_from_stream_json

    assert _extract_text_from_stream_json("") == ""
    assert _extract_text_from_stream_json("not json\nalso not json") == ""


def test_extract_text_from_stream_json_ignores_tool_use() -> None:
    """Test parser ignores tool_use content blocks."""
    import json

    from cheddar.agents.claude import _extract_text_from_stream_json

    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}]
                },
            }
        ),
    ]
    raw = "\n".join(lines)
    assert _extract_text_from_stream_json(raw) == ""
