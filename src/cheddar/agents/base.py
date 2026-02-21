"""Base agent implementation for CLI coding agents."""

import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from cheddar.errors import AgentError, AgentParseError, AgentTimeoutError
from cheddar.models import (
    DEFAULT_TIMEOUT_SECONDS,
    AgentName,
    ReviewReport,
    ReviewStatus,
)


def _get_repo_status(path: Path) -> set[str]:
    """Get set of modified/untracked files in repo.

    Returns empty set if not a git repo or git fails.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return set()
        # Parse porcelain output: "XY filename" or "XY filename -> newname"
        # Format: 2 status chars + space + filename (e.g. " M test.py" or "?? new.py")
        files: set[str] = set()
        for line in result.stdout.splitlines():
            if len(line) >= 4:
                # Handle renames: "R  old -> new"
                filename = line[3:]
                if " -> " in filename:
                    filename = filename.split(" -> ")[-1]
                files.add(filename)
        return files
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


# Minimum length for a valid review (after stripping ANSI codes).
# Shortest real review in 450-review experiment was 610 chars.
MIN_REVIEW_LENGTH = 500


class BaseAgent(ABC):
    """Base class for CLI agent implementations."""

    CAN_CHALLENGE = False
    CAN_REVIEW = True

    @property
    @abstractmethod
    def name(self) -> AgentName:
        """Agent identifier (e.g., 'claude', 'gemini')."""
        ...

    def _parse_output(self, raw_stdout: str) -> str:
        """Parse CLI stdout into plain text.

        Override in subclasses that use structured output formats (e.g., stream-json).
        Default: return as-is.
        """
        return raw_stdout

    @property
    def can_challenge(self) -> bool:
        """Whether this agent supports bug injection.

        Capability is defined by the class-level ``CAN_CHALLENGE`` flag.
        """
        return type(self).CAN_CHALLENGE

    @property
    def can_review(self) -> bool:
        """Whether this agent supports review.

        Capability is defined by the class-level ``CAN_REVIEW`` flag.
        """
        return type(self).CAN_REVIEW

    def _run_cli(
        self,
        cmd: list[str],
        cwd: Path,
        timeout: int,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute CLI command with timeout and capture.

        Args:
            cmd: Command and arguments to execute.
            cwd: Working directory for the command.
            timeout: Maximum execution time in seconds.
            env: Additional environment variables.

        Returns:
            CompletedProcess with stdout and stderr captured.

        Raises:
            subprocess.TimeoutExpired: If command exceeds timeout.
        """
        full_env = {**os.environ, **(env or {})}
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )

    @abstractmethod
    def _build_challenge_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for challenge."""
        ...

    @abstractmethod
    def _build_review_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for review."""
        ...

    def _get_env(self) -> dict[str, str] | None:
        """Return additional environment variables for CLI execution.

        Override in subclass to provide agent-specific env vars.
        """
        return None

    def _validate_review_output(self, raw_output: str) -> tuple[bool, str | None]:
        """Validate review output is a real review (not an auth prompt or error).

        Args:
            raw_output: Raw output from the agent.

        Returns:
            Tuple of (is_valid, failure_reason).
        """
        # Strip ANSI escape codes for checking
        clean_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw_output)
        clean_output = re.sub(r"\x1b\[\?[0-9;]*[a-zA-Z]", "", clean_output)

        if len(clean_output.strip()) < MIN_REVIEW_LENGTH:
            return (
                False,
                f"Review too short ({len(clean_output.strip())} chars < {MIN_REVIEW_LENGTH})",
            )

        return True, None

    def _count_raw_findings(self, sandbox_path: Path) -> int:
        """Count raw findings emitted under bugs/*.json in the sandbox."""
        bugs_dir = sandbox_path / "bugs"
        if not bugs_dir.is_dir():
            return 0
        return sum(1 for _ in bugs_dir.glob("*.json"))

    def challenge(
        self,
        sandbox_path: Path,
        prompt: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> tuple[str, str]:
        """Inject bugs into the codebase.

        The agent modifies files in the sandbox. Bug manifest extraction
        is handled separately via LLM analysis of the git diff.

        Args:
            sandbox_path: Path to sandboxed repository.
            prompt: Injection prompt.
            timeout_seconds: Maximum execution time.

        Returns:
            Tuple of (raw_stdout, raw_stderr).

        Raises:
            AgentTimeoutError: If agent exceeds timeout.
            AgentError: If CLI fails.
        """
        if not self.can_challenge:
            raise AgentError(
                agent=self.name,
                message=f"Agent {self.name} does not support bug injection",
                raw_output="",
            )

        cmd = self._build_challenge_cmd(prompt)
        env = self._get_env()

        try:
            result = self._run_cli(cmd, cwd=sandbox_path, timeout=timeout_seconds, env=env)
        except subprocess.TimeoutExpired as e:
            partial = self._parse_output(str(e.stdout)) if e.stdout else ""
            raise AgentTimeoutError(
                agent=self.name,
                message=f"Challenge timed out after {timeout_seconds}s",
                raw_output=partial,
            ) from e

        raw_output = self._parse_output(result.stdout)

        # Check for CLI failures
        if result.returncode != 0:
            raise AgentError(
                agent=self.name,
                message=f"CLI exited with code {result.returncode}: {result.stderr}",
                raw_output=raw_output,
            )

        return raw_output, result.stderr

    def review(
        self,
        sandbox_path: Path,
        prompt: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        detect_mutation: bool = True,
    ) -> ReviewReport:
        """Review codebase for bugs.

        Args:
            sandbox_path: Path to sandboxed repository.
            prompt: Review prompt.
            timeout_seconds: Maximum execution time.
            detect_mutation: If True, track whether review mutated the repo.

        Returns:
            ReviewReport with raw_output containing the full report text.

        Raises:
            AgentTimeoutError: If agent exceeds timeout.
            AgentError: If CLI fails.
        """
        if not self.can_review:
            raise AgentParseError(
                agent=self.name,
                message=f"Agent {self.name} does not support code review",
                raw_output="",
            )

        # Capture repo state before review
        files_before: set[str] = _get_repo_status(sandbox_path) if detect_mutation else set()

        cmd = self._build_review_cmd(prompt)
        env = self._get_env()
        start_time = time.time()

        try:
            result = self._run_cli(cmd, cwd=sandbox_path, timeout=timeout_seconds, env=env)
        except subprocess.TimeoutExpired as e:
            partial = self._parse_output(str(e.stdout)) if e.stdout else ""
            raise AgentTimeoutError(
                agent=self.name,
                message=f"Review timed out after {timeout_seconds}s",
                raw_output=partial,
            ) from e

        raw_output = self._parse_output(result.stdout)
        duration = time.time() - start_time

        # Capture repo state after review and detect mutations
        mutated_files: list[str] | None = None
        if detect_mutation:
            files_after = _get_repo_status(sandbox_path)
            new_changes = files_after - files_before
            if new_changes:
                mutated_files = sorted(new_changes)

        # Check for CLI failures
        if result.returncode != 0:
            raise AgentError(
                agent=self.name,
                message=f"CLI exited with code {result.returncode}: {result.stderr}",
                raw_output=raw_output,
            )

        raw_findings_total = self._count_raw_findings(sandbox_path)

        # At least one raw finding payload is required for a valid review.
        if raw_findings_total > 0:
            is_valid, failure_reason = True, None
        else:
            is_valid, failure_reason = False, "No findings in bugs/*.json"
        status: ReviewStatus = "complete" if is_valid else "failed"

        return ReviewReport(
            agent=self.name,
            raw_output=raw_output,
            raw_stderr=result.stderr,
            duration_seconds=duration,
            status=status,
            failure_reason=failure_reason,
            mutated_files=mutated_files,
            raw_findings_total=raw_findings_total,
        )
