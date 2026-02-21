"""Error classes for agent operations."""


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(self, agent: str, message: str, raw_output: str = "") -> None:
        """Initialize AgentError.

        Args:
            agent: Name of the agent that failed.
            message: Error message describing what went wrong.
            raw_output: Raw output from the agent (for debugging).
        """
        self.agent = agent
        self.message = message
        self.raw_output = raw_output
        super().__init__(f"[{agent}] {message}")


class AgentTimeoutError(AgentError):
    """Agent exceeded timeout."""

    pass


class AgentParseError(AgentError):
    """Failed to parse agent output."""

    pass


class ExtractionError(Exception):
    """Failed to extract bug manifest from challenge output."""

    def __init__(
        self,
        message: str,
        git_diff: str = "",
        raw_output: str = "",
    ) -> None:
        """Initialize ExtractionError.

        Args:
            message: Error message describing what went wrong.
            git_diff: Git diff output (for debugging).
            raw_output: Raw agent output (for debugging).
        """
        self.message = message
        self.git_diff = git_diff
        self.raw_output = raw_output
        super().__init__(message)
