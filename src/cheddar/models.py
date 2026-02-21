"""Pydantic models for cheddar-bench data structures."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Constants
DEFAULT_TIMEOUT_SECONDS = 3600  # 60 minutes

# Type aliases
AgentName = Literal["claude", "codex", "gemini"]
ChallengerName = Literal["claude", "codex", "gemini"]
ChallengeStatus = Literal["pending", "challenged", "reviewed", "matched", "reported", "failed"]
ReviewStatus = Literal["complete", "timeout", "error", "failed"]


def _empty_match_assignments() -> list["GlobalBugMatchAssignment"]:
    """Return a typed empty assignment list for pydantic defaults."""
    return []


class Bug(BaseModel):
    """An injected defect with ground truth metadata."""

    file: str = Field(..., min_length=1, description="Relative path from repo root")
    line: int = Field(..., ge=1, description="Line number after injection")
    type: str = Field(..., min_length=1, description="Bug category (free-form)")
    description: str = Field(..., min_length=1, description="What the bug does")
    original: str = Field(..., min_length=1, description="Original correct code")
    injected: str = Field(..., description="Buggy code (empty string for deletions)")
    change_kind: Literal["modify", "delete"] = Field(
        default="modify", description="Type of change: modify (replace) or delete (remove)"
    )


class BugManifest(BaseModel):
    """Ground truth output from challenge phase."""

    bug_count: int = Field(..., ge=1, description="Number of bugs injected")
    bugs: list[Bug] = Field(..., description="List of injected bugs")


class ReviewReport(BaseModel):
    """Output from review phase."""

    agent: AgentName = Field(..., description="Reviewer agent name")
    raw_output: str = Field(..., description="Full text of the review report")
    raw_stderr: str = Field(default="", description="Stderr output for debugging")
    duration_seconds: float = Field(..., ge=0, description="Time taken")
    status: ReviewStatus = Field(..., description="Completion status")
    failure_reason: str | None = Field(default=None, description="Why review failed")
    mutated_files: list[str] | None = Field(
        default=None, description="Files modified during review (if any)"
    )
    found_bugs: list["FoundBug"] | None = Field(
        default=None,
        description="Optional parsed findings retained for compatibility",
    )
    raw_findings_total: int | None = Field(
        default=None,
        ge=0,
        description="Total bugs/*.json files emitted by reviewer",
    )
    raw_findings_snapshot_dir: str | None = Field(
        default=None,
        description="Relative path under challenge dir containing copied raw bugs/*.json",
    )


class FoundBug(BaseModel):
    """Structured reviewer finding stored in bugs/*.json."""

    file: str = Field(..., min_length=1, description="Relative file path from repo root")
    line: int | None = Field(default=None, ge=1, description="Approximate line number")
    type: str = Field(..., min_length=1, description="Bug category (free-form)")
    why: str = Field(..., min_length=1, description="Why this is a bug")
    impact: str = Field(..., min_length=1, description="Expected impact if unfixed")


class BugMatch(BaseModel):
    """Result of matching one ground truth bug against the review report."""

    bug_index: int = Field(..., ge=0, description="Index into BugManifest.bugs")
    finding_index: int = Field(
        default=0,
        ge=0,
        description="1-based finding index (0 if unmatched)",
    )
    reasoning: str = Field(..., description="LLM explanation of the match decision")
    matched_quote: str = Field(..., description="Verbatim quote from review that matched this bug")
    matched_line: int = Field(
        default=0, description="Source line the review references (0 if not found)"
    )
    found: bool = Field(..., description="Was the bug mentioned in report?")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Match confidence")
    error: bool = Field(..., description="True if match failed due to LLM error")


class GlobalBugMatchAssignment(BaseModel):
    """One global-assignment decision for a bug."""

    bug_index: int = Field(..., ge=0, description="Index into BugManifest.bugs")
    finding_index: int = Field(
        ...,
        ge=0,
        description="1-based finding index (0 means unmatched)",
    )
    reasoning: str = Field(..., description="Brief explanation of the assignment decision")
    matched_quote: str = Field(
        default="",
        description="Supporting quote from findings text (empty if unmatched)",
    )
    matched_line: int = Field(
        default=0,
        ge=0,
        description="Referenced line for the matched finding (0 if unmatched)",
    )
    confidence: Literal["high", "medium", "low"] = Field(..., description="Assignment confidence")


class GlobalBugMatchOutput(BaseModel):
    """LLM output for global bug-to-finding assignment in one call."""

    assignments: list[GlobalBugMatchAssignment] = Field(
        default_factory=_empty_match_assignments,
        description="Per-bug assignment decisions (one entry per bug index expected)",
    )


class RawFindingPayload(BaseModel):
    """One raw reviewer finding payload from bugs/*.json."""

    filename: str = Field(..., min_length=1, description="Source filename under bugs/")
    content: str = Field(..., description="Raw file content (may be malformed JSON)")


class Score(BaseModel):
    """Aggregate scoring for one reviewer."""

    reviewer: str = Field(..., min_length=1, description="Agent name")
    total_bugs: int = Field(..., ge=1, description="Number of injected bugs")
    bugs_found: int = Field(..., ge=0, description="Number matched")
    detection_rate: float = Field(
        ..., ge=0, le=1, description="bugs_found / evaluated bugs (excludes LLM errors)"
    )
    model: str = Field(..., min_length=1, description="LLM model used for matching")
    matches: list[BugMatch] = Field(..., description="Per-bug match results")
    match_errors: int = Field(..., ge=0, description="Number of bugs that failed LLM matching")
    manifest_hash: str = Field(
        default="", description="SHA-256 of bugs.json content for staleness detection"
    )


class ChallengeConfig(BaseModel):
    """Configuration metadata for a benchmark challenge."""

    challenge_id: str | None = Field(default=None, description="Legacy: use run_id")
    run_id: str | None = Field(default=None, description="Unique identifier")
    repo: str = Field(..., description="Repository name")
    challenger: ChallengerName = Field(..., description="Injector agent")
    created_at: datetime = Field(..., description="Challenge start time")
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, description="Agent timeout")
    duration_seconds: float | None = Field(default=None, description="Agent execution time")

    @property
    def id(self) -> str:
        """Get the ID, supporting both old and new field names."""
        return self.run_id or self.challenge_id or ""


class Challenge(BaseModel):
    """Complete benchmark challenge state."""

    config: ChallengeConfig = Field(..., description="Challenge configuration")
    bugs: BugManifest | None = Field(default=None, description="Ground truth")
    reviews: dict[str, ReviewReport] = Field(
        default_factory=lambda: {}, description="Per-reviewer review reports"
    )
    scores: dict[str, Score] | None = Field(default=None, description="Per-reviewer scores")
    status: ChallengeStatus = Field(..., description="Overall status")
