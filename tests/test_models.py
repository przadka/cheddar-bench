"""Tests for Pydantic models and error classes."""

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError


def test_bug_valid_data(sample_bug_data: dict[str, Any]) -> None:
    """Test Bug model with valid data."""
    from cheddar.models import Bug

    bug = Bug(**sample_bug_data)
    assert bug.file == "main.py"
    assert bug.line == 10
    assert bug.type == "off-by-one"
    assert bug.description == "Loop iterates one extra time"
    assert bug.original == "for i in range(n):"
    assert bug.injected == "for i in range(n + 1):"


def test_bug_invalid_line() -> None:
    """Test Bug model rejects invalid line number."""
    from cheddar.models import Bug

    with pytest.raises(ValidationError) as exc_info:
        Bug(
            file="main.py",
            line=0,  # Must be >= 1
            type="off-by-one",
            description="test",
            original="x",
            injected="y",
        )
    assert "line" in str(exc_info.value)


def test_bug_accepts_freeform_type() -> None:
    """Test Bug model accepts free-form bug types (LLMs use varied phrasing)."""
    from cheddar.models import Bug

    # Should accept any non-empty string for type
    bug = Bug(
        file="main.py",
        line=1,
        type="edge case failure",  # Free-form, not strict enum
        description="test",
        original="x",
        injected="y",
    )
    assert bug.type == "edge case failure"


def test_bug_empty_file() -> None:
    """Test Bug model rejects empty file path."""
    from cheddar.models import Bug

    with pytest.raises(ValidationError) as exc_info:
        Bug(
            file="",  # Must be non-empty
            line=1,
            type="off-by-one",
            description="test",
            original="x",
            injected="y",
        )
    assert "file" in str(exc_info.value)


def test_bug_manifest_valid_data(sample_bugs_json: dict[str, Any]) -> None:
    """Test BugManifest model with valid data."""
    from cheddar.models import BugManifest

    manifest = BugManifest(**sample_bugs_json)
    assert manifest.bug_count == 2
    assert len(manifest.bugs) == 2


def test_bug_manifest_invalid_bug_count() -> None:
    """Test BugManifest model rejects bug_count < 1."""
    from cheddar.models import BugManifest

    with pytest.raises(ValidationError) as exc_info:
        BugManifest(bug_count=0, bugs=[])
    assert "bug_count" in str(exc_info.value)


def test_review_report_valid() -> None:
    """Test ReviewReport model with valid data (no issues field)."""
    from cheddar.models import ReviewReport

    report = ReviewReport(
        agent="claude",
        raw_output="# Code Review\n\nFound an off-by-one error...",
        duration_seconds=5.5,
        status="complete",
    )
    assert report.agent == "claude"
    assert "off-by-one" in report.raw_output
    assert report.status == "complete"


def test_review_report_invalid_agent() -> None:
    """Test ReviewReport rejects invalid agent name."""
    from cheddar.models import ReviewReport

    with pytest.raises(ValidationError) as exc_info:
        ReviewReport(
            agent="invalid-agent",
            raw_output="",
            duration_seconds=0.0,
            status="complete",
        )
    assert "agent" in str(exc_info.value)


def test_review_report_invalid_status() -> None:
    """Test ReviewReport rejects invalid status."""
    from cheddar.models import ReviewReport

    with pytest.raises(ValidationError) as exc_info:
        ReviewReport(
            agent="claude",
            raw_output="",
            duration_seconds=0.0,
            status="invalid-status",
        )
    assert "status" in str(exc_info.value)


# BugMatch model tests (new model)


def test_bug_match_valid() -> None:
    """Test BugMatch model with valid data."""
    from cheddar.models import BugMatch

    match = BugMatch(
        bug_index=0,
        reasoning="Report explicitly identifies off-by-one error at main.py:10",
        matched_quote="off-by-one error at main.py line 10",
        found=True,
        confidence="high",
        error=False,
    )
    assert match.bug_index == 0
    assert match.found is True
    assert match.confidence == "high"
    assert "off-by-one" in match.reasoning


def test_bug_match_not_found() -> None:
    """Test BugMatch model for not found case."""
    from cheddar.models import BugMatch

    match = BugMatch(
        bug_index=1,
        reasoning="Report does not mention this file or bug type",
        matched_quote="",
        found=False,
        confidence="high",
        error=False,
    )
    assert match.found is False
    assert match.bug_index == 1


def test_bug_match_invalid_confidence() -> None:
    """Test BugMatch rejects invalid confidence values."""
    from cheddar.models import BugMatch

    with pytest.raises(ValidationError) as exc_info:
        BugMatch(
            bug_index=0,
            reasoning="test",
            matched_quote="",
            found=True,
            confidence="very high",  # Invalid: must be high/medium/low
            error=False,
        )
    assert "confidence" in str(exc_info.value)


def test_bug_match_field_order() -> None:
    """Test BugMatch has correct field order (reasoning first for chain-of-thought)."""
    from cheddar.models import BugMatch

    # Field order matters for LLM structured output
    fields = list(BugMatch.model_fields.keys())
    assert fields[0] == "bug_index"
    assert fields[1] == "finding_index"
    assert fields[2] == "reasoning"
    assert fields[3] == "matched_quote"
    assert fields[4] == "matched_line"
    assert fields[5] == "found"
    assert fields[6] == "confidence"


def test_global_bug_match_output_valid() -> None:
    """Test GlobalBugMatchOutput with valid assignment rows."""
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    output = GlobalBugMatchOutput(
        assignments=[
            GlobalBugMatchAssignment(
                bug_index=0,
                finding_index=1,
                reasoning="Exact match",
                matched_quote="off-by-one error",
                matched_line=10,
                confidence="high",
            )
        ]
    )
    assert len(output.assignments) == 1
    assert output.assignments[0].finding_index == 1


def test_global_bug_match_output_invalid_confidence() -> None:
    """Test GlobalBugMatchOutput rejects invalid confidence values."""
    from cheddar.models import GlobalBugMatchAssignment, GlobalBugMatchOutput

    with pytest.raises(ValidationError) as exc_info:
        GlobalBugMatchOutput(
            assignments=[
                GlobalBugMatchAssignment(
                    bug_index=0,
                    finding_index=1,
                    reasoning="test",
                    matched_quote="",
                    matched_line=0,
                    confidence="very-high",  # Invalid: must be high/medium/low
                )
            ]
        )
    assert "confidence" in str(exc_info.value)


# Score model tests (simplified - no precision)


def test_score_valid() -> None:
    """Test Score model with valid data (simplified, no precision)."""
    from cheddar.models import BugMatch, Score

    score = Score(
        reviewer="claude",
        total_bugs=2,
        bugs_found=1,
        detection_rate=0.5,
        model="gpt-4.1-mini",
        matches=[
            BugMatch(
                bug_index=0,
                reasoning="Found",
                matched_quote="found it",
                found=True,
                confidence="high",
                error=False,
            ),
            BugMatch(
                bug_index=1,
                reasoning="Not found",
                matched_quote="",
                found=False,
                confidence="high",
                error=False,
            ),
        ],
        match_errors=0,
    )
    assert score.reviewer == "claude"
    assert score.detection_rate == 0.5
    assert len(score.matches) == 2


def test_score_no_precision_fields() -> None:
    """Test Score model doesn't have precision-related fields."""
    from cheddar.models import Score

    # Verify precision fields are not in the model
    assert "precision" not in Score.model_fields
    assert "total_issues" not in Score.model_fields
    assert "true_positives" not in Score.model_fields
    assert "false_positives" not in Score.model_fields


def test_score_invalid_detection_rate() -> None:
    """Test Score model rejects detection_rate > 1."""
    from cheddar.models import BugMatch, Score

    with pytest.raises(ValidationError) as exc_info:
        Score(
            reviewer="claude",
            total_bugs=1,
            bugs_found=1,
            detection_rate=1.5,  # Invalid: > 1
            model="gpt-4.1-mini",
            matches=[
                BugMatch(
                    bug_index=0,
                    reasoning="Found",
                    matched_quote="found it",
                    found=True,
                    confidence="high",
                    error=False,
                )
            ],
            match_errors=0,
        )
    assert "detection_rate" in str(exc_info.value)


def test_run_config_valid() -> None:
    """Test ChallengeReport model with valid data."""
    from cheddar.models import ChallengeReport

    config = ChallengeReport(
        challenge_id="2026-01-25T14-30-00",
        repo="nanoid",
        challenger="claude",
        created_at=datetime(2026, 1, 25, 14, 30, 0),
        timeout_seconds=600,
        raw_output="ok",
        raw_stderr="",
        duration_seconds=12.5,
        status="complete",
        failure_reason=None,
    )
    assert config.challenge_id == "2026-01-25T14-30-00"
    assert config.challenger == "claude"
    assert config.timeout_seconds == 600


def test_run_config_invalid_challenger() -> None:
    """Test ChallengeReport rejects invalid challenger name."""
    from cheddar.models import ChallengeReport

    with pytest.raises(ValidationError) as exc_info:
        ChallengeReport(
            challenge_id="2026-01-25T14-30-00",
            repo="nanoid",
            challenger="invalid-agent",  # Invalid: not a known agent
            created_at=datetime(2026, 1, 25, 14, 30, 0),
            raw_output="ok",
            raw_stderr="",
            duration_seconds=1.0,
            status="complete",
            failure_reason=None,
        )
    assert "challenger" in str(exc_info.value)


def test_run_valid() -> None:
    """Test Challenge model with valid data."""
    from cheddar.models import Challenge, ChallengeReport

    config = ChallengeReport(
        challenge_id="2026-01-25T14-30-00",
        repo="nanoid",
        challenger="claude",
        created_at=datetime(2026, 1, 25, 14, 30, 0),
        raw_output="ok",
        raw_stderr="",
        duration_seconds=1.0,
        status="complete",
        failure_reason=None,
    )
    run = Challenge(config=config, status="challenged")
    assert run.config.challenge_id == "2026-01-25T14-30-00"
    assert run.status == "challenged"
    assert run.bugs is None
    assert run.reviews == {}
    assert run.scores is None


def test_run_invalid_status() -> None:
    """Test Challenge model rejects invalid status."""
    from cheddar.models import Challenge, ChallengeReport

    config = ChallengeReport(
        challenge_id="2026-01-25T14-30-00",
        repo="nanoid",
        challenger="claude",
        created_at=datetime(2026, 1, 25, 14, 30, 0),
        raw_output="ok",
        raw_stderr="",
        duration_seconds=1.0,
        status="complete",
        failure_reason=None,
    )
    with pytest.raises(ValidationError) as exc_info:
        Challenge(config=config, status="invalid-status")
    assert "status" in str(exc_info.value)


# Error class tests


def test_agent_error() -> None:
    """Test AgentError exception."""
    from cheddar.errors import AgentError

    error = AgentError(agent="claude", message="Something went wrong", raw_output="raw")
    assert error.agent == "claude"
    assert error.message == "Something went wrong"
    assert error.raw_output == "raw"
    assert "[claude]" in str(error)
    assert "Something went wrong" in str(error)


def test_agent_timeout_error() -> None:
    """Test AgentTimeoutError exception."""
    from cheddar.errors import AgentError, AgentTimeoutError

    error = AgentTimeoutError(agent="codex", message="Timed out after 600s")
    assert isinstance(error, AgentError)
    assert error.agent == "codex"
    assert "Timed out" in str(error)


def test_agent_parse_error() -> None:
    """Test AgentParseError exception."""
    from cheddar.errors import AgentError, AgentParseError

    error = AgentParseError(
        agent="gemini",
        message="Failed to parse JSON",
        raw_output='{"invalid": json}',
    )
    assert isinstance(error, AgentError)
    assert error.agent == "gemini"
    assert error.raw_output == '{"invalid": json}'
