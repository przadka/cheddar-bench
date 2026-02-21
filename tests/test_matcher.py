"""Tests for matcher module using one-shot global bug-to-finding matching."""

from unittest.mock import patch

import pytest

from cheddar.models import (
    Bug,
    BugManifest,
    BugMatch,
    GlobalBugMatchAssignment,
    GlobalBugMatchOutput,
    RawFindingPayload,
    ReviewReport,
    Score,
)


@pytest.fixture
def sample_manifest() -> BugManifest:
    """Sample bug manifest with two injected bugs."""
    return BugManifest(
        bug_count=2,
        bugs=[
            Bug(
                file="main.py",
                line=10,
                type="off-by-one",
                description="Loop iterates one extra time",
                original="for i in range(n):",
                injected="for i in range(n + 1):",
            ),
            Bug(
                file="utils.py",
                line=25,
                type="null-handling",
                description="Missing null check",
                original="return data.value if data else None",
                injected="return data.value",
            ),
        ],
    )


@pytest.fixture
def sample_report_found() -> ReviewReport:
    """Sample review report metadata."""
    return ReviewReport(
        agent="claude",
        raw_output="unused by matcher",
        duration_seconds=10.0,
        status="complete",
    )


def test_match_bugs_to_raw_report_global_assignments(sample_manifest: BugManifest) -> None:
    """Raw matcher maps assignments using raw finding payload list."""
    from cheddar.matcher import match_bugs_to_raw_report

    raw_findings = [
        RawFindingPayload(
            filename="a.json",
            content='{"file":"main.py","line":10,"type":"off-by-one"}',
        ),
        RawFindingPayload(
            filename="b.json",
            content='{"file":"utils.py","line":25,"type":"null-handling"}',
        ),
    ]
    output = GlobalBugMatchOutput(
        assignments=[
            GlobalBugMatchAssignment(
                bug_index=0,
                finding_index=1,
                reasoning="raw exact match",
                matched_quote="main.py",
                matched_line=10,
                confidence="high",
            ),
            GlobalBugMatchAssignment(
                bug_index=1,
                finding_index=0,
                reasoning="no convincing raw match",
                matched_quote="",
                matched_line=0,
                confidence="low",
            ),
        ]
    )

    with patch("cheddar.matcher.complete_structured", return_value=output):
        results = match_bugs_to_raw_report(sample_manifest, raw_findings, model="azure/gpt-5.2")

    assert len(results) == 2
    assert results[0].found is True
    assert results[0].finding_index == 1
    assert results[1].found is False
    assert results[1].finding_index == 0


def test_match_bugs_to_raw_report_global_error_raises(sample_manifest: BugManifest) -> None:
    """Raw matcher failure raises RuntimeError and reports each bug to callback."""
    from cheddar.matcher import match_bugs_to_raw_report

    errors: list[int] = []
    raw_findings = [RawFindingPayload(filename="a.json", content='{"x":1}')]

    with (
        patch("cheddar.matcher.complete_structured", side_effect=RuntimeError("api down")),
        pytest.raises(RuntimeError, match="Global raw bug matching failed"),
    ):
        match_bugs_to_raw_report(
            sample_manifest,
            raw_findings,
            on_error=lambda idx, _exc: errors.append(idx),
        )

    assert errors == [0, 1]


def test_score_from_matches_basic(
    sample_manifest: BugManifest, sample_report_found: ReviewReport
) -> None:
    """Score calculation with one found and one missed bug."""
    from cheddar.matcher import score_from_matches

    matches = [
        BugMatch(
            bug_index=0,
            finding_index=1,
            reasoning="Found",
            matched_quote="found it",
            found=True,
            confidence="high",
            error=False,
        ),
        BugMatch(
            bug_index=1,
            finding_index=0,
            reasoning="Not found",
            matched_quote="",
            found=False,
            confidence="low",
            error=False,
        ),
    ]

    score = score_from_matches(sample_manifest, sample_report_found, matches)
    assert score.total_bugs == 2
    assert score.bugs_found == 1
    assert score.detection_rate == 0.5
    assert score.match_errors == 0


def test_score_from_matches_with_errors(
    sample_manifest: BugManifest, sample_report_found: ReviewReport
) -> None:
    """Errored matches count as not found in detection rate."""
    from cheddar.matcher import score_from_matches

    matches = [
        BugMatch(
            bug_index=0,
            finding_index=1,
            reasoning="Found",
            matched_quote="found it",
            found=True,
            confidence="high",
            error=False,
        ),
        BugMatch(
            bug_index=1,
            finding_index=0,
            reasoning="LLM error: timeout",
            matched_quote="",
            found=False,
            confidence="low",
            error=True,
        ),
    ]

    score = score_from_matches(sample_manifest, sample_report_found, matches)
    assert score.total_bugs == 2
    assert score.bugs_found == 1
    assert score.match_errors == 1
    assert score.detection_rate == 0.5


def test_select_median_score_by_bugs_found() -> None:
    """Median selection picks the center bugs_found score."""
    from cheddar.matcher import select_median_score

    scores = [
        Score(
            reviewer="claude",
            total_bugs=10,
            bugs_found=3,
            detection_rate=0.3,
            model="azure/gpt-5.2",
            matches=[],
            match_errors=0,
            manifest_hash="h",
        ),
        Score(
            reviewer="claude",
            total_bugs=10,
            bugs_found=7,
            detection_rate=0.7,
            model="azure/gpt-5.2",
            matches=[],
            match_errors=0,
            manifest_hash="h",
        ),
        Score(
            reviewer="claude",
            total_bugs=10,
            bugs_found=5,
            detection_rate=0.5,
            model="azure/gpt-5.2",
            matches=[],
            match_errors=0,
            manifest_hash="h",
        ),
    ]

    chosen = select_median_score(scores)
    assert chosen.bugs_found == 5


def test_select_median_score_index_tie_breaks_by_original_order() -> None:
    """Median index selection should use stable original-order tie breaks."""
    from cheddar.matcher import select_median_score_index

    scores = [
        Score(
            reviewer="claude",
            total_bugs=10,
            bugs_found=4,
            detection_rate=0.4,
            model="azure/gpt-5.2",
            matches=[],
            match_errors=0,
            manifest_hash="h",
        ),
        Score(
            reviewer="claude",
            total_bugs=10,
            bugs_found=6,
            detection_rate=0.6,
            model="azure/gpt-5.2",
            matches=[],
            match_errors=0,
            manifest_hash="h",
        ),
        Score(
            reviewer="claude",
            total_bugs=10,
            bugs_found=6,
            detection_rate=0.6,
            model="azure/gpt-5.2",
            matches=[],
            match_errors=0,
            manifest_hash="h",
        ),
    ]

    assert select_median_score_index(scores) == 1


def test_select_median_score_empty_raises() -> None:
    """Median selection should reject empty score lists."""
    from cheddar.matcher import select_median_score

    with pytest.raises(ValueError, match="must not be empty"):
        select_median_score([])
