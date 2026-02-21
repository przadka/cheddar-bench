"""Bug-to-report matching logic for scoring reviewers using one-shot global matching."""

import json
from collections.abc import Callable

from cheddar.llm import complete_structured
from cheddar.models import (
    BugManifest,
    BugMatch,
    GlobalBugMatchAssignment,
    GlobalBugMatchOutput,
    RawFindingPayload,
    ReviewReport,
    Score,
)
from cheddar.prompts import load_prompt_pair


def _build_bugs_payload(manifest: BugManifest) -> list[dict[str, str | int]]:
    """Build normalized bug payload for matcher prompts."""
    return [
        {
            "bug_index": idx,
            "file": bug.file,
            "line": bug.line,
            "type": bug.type,
            "description": bug.description,
            "injected": bug.injected,
        }
        for idx, bug in enumerate(manifest.bugs)
    ]


def _build_raw_match_messages(
    manifest: BugManifest,
    raw_findings: list[RawFindingPayload],
) -> list[dict[str, str]]:
    """Build messages for one-shot global assignment from raw finding payloads."""
    bugs_payload = _build_bugs_payload(manifest)
    findings_payload: list[dict[str, str | int]] = [
        {
            "finding_index": idx,
            "source_filename": payload.filename,
            "raw_finding_json": payload.content,
        }
        for idx, payload in enumerate(raw_findings, start=1)
    ]

    return load_prompt_pair(
        "match-all-bugs-report",
        variables={
            "bugs_json": json.dumps(bugs_payload, indent=2),
            "findings_json": json.dumps(findings_payload, indent=2),
        },
    )


def _assignments_to_matches(
    assignments: list[GlobalBugMatchAssignment], num_bugs: int
) -> list[BugMatch]:
    base_matches = [
        BugMatch(
            bug_index=idx,
            finding_index=0,
            reasoning="Not matched",
            matched_quote="",
            matched_line=0,
            found=False,
            confidence="low",
            error=False,
        )
        for idx in range(num_bugs)
    ]

    for assignment in assignments:
        if assignment.bug_index < 0 or assignment.bug_index >= num_bugs:
            continue
        found = assignment.finding_index > 0
        base_matches[assignment.bug_index] = BugMatch(
            bug_index=assignment.bug_index,
            finding_index=assignment.finding_index,
            reasoning=assignment.reasoning,
            matched_quote=assignment.matched_quote if found else "",
            matched_line=assignment.matched_line if found else 0,
            found=found,
            confidence=assignment.confidence,
            error=False,
        )
    return base_matches


def _run_global_match(
    messages: list[dict[str, str]],
    num_bugs: int,
    model: str,
    on_error: Callable[[int, Exception], None] | None,
    error_prefix: str,
) -> list[BugMatch]:
    """Run one global matching call and convert assignments to BugMatch rows."""
    base_matches = _assignments_to_matches(assignments=[], num_bugs=num_bugs)

    try:
        model_name = model.split("/", 1)[-1]
        output = complete_structured(
            model=model,
            messages=messages,
            response_model=GlobalBugMatchOutput,
            reasoning_effort="medium" if model_name.startswith("gpt-5") else None,
        )
        base_matches = _assignments_to_matches(assignments=output.assignments, num_bugs=num_bugs)
    except Exception as e:
        for idx in range(num_bugs):
            base_matches[idx] = BugMatch(
                bug_index=idx,
                finding_index=0,
                reasoning=f"LLM error: {e}",
                matched_quote="",
                matched_line=0,
                found=False,
                confidence="low",
                error=True,
            )
            if on_error:
                on_error(idx, e)
        raise RuntimeError(f"{error_prefix}: {e}") from e
    return base_matches


def match_bugs_to_raw_report(
    manifest: BugManifest,
    raw_findings: list[RawFindingPayload],
    model: str = "azure/gpt-5.2",
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[BugMatch]:
    """Match all bugs against raw findings using the standard assignment prompt."""
    messages = _build_raw_match_messages(manifest, raw_findings)
    return _run_global_match(
        messages=messages,
        num_bugs=len(manifest.bugs),
        model=model,
        on_error=on_error,
        error_prefix="Global raw bug matching failed",
    )


def score_from_matches(
    manifest: BugManifest,
    report: ReviewReport,
    matches: list[BugMatch],
    model: str = "azure/gpt-5.2",
    manifest_hash: str = "",
) -> Score:
    """Calculate detection metrics from match results. Pure function, no LLM calls.

    Args:
        manifest: Ground truth bugs.
        report: Reviewer's report.
        matches: Match results from match_bugs_to_raw_report.
        model: LLM model that was used for matching.
        manifest_hash: SHA-256 of bugs.json content for staleness detection.

    Returns:
        Score object with detection rate calculated.
    """
    total_bugs = len(manifest.bugs)
    assert total_bugs > 0, "BugManifest must have at least one bug"
    match_errors = sum(1 for m in matches if m.error)
    bugs_found = sum(1 for m in matches if m.found and not m.error)
    detection_rate = bugs_found / total_bugs

    return Score(
        reviewer=report.agent,
        total_bugs=total_bugs,
        bugs_found=bugs_found,
        detection_rate=detection_rate,
        model=model,
        matches=matches,
        match_errors=match_errors,
        manifest_hash=manifest_hash,
    )


def select_median_score(scores: list[Score]) -> Score:
    """Select a deterministic median score by bugs_found.

    For odd-length lists this picks the exact median. For even-length lists this picks the
    upper-middle element to avoid averaging and to preserve a concrete run artifact.
    """
    if not scores:
        raise ValueError("scores must not be empty")

    return scores[select_median_score_index(scores)]


def select_median_score_index(scores: list[Score]) -> int:
    """Select index of deterministic median score by bugs_found."""
    if not scores:
        raise ValueError("scores must not be empty")

    ranked = sorted(enumerate(scores), key=lambda item: (item[1].bugs_found, item[0]))
    median_idx = len(ranked) // 2
    return ranked[median_idx][0]
