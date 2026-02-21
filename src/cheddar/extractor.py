"""LLM-based bug manifest extraction from git diff and agent output."""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cheddar.errors import ExtractionError
from cheddar.llm import LLMError, complete_structured
from cheddar.models import Bug, BugManifest
from cheddar.prompts import load_prompt_pair

logger = logging.getLogger(__name__)

# Default model for extraction, can be overridden with CHEDDAR_EXTRACT_MODEL env var
_DEFAULT_EXTRACT_MODEL = os.environ.get("CHEDDAR_EXTRACT_MODEL", "gpt-4.1-mini")


def _empty_line_pairs() -> list[tuple[int, str]]:
    """Return a typed empty list of (line_number, content) tuples."""
    return []


@dataclass
class DiffChange:
    """A contiguous code change extracted from a unified diff."""

    file: str
    change_kind: str  # "modify", "delete", "add"
    removed: list[tuple[int, str]] = field(default_factory=_empty_line_pairs)
    added: list[tuple[int, str]] = field(default_factory=_empty_line_pairs)
    new_line: int = 0  # new-file line number for Bug.line


def parse_diff_changes(git_diff: str) -> list[DiffChange]:
    """Parse a unified diff into structured changes with exact line numbers.

    Walks each hunk, tracking old/new line counters to compute exact line numbers
    for every removed and added line. Groups contiguous -/+ lines into changes.
    """
    changes: list[DiffChange] = []
    current_file: str | None = None
    old_lineno = 0
    new_lineno = 0
    pending_removed: list[tuple[int, str]] = []
    pending_added: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal pending_removed, pending_added
        if not pending_removed and not pending_added:
            return
        assert current_file is not None
        if pending_removed and pending_added:
            kind = "modify"
        elif pending_removed:
            kind = "delete"
        else:
            kind = "add"
        nl = pending_added[0][0] if pending_added else new_lineno
        changes.append(
            DiffChange(
                file=current_file,
                change_kind=kind,
                removed=pending_removed[:],
                added=pending_added[:],
                new_line=nl,
            )
        )
        pending_removed = []
        pending_added = []

    for line in git_diff.splitlines():
        diff_match = re.match(r"^diff --git [a-z]/(.+?) [a-z]/(.+?)$", line)
        if diff_match:
            flush()
            current_file = diff_match.group(2)
            continue

        hunk_match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk_match:
            flush()
            old_lineno = int(hunk_match.group(1))
            new_lineno = int(hunk_match.group(2))
            continue

        if current_file is None:
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("index ") or line.startswith("Binary ") or line.startswith("\\"):
            continue

        if line.startswith(" "):
            flush()
            old_lineno += 1
            new_lineno += 1
        elif line.startswith("-"):
            pending_removed.append((old_lineno, line[1:]))
            old_lineno += 1
        elif line.startswith("+"):
            pending_added.append((new_lineno, line[1:]))
            new_lineno += 1

    flush()
    return changes


def format_line_map(changes: list[DiffChange]) -> str:
    """Format diff changes as a structured line map for prompt injection."""
    if not changes:
        return "(no changes detected)"
    lines: list[str] = []
    for i, change in enumerate(changes, 1):
        lines.append(f"CHANGE {i}: {change.file}:{change.new_line} [{change.change_kind}]")
        for old_no, content in change.removed:
            lines.append(f"  - (old {old_no}) {content}")
        if change.change_kind == "delete":
            lines.append("  + (deleted)")
        else:
            for new_no, content in change.added:
                lines.append(f"  + (new {new_no}) {content}")
        lines.append("")
    return "\n".join(lines)


def count_diff_hunks(git_diff: str) -> int:
    """Count the number of diff hunks in a git diff."""
    return len(re.findall(r"^@@ ", git_diff, re.MULTILINE))


def extract_diff_paths(git_diff: str) -> set[str]:
    """Extract all file paths from a git diff.

    Parses 'diff --git' lines and strips the a/ or c/ and b/ or i/ prefixes.

    Returns:
        Set of relative file paths that were modified.
    """
    paths: set[str] = set()
    # Match: diff --git a/path/to/file b/path/to/file
    # Or:    diff --git c/path/to/file i/path/to/file
    pattern = r"^diff --git [a-z]/(.+?) [a-z]/(.+?)$"
    for match in re.finditer(pattern, git_diff, re.MULTILINE):
        # Both paths should be the same, use the second one (the "b" or "i" side)
        paths.add(match.group(2))
    return paths


def correct_bug_path(bug: Bug, valid_paths: set[str], sandbox_path: Path) -> Bug:
    """Correct a bug's file path if it doesn't exist but a similar path does.

    The LLM sometimes drops intermediate directories. This function attempts
    to find the correct path by matching the filename.

    Returns:
        Bug with corrected file path if found, otherwise unchanged.
    """
    # If path already exists, no correction needed
    if (sandbox_path / bug.file).exists():
        return bug

    # Try to find correct path from valid_paths
    filename = Path(bug.file).name
    candidates = [p for p in valid_paths if Path(p).name == filename]

    if len(candidates) == 1:
        # Unique match - use it
        return bug.model_copy(update={"file": candidates[0]})

    # If file exists in sandbox but wasn't in valid_paths, keep original
    # Otherwise, no correction possible
    return bug


def filter_phantom_bugs(bugs: list[Bug]) -> tuple[list[Bug], list[Bug]]:
    """Remove bugs where original and injected code are identical.

    These "phantom" bugs appear when the LLM hallucinates a change that
    doesn't actually exist in the diff.

    Returns:
        Tuple of (kept_bugs, removed_bugs).
    """
    kept: list[Bug] = []
    removed: list[Bug] = []
    for bug in bugs:
        if bug.original.strip() == bug.injected.strip():
            logger.warning(
                "Phantom bug removed: %s:%d — original and injected are identical",
                bug.file,
                bug.line,
            )
            removed.append(bug)
        else:
            kept.append(bug)
    return kept, removed


def _parse_diff_by_file(git_diff: str) -> dict[str, tuple[str, str]]:
    """Parse a git diff into per-file removed/added content.

    Returns:
        Dict mapping file path to (removed_content, added_content) where each
        is the joined text of all -/+ lines (with leading -/+ stripped).
    """
    files: dict[str, tuple[list[str], list[str]]] = {}
    current_file: str | None = None

    for line in git_diff.splitlines():
        diff_match = re.match(r"^diff --git [a-z]/(.+?) [a-z]/(.+?)$", line)
        if diff_match:
            current_file = diff_match.group(2)
            assert current_file is not None  # group 2 always participates in this regex
            if current_file not in files:
                files[current_file] = ([], [])
            continue

        if current_file is None:
            continue

        if line.startswith("-") and not line.startswith("---"):
            files[current_file][0].append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            files[current_file][1].append(line[1:])
        elif line.startswith(" "):
            # Context lines are valid in both original and injected snippets
            files[current_file][0].append(line[1:])
            files[current_file][1].append(line[1:])

    return {
        path: ("\n".join(removed), "\n".join(added)) for path, (removed, added) in files.items()
    }


def _normalize(text: str) -> str:
    """Collapse all whitespace for fuzzy substring matching."""
    return " ".join(text.split())


def validate_bugs_against_diff(
    bugs: list[Bug], git_diff: str
) -> tuple[list[Bug], list[tuple[Bug, str]]]:
    """Filter bugs whose original/injected fields don't appear in the diff.

    Returns:
        Tuple of (valid_bugs, rejected) where rejected is a list of
        (bug, reason) pairs.
    """
    diff_content = _parse_diff_by_file(git_diff)
    valid: list[Bug] = []
    rejected: list[tuple[Bug, str]] = []

    for bug in bugs:
        file_content = diff_content.get(bug.file)
        if file_content is None:
            reason = f"file '{bug.file}' not found in diff"
            logger.warning("Bug rejected: %s:%d — %s", bug.file, bug.line, reason)
            rejected.append((bug, reason))
            continue

        removed, added = file_content

        if bug.original and _normalize(bug.original) not in _normalize(removed):
            reason = f"'original' not found in diff removed lines: '{bug.original[:100]}'"
            logger.warning("Bug rejected: %s:%d — %s", bug.file, bug.line, reason)
            rejected.append((bug, reason))
            continue

        # Skip injected check for deletion bugs — they have no added lines
        if (
            bug.change_kind != "delete"
            and bug.injected
            and _normalize(bug.injected) not in _normalize(added)
        ):
            reason = f"'injected' not found in diff added lines: '{bug.injected[:100]}'"
            logger.warning("Bug rejected: %s:%d — %s", bug.file, bug.line, reason)
            rejected.append((bug, reason))
            continue

        valid.append(bug)

    return valid, rejected


def get_git_diff(sandbox_path: Path) -> str:
    """Get git diff of all changes in the sandbox.

    Stages all files and returns the cached diff to capture all modifications.

    Args:
        sandbox_path: Path to the sandbox directory with a git repo.

    Returns:
        Git diff output as a string.

    Raises:
        ExtractionError: If git commands fail.
    """
    try:
        # Stage all changes (including new files)
        subprocess.run(
            ["git", "add", "-A"],
            cwd=sandbox_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # Get diff of staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "-U10"],
            cwd=sandbox_path,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout

    except subprocess.CalledProcessError as e:
        raise ExtractionError(
            message=f"Git command failed: {e.stderr}",
            git_diff="",
            raw_output="",
        ) from e


def _parse_manifest_direct(agent_bugs_json: str) -> BugManifest | None:
    """Try to parse agent JSON directly into a BugManifest.

    Returns None if JSON is malformed or doesn't match the schema.
    """
    try:
        data = json.loads(agent_bugs_json, strict=False)
        return BugManifest.model_validate(data)
    except (json.JSONDecodeError, ValueError, Exception):
        return None


def _parse_manifest_via_llm(
    agent_bugs_json: str,
    git_diff: str,
    expected_bug_count: int,
    model: str,
) -> tuple[BugManifest, str]:
    """Use LLM to repair/reformat agent JSON into a BugManifest.

    Returns (manifest, rendered_prompt). Used as fallback when direct parsing fails.
    """
    messages = load_prompt_pair(
        "extract-manifest",
        variables={
            "git_diff": git_diff,
            "agent_bugs_json": agent_bugs_json,
            "expected_bug_count": str(expected_bug_count),
        },
    )
    rendered_prompt = "\n\n---\n\n".join(f"## [{m['role']}]\n\n{m['content']}" for m in messages)

    manifest = complete_structured(
        model=model,
        messages=messages,
        response_model=BugManifest,
    )
    return manifest, rendered_prompt


def reformat_manifest(
    sandbox_path: Path,
    agent_bugs_json: str,
    model: str | None = None,
    expected_bug_count: int = 0,
) -> tuple[BugManifest, str, str]:
    """Parse agent bugs.json into a validated BugManifest.

    Tries direct JSON parsing first. Falls back to LLM only when the raw
    JSON is malformed or doesn't match the schema.

    Args:
        sandbox_path: Path to sandbox directory (for running git diff).
        agent_bugs_json: Raw content of the agent's bugs.json file.
        model: LLM model to use if repair is needed.
        expected_bug_count: Number of bugs the challenger was asked to inject.

    Returns:
        Tuple of (BugManifest, git_diff, rendered_prompt).

    Raises:
        ExtractionError: If extraction fails (no diff, LLM error, empty manifest).
    """
    git_diff = get_git_diff(sandbox_path)

    if not git_diff.strip():
        raise ExtractionError(
            message="Agent did not modify any files - no bugs were injected",
            git_diff="",
            raw_output=agent_bugs_json,
        )

    # Try direct parse — no LLM, no corruption risk
    manifest = _parse_manifest_direct(agent_bugs_json)
    rendered_prompt = ""
    used_llm = False

    if manifest and manifest.bugs:
        logger.info("Parsed %d bugs directly from agent JSON", len(manifest.bugs))
    else:
        # Fallback: LLM repairs malformed JSON / maps to schema
        used_llm = True
        logger.warning("Direct parse failed, falling back to LLM extraction")
        effective_model = model if model is not None else _DEFAULT_EXTRACT_MODEL
        try:
            manifest, rendered_prompt = _parse_manifest_via_llm(
                agent_bugs_json, git_diff, expected_bug_count, effective_model
            )
        except LLMError as e:
            raise ExtractionError(
                message=f"LLM reformatting failed: {e.message}",
                git_diff=git_diff,
                raw_output=agent_bugs_json,
            ) from e

        if not manifest.bugs:
            raise ExtractionError(
                message="LLM returned empty manifest - no bugs reformatted",
                git_diff=git_diff,
                raw_output=agent_bugs_json,
            )

    # Post-processing: path correction, phantom filter, diff validation
    valid_paths = extract_diff_paths(git_diff)
    bugs = [correct_bug_path(bug, valid_paths, sandbox_path) for bug in manifest.bugs]
    initial_count = len(bugs)

    bugs, phantom_bugs = filter_phantom_bugs(bugs)
    bugs, rejected_bugs = validate_bugs_against_diff(bugs, git_diff)

    manifest.bugs = bugs
    manifest.bug_count = len(bugs)

    # Summary logging
    source = "LLM" if used_llm else "direct parse"
    if initial_count != expected_bug_count and expected_bug_count > 0:
        logger.warning(
            "%s returned %d bugs but agent reported %d", source, initial_count, expected_bug_count
        )
    if phantom_bugs:
        logger.warning("Filtered %d phantom bugs (original == injected)", len(phantom_bugs))
    if rejected_bugs:
        logger.warning("Filtered %d bugs that didn't match diff", len(rejected_bugs))
    if manifest.bug_count != initial_count:
        logger.warning(
            "Bug count changed during validation: %d → %d", initial_count, manifest.bug_count
        )

    if not manifest.bugs:
        raise ExtractionError(
            message=(
                "All bugs were filtered out during validation"
                f" ({len(phantom_bugs)} phantom, {len(rejected_bugs)} rejected by diff)"
            ),
            git_diff=git_diff,
            raw_output=agent_bugs_json,
        )

    return manifest, git_diff, rendered_prompt
