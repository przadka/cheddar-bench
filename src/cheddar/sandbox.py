"""Sandbox creation and management for benchmark challenges."""

import shutil
import subprocess
import tempfile
from pathlib import Path


class SandboxError(Exception):
    """Error during sandbox operations."""

    pass


def get_sandbox_path() -> Path:
    """Create a unique sandbox directory in the system temp location.

    Returns:
        Path to a newly created sandbox directory.
    """
    return Path(tempfile.mkdtemp(prefix="cheddar-sandbox-"))


def create_challenge_sandbox(
    repo_path: Path,
    challenge_dir: Path,
) -> Path:
    """Create a sandbox for the challenge phase.

    Args:
        repo_path: Path to the source repository.
        challenge_dir: Path to the challenge directory for storing results.

    Returns:
        Path to the created sandbox.

    Raises:
        SandboxError: If sandbox creation fails.
    """
    if not repo_path.exists():
        raise SandboxError(f"Repository not found: {repo_path}")

    # Create challenge directory if needed
    challenge_dir.mkdir(parents=True, exist_ok=True)

    # Create sandbox in system temp directory
    sandbox_path = get_sandbox_path()

    try:
        # Copy repository to sandbox, excluding .git (we'll init fresh).
        # symlinks=True preserves symlinks instead of dereferencing host files.
        shutil.copytree(
            repo_path,
            sandbox_path,
            symlinks=True,
            dirs_exist_ok=True,  # sandbox_path already created by mkdtemp
            ignore=shutil.ignore_patterns(".git"),
        )

        # Initialize fresh git repo for agent compatibility.
        # Disable global hooks (core.hooksPath=/dev/null) to prevent external
        # hooks (e.g., beads pre-commit) from failing on this throwaway repo.
        subprocess.run(
            ["git", "init"],
            cwd=sandbox_path,
            capture_output=True,
            check=True,
        )

        # Create baseline commit so git diff shows only agent changes
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "add", "-A"],
            cwd=sandbox_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "user.name=cheddar",
                "-c",
                "user.email=cheddar@localhost",
                "commit",
                "--allow-empty",
                "-m",
                "baseline",
            ],
            cwd=sandbox_path,
            capture_output=True,
            check=True,
        )

        return sandbox_path

    except Exception as e:
        # Clean up on failure
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
        raise SandboxError(f"Failed to create sandbox: {e}") from e


def create_review_sandbox(challenge_dir: Path) -> Path:
    """Create a sandbox for the review phase.

    Uses the persisted repo from the challenge phase (challenges/<challenge-id>/repo)
    and strips ground truth files to prevent leakage.

    Args:
        challenge_dir: Path to the challenge directory containing the persisted repo.

    Returns:
        Path to the created sandbox.

    Raises:
        SandboxError: If sandbox creation fails or persisted repo not found.
    """
    persisted_repo = challenge_dir / "repo"
    if not persisted_repo.exists():
        raise SandboxError(
            f"Persisted repo not found: {persisted_repo}. Run 'cheddar challenge' first."
        )

    # Create sandbox in system temp directory
    sandbox_path = get_sandbox_path()

    try:
        # Copy persisted repo WITHOUT .git to prevent leaking which files
        # were modified by the challenger. Without this, reviewers can just
        # run `git diff` to see all injected bugs.
        shutil.copytree(
            persisted_repo,
            sandbox_path,
            symlinks=True,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git"),
        )

        # Strip ground truth files to prevent leakage
        for ground_truth_file in ["bugs.json"]:
            gt_path = sandbox_path / ground_truth_file
            if gt_path.exists():
                gt_path.unlink()

        # Initialize fresh git repo so mutation detection (git status) works
        subprocess.run(
            ["git", "init"],
            cwd=sandbox_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "add", "-A"],
            cwd=sandbox_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "user.name=cheddar",
                "-c",
                "user.email=cheddar@localhost",
                "commit",
                "--allow-empty",
                "-m",
                "review-baseline",
            ],
            cwd=sandbox_path,
            capture_output=True,
            check=True,
        )

        return sandbox_path

    except Exception as e:
        # Clean up on failure
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
        raise SandboxError(f"Failed to create review sandbox: {e}") from e


def persist_sandbox(sandbox_path: Path, challenge_dir: Path) -> Path:
    """Persist a sandbox to the challenge directory.

    Copies the mutated sandbox to challenges/<challenge-id>/repo for later review/matching.

    Args:
        sandbox_path: Path to the sandbox to persist.
        challenge_dir: Path to the challenge directory.

    Returns:
        Path to the persisted repo in challenge_dir.

    Raises:
        SandboxError: If persistence fails.
    """
    persisted_path = challenge_dir / "repo"
    try:
        if persisted_path.exists():
            shutil.rmtree(persisted_path)
        shutil.copytree(sandbox_path, persisted_path, symlinks=True)
        return persisted_path
    except Exception as e:
        raise SandboxError(f"Failed to persist sandbox: {e}") from e


def cleanup_sandbox(sandbox_path: Path) -> None:
    """Remove a sandbox directory.

    Args:
        sandbox_path: Path to the sandbox to clean up.
    """
    if sandbox_path.exists():
        shutil.rmtree(sandbox_path)
