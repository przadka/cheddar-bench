"""Tests for sandbox creation and management."""

from pathlib import Path

import pytest


def test_create_challenge_sandbox(tmp_repo: Path, tmp_path: Path) -> None:
    """Test creating a sandbox for challenge phase."""
    from cheddar.sandbox import cleanup_sandbox, create_challenge_sandbox

    sandbox_path = create_challenge_sandbox(
        repo_path=tmp_repo,
        challenge_dir=tmp_path / "run",
    )

    try:
        # Sandbox should exist
        assert sandbox_path.exists()
        assert sandbox_path.is_dir()

        # Sandbox should be in /tmp with random suffix
        assert "sandbox-" in sandbox_path.name

        # Repository content should be copied
        assert (sandbox_path / "main.py").exists()

        # Should have .git directory (for agent compatibility)
        assert (sandbox_path / ".git").exists()
    finally:
        cleanup_sandbox(sandbox_path)


def test_create_challenge_sandbox_creates_challenge_dir(tmp_repo: Path, tmp_path: Path) -> None:
    """Test that challenge sandbox creates run directory if needed."""
    from cheddar.sandbox import cleanup_sandbox, create_challenge_sandbox

    challenge_dir = tmp_path / "nonexistent" / "run"
    sandbox_path = create_challenge_sandbox(
        repo_path=tmp_repo,
        challenge_dir=challenge_dir,
    )

    try:
        assert sandbox_path.exists()
        assert challenge_dir.exists()
    finally:
        cleanup_sandbox(sandbox_path)


def test_create_challenge_sandbox_nonexistent_repo(tmp_path: Path) -> None:
    """Test error when repo doesn't exist."""
    from cheddar.sandbox import SandboxError, create_challenge_sandbox

    with pytest.raises(SandboxError) as exc_info:
        create_challenge_sandbox(
            repo_path=tmp_path / "nonexistent",
            challenge_dir=tmp_path / "run",
        )
    assert "not found" in str(exc_info.value).lower()


def test_cleanup_sandbox(tmp_repo: Path, tmp_path: Path) -> None:
    """Test cleaning up a sandbox."""
    from cheddar.sandbox import cleanup_sandbox, create_challenge_sandbox

    sandbox_path = create_challenge_sandbox(
        repo_path=tmp_repo,
        challenge_dir=tmp_path / "run",
    )

    assert sandbox_path.exists()

    cleanup_sandbox(sandbox_path)

    assert not sandbox_path.exists()


def test_cleanup_nonexistent_sandbox(tmp_path: Path) -> None:
    """Test cleanup of non-existent sandbox doesn't error."""
    from cheddar.sandbox import cleanup_sandbox

    # Should not raise
    cleanup_sandbox(tmp_path / "nonexistent")


def test_get_sandbox_path_format() -> None:
    """Test sandbox path follows expected format."""
    import tempfile

    from cheddar.sandbox import cleanup_sandbox, get_sandbox_path

    path = get_sandbox_path()

    try:
        # Should be in system temp directory
        assert path.parent == Path(tempfile.gettempdir())

        # Should have cheddar-sandbox- prefix and random suffix
        assert path.name.startswith("cheddar-sandbox-")
        assert len(path.name) > len("cheddar-sandbox-")  # Has random suffix

        # Directory should already exist (mkdtemp creates it)
        assert path.exists()
    finally:
        cleanup_sandbox(path)


def test_persist_sandbox(tmp_repo: Path, tmp_path: Path) -> None:
    """Test persisting a sandbox to run directory."""
    from cheddar.sandbox import cleanup_sandbox, create_challenge_sandbox, persist_sandbox

    challenge_dir = tmp_path / "run"
    challenge_dir.mkdir()

    sandbox_path = create_challenge_sandbox(
        repo_path=tmp_repo,
        challenge_dir=challenge_dir,
    )

    try:
        # Modify sandbox content (simulating agent changes)
        (sandbox_path / "modified.txt").write_text("agent was here")

        # Persist sandbox
        persisted_path = persist_sandbox(sandbox_path, challenge_dir)

        # Persisted path should be challenge_dir/repo
        assert persisted_path == challenge_dir / "repo"
        assert persisted_path.exists()

        # Original content should be preserved
        assert (persisted_path / "main.py").exists()

        # Modified content should be preserved
        assert (persisted_path / "modified.txt").exists()
        assert (persisted_path / "modified.txt").read_text() == "agent was here"
    finally:
        cleanup_sandbox(sandbox_path)


def test_persist_sandbox_overwrites_existing(tmp_repo: Path, tmp_path: Path) -> None:
    """Test that persist_sandbox overwrites existing repo dir."""
    from cheddar.sandbox import cleanup_sandbox, create_challenge_sandbox, persist_sandbox

    challenge_dir = tmp_path / "run"
    challenge_dir.mkdir()

    # Create an existing repo dir
    existing_repo = challenge_dir / "repo"
    existing_repo.mkdir()
    (existing_repo / "old_file.txt").write_text("should be removed")

    sandbox_path = create_challenge_sandbox(
        repo_path=tmp_repo,
        challenge_dir=challenge_dir,
    )

    try:
        persisted_path = persist_sandbox(sandbox_path, challenge_dir)

        # Old content should be gone
        assert not (persisted_path / "old_file.txt").exists()

        # New content should be present
        assert (persisted_path / "main.py").exists()
    finally:
        cleanup_sandbox(sandbox_path)


def test_create_review_sandbox(tmp_repo: Path, tmp_path: Path) -> None:
    """Test creating a review sandbox from persisted repo."""
    from cheddar.sandbox import (
        cleanup_sandbox,
        create_challenge_sandbox,
        create_review_sandbox,
        persist_sandbox,
    )

    challenge_dir = tmp_path / "run"
    challenge_dir.mkdir()

    # First, challenge phase: create and persist sandbox
    challenge_sandbox = create_challenge_sandbox(
        repo_path=tmp_repo,
        challenge_dir=challenge_dir,
    )

    # Simulate agent injecting bugs
    (challenge_sandbox / "bugs.json").write_text('{"bugs": []}')
    (challenge_sandbox / "injected_bug.py").write_text("# bug here")

    # Persist the sandbox
    persist_sandbox(challenge_sandbox, challenge_dir)
    cleanup_sandbox(challenge_sandbox)

    # Now review phase: create review sandbox
    review_sandbox = create_review_sandbox(challenge_dir)

    # Should exist
    assert review_sandbox.exists()

    # Should have injected content
    assert (review_sandbox / "injected_bug.py").exists()

    # bugs.json should be stripped (ground truth leakage prevention)
    assert not (review_sandbox / "bugs.json").exists()

    cleanup_sandbox(review_sandbox)


def test_create_review_sandbox_no_persisted_repo(tmp_path: Path) -> None:
    """Test error when persisted repo doesn't exist."""
    from cheddar.sandbox import SandboxError, create_review_sandbox

    challenge_dir = tmp_path / "run"
    challenge_dir.mkdir()

    with pytest.raises(SandboxError) as exc_info:
        create_review_sandbox(challenge_dir)

    assert "not found" in str(exc_info.value).lower()
    assert "challenge" in str(exc_info.value).lower()
