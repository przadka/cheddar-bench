"""Tests for raw finding payload loading."""

from pathlib import Path
from unittest.mock import patch

from cheddar.findings import load_raw_finding_payloads
from cheddar.models import ReviewReport


def test_load_raw_finding_payloads_reads_sorted_json_files(tmp_path: Path) -> None:
    """Loader returns only .json files in deterministic order."""
    challenge_dir = tmp_path / "challenge"
    raw_dir = challenge_dir / "raw_findings" / "claude"
    raw_dir.mkdir(parents=True)

    (raw_dir / "b.json").write_text('{"id":2}')
    (raw_dir / "a.json").write_text('{"id":1}')
    (raw_dir / "note.txt").write_text("ignore")

    report = ReviewReport(
        agent="claude",
        raw_output="review",
        duration_seconds=1.0,
        status="complete",
        raw_findings_snapshot_dir="raw_findings/claude",
    )

    payloads = load_raw_finding_payloads(report=report, challenge_dir=challenge_dir)

    assert [payload.filename for payload in payloads] == ["a.json", "b.json"]
    assert [payload.content for payload in payloads] == ['{"id":1}', '{"id":2}']


def test_load_raw_finding_payloads_skips_unreadable_json(tmp_path: Path) -> None:
    """Loader skips JSON files that raise OSError during read."""
    challenge_dir = tmp_path / "challenge"
    raw_dir = challenge_dir / "raw_findings" / "codex"
    raw_dir.mkdir(parents=True)

    ok_file = raw_dir / "ok.json"
    bad_file = raw_dir / "bad.json"
    ok_file.write_text('{"ok":true}')
    bad_file.write_text('{"bad":true}')

    report = ReviewReport(
        agent="codex",
        raw_output="review",
        duration_seconds=1.0,
        status="complete",
        raw_findings_snapshot_dir="raw_findings/codex",
    )

    original_read_text = Path.read_text

    def flaky_read_text(path: Path, *_args: object, **_kwargs: object) -> str:
        if path == bad_file:
            raise OSError("read failed")
        return original_read_text(path)

    with patch.object(Path, "read_text", autospec=True, side_effect=flaky_read_text):
        payloads = load_raw_finding_payloads(report=report, challenge_dir=challenge_dir)

    assert len(payloads) == 1
    assert payloads[0].filename == "ok.json"
