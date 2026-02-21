"""Helpers for loading reviewer raw finding payloads."""

from __future__ import annotations

from pathlib import Path

from cheddar.models import RawFindingPayload, ReviewReport


def load_raw_finding_payloads(report: ReviewReport, challenge_dir: Path) -> list[RawFindingPayload]:
    """Load raw reviewer bugs/*.json payloads persisted in challenge artifacts."""
    raw_snapshot_rel = report.raw_findings_snapshot_dir
    if not raw_snapshot_rel:
        return []

    raw_dir = challenge_dir / raw_snapshot_rel
    if not raw_dir.is_dir():
        return []

    payloads: list[RawFindingPayload] = []
    for finding_file in sorted(raw_dir.glob("*.json")):
        try:
            content = finding_file.read_text()
        except OSError:
            continue
        payloads.append(RawFindingPayload(filename=finding_file.name, content=content))
    return payloads
