"""Backfill missing Gemini reviews with retry/backoff and logging.

Writes logs under artifacts/ and only creates missing review artifacts under
challenges/<id>/reviews/gemini.json via `cheddar review` commands.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def _count_missing(challenges_dir: Path) -> tuple[int, int, int]:
    present = 0
    complete = 0
    missing = 0
    for d in challenges_dir.iterdir():
        if not d.is_dir() or not (d / "bugs.json").exists() or "-" not in d.name:
            continue
        review_file = d / "reviews" / "gemini.json"
        if not review_file.exists():
            missing += 1
            continue
        present += 1
        try:
            if json.loads(review_file.read_text()).get("status") == "complete":
                complete += 1
        except Exception:
            pass
    return present, complete, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenges-dir",
        type=Path,
        default=Path.cwd() / "challenges",
        help="Challenges directory to process.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-review timeout in seconds.",
    )
    parser.add_argument(
        "--max-enotfound-streak",
        type=int,
        default=12,
        help="Stop after this many consecutive ENOTFOUND failures.",
    )
    parser.add_argument(
        "--sleep-enotfound",
        type=int,
        default=120,
        help="Sleep seconds after ENOTFOUND failures.",
    )
    parser.add_argument(
        "--sleep-429",
        type=int,
        default=600,
        help="Sleep seconds after RESOURCE_EXHAUSTED/429 failures.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    challenges_dir = args.challenges_dir.resolve()
    if not challenges_dir.is_dir():
        raise SystemExit(f"Challenges dir not found: {challenges_dir}")

    run_id = datetime.now(UTC).strftime("gemini-backfill-%Y%m%d-%H%M%SZ")
    log_path = repo_root / "artifacts" / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0523046142"
    env["GOOGLE_CLOUD_LOCATION"] = "global"
    env["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

    def w(msg: str) -> None:
        line = f"[{datetime.now().strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    challenge_ids = sorted(
        d.name
        for d in challenges_dir.iterdir()
        if d.is_dir() and (d / "bugs.json").exists() and "-" in d.name
    )
    missing = [
        cid
        for cid in challenge_ids
        if not (challenges_dir / cid / "reviews" / "gemini.json").exists()
    ]
    w(f"START total={len(challenge_ids)} missing={len(missing)} log={log_path}")

    ok = 0
    fail = 0
    enotfound_streak = 0
    quota_fails = 0

    for idx, cid in enumerate(missing, 1):
        repo = cid.split("-", 1)[1]
        cmd = [
            "uv",
            "run",
            "cheddar",
            "review",
            "gemini",
            repo,
            "-c",
            cid,
            "--challenges-dir",
            str(challenges_dir),
            "--timeout",
            str(args.timeout),
        ]

        w(f"{idx}/{len(missing)} RUN {cid}")
        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout + 300,
        )
        dur = time.time() - start

        out = proc.stdout or ""
        err = proc.stderr or ""
        blob = f"{out}\n{err}".lower()
        tail = (err[-400:] if err else out[-400:]).replace("\n", " ")

        review_file = challenges_dir / cid / "reviews" / "gemini.json"
        if proc.returncode == 0 and review_file.exists():
            ok += 1
            enotfound_streak = 0
            w(f"OK {cid} dur={dur:.1f}s")
            continue

        fail += 1
        w(f"FAIL {cid} rc={proc.returncode} dur={dur:.1f}s tail={tail}")

        if "enotfound" in blob or "getaddrinfo" in blob:
            enotfound_streak += 1
            w(f"NET_BACKOFF streak={enotfound_streak} sleep={args.sleep_enotfound}s")
            time.sleep(args.sleep_enotfound)
            if enotfound_streak >= args.max_enotfound_streak:
                w("EARLY_STOP max ENOTFOUND streak reached")
                break
            continue

        enotfound_streak = 0
        if "resource_exhausted" in blob or "429" in blob:
            quota_fails += 1
            w(f"QUOTA_BACKOFF count={quota_fails} sleep={args.sleep_429}s")
            time.sleep(args.sleep_429)

    present, complete, missing_now = _count_missing(challenges_dir)
    w(
        "DONE "
        f"ok={ok} fail={fail} quota_fails={quota_fails} "
        f"snapshot_present={present} complete={complete} missing={missing_now}"
    )
    print(f"LOG={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
