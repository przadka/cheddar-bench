"""Cheddar-bench CLI application."""

import os
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from cheddar import __version__
from cheddar.models import DEFAULT_TIMEOUT_SECONDS

# Initialize Typer app
app = typer.Typer(
    name="cheddar",
    help="Benchmark for evaluating CLI coding agents on bug detection.",
    no_args_is_help=True,
)

# Rich console for output
console = Console()
err_console = Console(stderr=True)


class AppState:
    """Global application state."""

    def __init__(self) -> None:
        self.verbose: bool = False

    def get_challenges_dir(self) -> Path:
        """Get the challenges directory path."""
        env_dir = os.environ.get("CHEDDAR_CHALLENGES_DIR")
        if env_dir:
            return Path(env_dir)
        return Path.cwd() / "challenges"

    def get_repos_dir(self) -> Path:
        """Get the repos directory path."""
        env_dir = os.environ.get("CHEDDAR_REPOS_DIR")
        if env_dir:
            return Path(env_dir)
        return Path.cwd() / "repos"


state = AppState()


# File extensions to count as source code (exclude tests, configs, docs)
SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".rb",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
}
EXCLUDE_PATTERNS = {
    "test",
    "tests",
    "spec",
    "specs",
    "__tests__",
    "t",
    "__pycache__",
    "node_modules",
    ".git",
    "vendor",
}

# Bug count scaling: 1 bug per LOC_PER_BUG lines, clamped to [MIN, MAX]
MIN_BUGS_PER_CHALLENGE = 5
MAX_BUGS_PER_CHALLENGE = 24
LOC_PER_BUG = 100


def count_loc(repo_path: Path) -> int:
    """Count lines of source code in a repository."""
    total = 0
    for file in repo_path.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix not in SOURCE_EXTENSIONS:
            continue
        # Check path components instead of substring matching
        # This avoids false positives like "latest.py" matching "test"
        if any(excl in file.parts for excl in EXCLUDE_PATTERNS):
            continue
        try:
            total += len(file.read_text().splitlines())
        except (UnicodeDecodeError, PermissionError):
            continue
    return total


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"cheddar {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output."),
    ] = False,
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Cheddar-bench: Benchmark CLI coding agents on bug detection."""
    _ = _version  # Option is consumed by Typer callback; keep parameter for CLI contract.
    _load_dotenv()
    state.verbose = verbose


def _load_dotenv() -> None:
    """Load .env from project root if it exists (no dependency needed)."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key and _ and key not in os.environ:
            os.environ[key] = value


# List command resource types
class ListResource(str, Enum):
    """Resources that can be listed."""

    AGENTS = "agents"
    REPOS = "repos"


# Derive agent lists from registry to avoid duplication
def _get_challenger_agents() -> list[str]:
    from cheddar.agents import get_challengers

    return get_challengers()


def _get_reviewer_agents() -> list[str]:
    from cheddar.agents import get_reviewers

    return get_reviewers()


@app.command(name="list")
def list_command(
    resource: Annotated[
        ListResource | None,
        typer.Argument(help="Resource to list: agents or repos"),
    ] = None,
) -> None:
    """List available agents or repositories."""
    if resource is None:
        # List all resources
        _list_agents()
        console.print()
        _list_repos()
    elif resource == ListResource.AGENTS:
        _list_agents()
    elif resource == ListResource.REPOS:
        _list_repos()


def _list_agents() -> None:
    """Display agents table."""
    table = Table(title="Available Agents")
    table.add_column("Agent", style="cyan")
    table.add_column("Can Challenge", style="green")
    table.add_column("Can Review", style="green")

    challengers = set(_get_challenger_agents())
    for agent in _get_reviewer_agents():
        can_challenge = "Yes" if agent in challengers else "No"
        table.add_row(agent, can_challenge, "Yes")

    console.print(table)


def _get_repos() -> list[str]:
    """Get list of available repositories."""
    repos_dir = state.get_repos_dir()
    if not repos_dir.exists():
        return []
    return sorted(d.name for d in repos_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


def _list_repos() -> None:
    """Display repos table."""
    repos = _get_repos()
    table = Table(title="Available Repositories")
    table.add_column("Repository", style="cyan")

    if not repos:
        console.print("[yellow]No repositories found in repos/[/yellow]")
        return

    for repo in repos:
        table.add_row(repo)

    console.print(table)


def _persist_raw_review_findings(
    sandbox_path: Path,
    challenge_dir: Path,
    reviewer: str,
) -> str | None:
    """Persist raw reviewer bugs/*.json before sandbox cleanup.

    Returns a challenge-dir relative path when raw findings were copied,
    otherwise None.
    """
    source_dir = sandbox_path / "bugs"
    if not source_dir.is_dir():
        return None

    target_dir = challenge_dir / "raw_findings" / reviewer
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return str(target_dir.relative_to(challenge_dir))


# Challenge command


class ChallengerAgent(str, Enum):
    """Agents that can challenge (inject bugs)."""

    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


@app.command()
def challenge(
    agent: Annotated[
        ChallengerAgent,
        typer.Argument(help="Challenger agent: claude, codex, or gemini"),
    ],
    repo: Annotated[
        str,
        typer.Argument(help="Repository name from repos/"),
    ],
    timeout: Annotated[
        int,
        typer.Option("--timeout", "-t", help="Agent timeout in seconds"),
    ] = DEFAULT_TIMEOUT_SECONDS,
    challenge_id: Annotated[
        str | None,
        typer.Option("--challenge-id", help="Custom challenge ID (default: <agent>-<repo>)"),
    ] = None,
    challenges_dir: Annotated[
        Path | None,
        typer.Option("--challenges-dir", help="Custom challenges directory (default: challenges/)"),
    ] = None,
) -> None:
    """Inject bugs into a repository using an agent."""
    from datetime import datetime

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from cheddar.agents import get_agent
    from cheddar.errors import AgentError, AgentTimeoutError
    from cheddar.extractor import reformat_manifest
    from cheddar.models import ChallengeReport
    from cheddar.prompts import load_prompt
    from cheddar.sandbox import cleanup_sandbox, create_challenge_sandbox, persist_sandbox

    # Validate repository exists
    repos_dir = state.get_repos_dir()
    repo_path = repos_dir / repo
    if not repo_path.exists():
        err_console.print(f"[red]Error: Repository '{repo}' not found in {repos_dir}[/red]")
        err_console.print(f"Available repositories: {_get_repos()}")
        raise typer.Exit(2)

    # Generate challenge ID if not provided
    if challenge_id is None:
        challenge_id = f"{agent.value}-{repo}"

    # Determine challenges directory (explicit option overrides default)
    effective_challenges_dir = (
        challenges_dir if challenges_dir is not None else state.get_challenges_dir()
    )
    challenge_dir = effective_challenges_dir / challenge_id
    challenge_dir.mkdir(parents=True, exist_ok=True)

    # Create challenge report (updated after agent execution)
    config = ChallengeReport(
        challenge_id=challenge_id,
        repo=repo,
        challenger=agent.value,
        created_at=datetime.now(),
        timeout_seconds=timeout,
        raw_output="",
        raw_stderr="",
        duration_seconds=0.0,
        status="failed",
        failure_reason="Challenge not completed",
    )

    # Save config
    config_file = challenge_dir / "config.json"
    config_file.write_text(config.model_dump_json(indent=2))

    if state.verbose:
        console.print(f"[dim]Created challenge directory: {challenge_dir}[/dim]")

    # Create sandbox
    sandbox_path = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Create sandbox
            task = progress.add_task("Creating sandbox...", total=None)
            sandbox_path = create_challenge_sandbox(repo_path, challenge_dir)
            progress.update(task, description="Sandbox created")

            if state.verbose:
                console.print(f"[dim]Sandbox: {sandbox_path}[/dim]")

            # Count LOC and calculate bug count (capped at 24 to keep challenges reasonable)
            progress.update(task, description="Counting LOC...")
            loc = count_loc(sandbox_path)
            bug_count = min(MAX_BUGS_PER_CHALLENGE, max(MIN_BUGS_PER_CHALLENGE, loc // LOC_PER_BUG))

            if state.verbose:
                console.print(f"[dim]LOC: {loc}, bugs to inject: {bug_count}[/dim]")

            # Load prompt
            progress.update(task, description="Loading prompt...")
            prompt = load_prompt(
                "inject-bugs",
                variables={"repo_path": str(sandbox_path), "bug_count": str(bug_count)},
            )

            # Run agent (retry once with fewer bugs if agent fails to produce output)
            agent_instance = get_agent(agent.value)
            max_attempts = 2
            agent_bugs_json = ""
            for attempt in range(max_attempts):
                if attempt > 0:
                    bug_count = max(MIN_BUGS_PER_CHALLENGE, bug_count * 2 // 3)
                    console.print(
                        f"[yellow]Retrying with {bug_count} bugs (attempt {attempt + 1})...[/yellow]"
                    )
                    cleanup_sandbox(sandbox_path)
                    sandbox_path = create_challenge_sandbox(repo_path, challenge_dir)
                    prompt = load_prompt(
                        "inject-bugs",
                        variables={"repo_path": str(sandbox_path), "bug_count": str(bug_count)},
                    )

                progress.update(task, description=f"Running {agent.value} agent...")
                challenge_start = time.time()
                config.raw_output = ""
                config.raw_stderr = ""
                raw_output, raw_stderr = agent_instance.challenge(
                    sandbox_path=sandbox_path,
                    prompt=prompt,
                    timeout_seconds=timeout,
                )
                config.duration_seconds = time.time() - challenge_start
                config.raw_output = raw_output
                config.raw_stderr = raw_stderr
                config.status = "complete"
                config.failure_reason = None
                config_file.write_text(config.model_dump_json(indent=2))

                # Persist immediately so artifacts survive extraction failures
                progress.update(task, description="Persisting sandbox...")
                persisted_repo = persist_sandbox(sandbox_path, challenge_dir)

                (challenge_dir / "raw_challenge_output.txt").write_text(raw_output)
                if raw_stderr:
                    (challenge_dir / "raw_challenge_stderr.txt").write_text(raw_stderr)
                (challenge_dir / "prompt_challenge.md").write_text(prompt)

                if state.verbose:
                    console.print(f"[dim]Persisted repo: {persisted_repo}[/dim]")

                # Validate: bugs.json must exist and not be empty.
                # Strict JSON validation is left to the extractor.
                agent_bugs_file = sandbox_path / "bugs.json"
                if agent_bugs_file.exists():
                    agent_bugs_json = agent_bugs_file.read_text().strip()
                    if agent_bugs_json:
                        break  # let the extractor handle parsing/cleanup
                    reason = "bugs.json is empty"
                else:
                    reason = "bugs.json not produced"

                if attempt < max_attempts - 1:
                    console.print(f"[yellow]{reason} — will retry with fewer bugs[/yellow]")
                else:
                    raise AgentError(
                        agent=agent.value,
                        message=f"Challenge failed after {max_attempts} attempts: {reason}",
                        raw_output=raw_output,
                    )

            # Save raw agent bugs.json before reformatting
            (challenge_dir / "bugs_raw.json").write_text(agent_bugs_json)

            # Reformat agent's bugs.json into validated BugManifest
            progress.update(task, description="Reformatting bug manifest...")
            manifest, git_diff, extract_prompt = reformat_manifest(
                sandbox_path=sandbox_path,
                agent_bugs_json=agent_bugs_json,
                expected_bug_count=bug_count,
            )

        # Save validated manifest and artifacts
        bugs_file = challenge_dir / "bugs.json"
        bugs_file.write_text(manifest.model_dump_json(indent=2))
        (challenge_dir / "git_diff.txt").write_text(git_diff)
        (challenge_dir / "prompt_extract.md").write_text(extract_prompt)

        # Display results
        console.print("[green]Challenge complete![/green]")
        console.print(f"  Challenge ID: {challenge_id}")
        console.print(f"  Bugs injected: {manifest.bug_count}")
        console.print(f"  Duration: {config.duration_seconds:.1f}s")
        console.print(f"  Output: {bugs_file}")

    except Exception as e:
        if isinstance(e, AgentTimeoutError):
            config.status = "timeout"
        elif isinstance(e, AgentError):
            config.status = "error"
            if e.raw_output:
                config.raw_output = e.raw_output
        else:
            config.status = "failed"
        config.failure_reason = str(e)
        config_file.write_text(config.model_dump_json(indent=2))

        err_console.print(f"[red]Error: {e}[/red]")
        if state.verbose and hasattr(e, "raw_output"):
            raw_out = getattr(e, "raw_output", "")
            err_console.print(f"[dim]Raw output: {raw_out[:500]}...[/dim]")
        raise typer.Exit(1) from e

    finally:
        # Cleanup sandbox
        if sandbox_path:
            cleanup_sandbox(sandbox_path)
            if state.verbose:
                console.print("[dim]Cleaned up sandbox[/dim]")


# Review command


class ReviewerAgent(str, Enum):
    """Agents that can review (detect bugs)."""

    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


@app.command()
def review(
    agent: Annotated[
        ReviewerAgent,
        typer.Argument(help="Reviewer agent: claude, codex, or gemini"),
    ],
    repo: Annotated[
        str,
        typer.Argument(help="Repository name (must match challenge)"),
    ],
    challenge_id: Annotated[
        str,
        typer.Option("--challenge", "-c", help="Challenge ID from challenge phase"),
    ],
    timeout: Annotated[
        int,
        typer.Option("--timeout", "-t", help="Agent timeout in seconds"),
    ] = DEFAULT_TIMEOUT_SECONDS,
    challenges_dir: Annotated[
        Path | None,
        typer.Option("--challenges-dir", help="Custom challenges directory (default: challenges/)"),
    ] = None,
) -> None:
    """Review a challenged repository for bugs."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from cheddar.agents import get_agent
    from cheddar.models import ChallengeReport
    from cheddar.prompts import load_prompt
    from cheddar.sandbox import cleanup_sandbox, create_review_sandbox

    # Determine challenges directory (explicit option overrides default)
    effective_challenges_dir = (
        challenges_dir if challenges_dir is not None else state.get_challenges_dir()
    )
    challenge_dir = effective_challenges_dir / challenge_id
    if not challenge_dir.exists():
        err_console.print(
            f"[red]Error: Challenge '{challenge_id}' not found in {effective_challenges_dir}[/red]"
        )
        raise typer.Exit(2)

    # Load and validate challenge config
    config_file = challenge_dir / "config.json"
    if not config_file.exists():
        err_console.print(f"[red]Error: config.json not found for challenge '{challenge_id}'[/red]")
        raise typer.Exit(2)

    challenge_config = ChallengeReport.model_validate_json(config_file.read_text())
    if challenge_config.repo != repo:
        err_console.print(
            f"[red]Error: Repository mismatch - challenge '{challenge_id}' was created for "
            f"'{challenge_config.repo}', not '{repo}'[/red]"
        )
        raise typer.Exit(2)

    # Validate persisted repo exists (from challenge phase)
    persisted_repo = challenge_dir / "repo"
    if not persisted_repo.exists():
        err_console.print(
            f"[red]Error: Persisted repo not found for challenge '{challenge_id}'[/red]"
        )
        err_console.print("Run 'cheddar challenge' first to create a challenged repository")
        raise typer.Exit(2)

    if state.verbose:
        console.print(f"[dim]Using challenge: {challenge_dir}[/dim]")

    # Create reviews directory
    reviews_dir = challenge_dir / "reviews"
    reviews_dir.mkdir(exist_ok=True)

    # Create sandbox
    sandbox_path = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Create review sandbox (strips .git and bugs.json to prevent leakage)
            task = progress.add_task("Creating review sandbox...", total=None)
            sandbox_path = create_review_sandbox(challenge_dir)
            progress.update(task, description="Sandbox created")

            if state.verbose:
                console.print(f"[dim]Sandbox: {sandbox_path}[/dim]")

            # Load prompt
            progress.update(task, description="Loading prompt...")
            prompt = load_prompt(
                "review-bugs",
                variables={"repo_path": str(sandbox_path)},
            )

            # Run agent
            progress.update(task, description=f"Running {agent.value} agent...")
            agent_instance = get_agent(agent.value)

            if not agent_instance.can_review:
                err_console.print(f"[red]Error: Agent '{agent.value}' cannot review[/red]")
                raise typer.Exit(2)

            report = agent_instance.review(
                sandbox_path=sandbox_path,
                prompt=prompt,
                timeout_seconds=timeout,
            )

            raw_snapshot_dir = _persist_raw_review_findings(
                sandbox_path=sandbox_path,
                challenge_dir=challenge_dir,
                reviewer=agent.value,
            )
            report.raw_findings_snapshot_dir = raw_snapshot_dir

        # Save review report
        report_file = reviews_dir / f"{agent.value}.json"
        report_file.write_text(report.model_dump_json(indent=2))

        # Save raw output for debugging
        raw_file = challenge_dir / f"raw_review_{agent.value}_output.txt"
        raw_file.write_text(report.raw_output)

        # Save stderr for debugging (if any)
        if report.raw_stderr:
            stderr_file = challenge_dir / f"raw_review_{agent.value}_stderr.txt"
            stderr_file.write_text(report.raw_stderr)

        # Save prompt for traceability
        prompt_file = reviews_dir / f"prompt_{agent.value}.md"
        prompt_file.write_text(prompt)

        # Display results
        console.print("[green]Review complete![/green]")
        console.print(f"  Challenge ID: {challenge_id}")
        console.print(f"  Agent: {agent.value}")
        console.print(f"  Status: {report.status}")
        console.print(f"  Duration: {report.duration_seconds:.1f}s")
        console.print(f"  Output: {report_file}")

    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        if state.verbose and hasattr(e, "raw_output"):
            raw_out = getattr(e, "raw_output", "")
            err_console.print(f"[dim]Raw output: {raw_out[:500]}...[/dim]")
        raise typer.Exit(1) from e

    finally:
        # Cleanup sandbox
        if sandbox_path:
            cleanup_sandbox(sandbox_path)
            if state.verbose:
                console.print("[dim]Cleaned up sandbox[/dim]")


# Match command


# Sentinel to detect when no --model was passed (env var read after .env loads)
_MODEL_FROM_ENV = "__FROM_ENV__"


@app.command()
def match(
    challenge_id: Annotated[
        str,
        typer.Option("--challenge", "-c", help="Challenge ID to score"),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model for matching (default: $CHEDDAR_MATCH_MODEL or azure/gpt-5.2)",
        ),
    ] = _MODEL_FROM_ENV,
    challenges_dir: Annotated[
        Path | None,
        typer.Option("--challenges-dir", help="Custom challenges directory (default: challenges/)"),
    ] = None,
    reviewer: Annotated[
        str | None,
        typer.Option("--reviewer", "-r", help="Score only this reviewer (default: all)"),
    ] = None,
    repeat: Annotated[
        int,
        typer.Option(
            "--repeat",
            help="Number of matcher runs per reviewer (default: 1; use with --aggregate median)",
            min=1,
        ),
    ] = 1,
    aggregate: Annotated[
        Literal["single", "median"],
        typer.Option(
            "--aggregate",
            help="How to select final score when --repeat > 1",
            case_sensitive=False,
        ),
    ] = "single",
    scores_root: Annotated[
        Path | None,
        typer.Option(
            "--scores-root",
            help="Custom output root for score files (default: challenge scores/)",
        ),
    ] = None,
) -> None:
    """Score reviewers by matching bugs to report text using LLM."""
    if model == _MODEL_FROM_ENV:
        model = os.environ.get("CHEDDAR_MATCH_MODEL", "azure/gpt-5.2")

    import json

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    from cheddar.findings import load_raw_finding_payloads
    from cheddar.matcher import (
        match_bugs_to_raw_report,
        score_from_matches,
        select_median_score_index,
    )
    from cheddar.models import BugManifest, ReviewReport, Score

    # Determine challenges directory (explicit option overrides default)
    effective_challenges_dir = (
        challenges_dir if challenges_dir is not None else state.get_challenges_dir()
    )
    challenge_dir = effective_challenges_dir / challenge_id
    if not challenge_dir.exists():
        err_console.print(
            f"[red]Error: Challenge '{challenge_id}' not found in {effective_challenges_dir}[/red]"
        )
        raise typer.Exit(2)

    # Validate bugs.json exists
    bugs_file = challenge_dir / "bugs.json"
    if not bugs_file.exists():
        err_console.print(f"[red]Error: bugs.json not found for challenge '{challenge_id}'[/red]")
        err_console.print("Run 'cheddar challenge' first to create bugs")
        raise typer.Exit(2)

    # Load manifest and compute hash for staleness detection
    import hashlib

    bugs_content = bugs_file.read_text()
    manifest_hash = hashlib.sha256(bugs_content.encode()).hexdigest()
    try:
        manifest = BugManifest.model_validate_json(bugs_content)
    except Exception as e:
        err_console.print(f"[red]Error: Failed to parse bugs.json: {e}[/red]")
        raise typer.Exit(1) from e

    # Load reviews (optionally filtered by --reviewer)
    reviews_dir = challenge_dir / "reviews"
    reviews: dict[str, ReviewReport] = {}

    if reviews_dir.exists():
        if reviewer:
            # Load only the specified reviewer
            review_file = reviews_dir / f"{reviewer}.json"
            if review_file.exists():
                try:
                    reviews[reviewer] = ReviewReport.model_validate_json(review_file.read_text())
                except Exception as e:
                    err_console.print(f"[yellow]Warning: Skipping {reviewer}.json: {e}[/yellow]")
            else:
                err_console.print(
                    f"[red]Error: Review '{reviewer}.json' not found in {reviews_dir}[/red]"
                )
                raise typer.Exit(2)
        else:
            for review_file in reviews_dir.glob("*.json"):
                agent_name = review_file.stem
                try:
                    reviews[agent_name] = ReviewReport.model_validate_json(review_file.read_text())
                except Exception as e:
                    err_console.print(f"[yellow]Warning: Skipping {agent_name}.json: {e}[/yellow]")
                    continue

    if not reviews:
        err_console.print(f"[yellow]Warning: No reviews found in {reviews_dir}[/yellow]")
        console.print("[yellow]No reviewers to score[/yellow]")
        raise typer.Exit(0)

    if repeat > 1 and aggregate == "single":
        err_console.print(
            "[red]Error: --repeat > 1 requires --aggregate median to avoid wasting runs[/red]"
        )
        raise typer.Exit(2)

    if aggregate == "median" and repeat < 3:
        err_console.print("[red]Error: --aggregate median requires --repeat >= 3[/red]")
        raise typer.Exit(2)

    output_scores_dir = (
        scores_root / challenge_id / "scores"
        if scores_root is not None
        else challenge_dir / "scores"
    )

    # Score each reviewer
    scores: list[Score] = []
    failed_reviewers: list[tuple[str, Exception]] = []
    skipped_reviewers: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=os.environ.get("CHEDDAR_PLAIN_PROGRESS") != "1",
        disable=os.environ.get("CHEDDAR_PLAIN_PROGRESS") == "1",
        refresh_per_second=2,
    ) as progress:
        run_note = "" if repeat == 1 else f" (repeat={repeat}, aggregate={aggregate})"
        console.print(f"Matching bugs to reports using {model}{run_note}...")

        for reviewer_name, report in reviews.items():
            raw_findings = load_raw_finding_payloads(report=report, challenge_dir=challenge_dir)
            # Skip non-complete reviews (failed, timeout, error)
            has_raw_payloads = bool(raw_findings)
            if report.status != "complete" and not has_raw_payloads:
                skipped_reviewers.append(reviewer_name)
                reason = report.failure_reason or report.status
                err_console.print(
                    f"[yellow]Skipping {reviewer_name}: review status '{report.status}' ({reason})[/yellow]"
                )
                continue
            if report.status != "complete" and has_raw_payloads:
                reason = report.failure_reason or report.status
                err_console.print(
                    f"[yellow]Proceeding with {reviewer_name}: review status '{report.status}' "
                    f"({reason}) but raw findings are present[/yellow]"
                )
            try:
                task = progress.add_task(
                    f"  Scoring {reviewer_name}: global match x {repeat} run(s)",
                    total=repeat,
                )

                run_scores: list[Score] = []
                for _ in range(repeat):
                    matches = match_bugs_to_raw_report(
                        manifest=manifest,
                        raw_findings=raw_findings,
                        model=model,
                    )

                    progress.advance(task, advance=1)

                    run_scores.append(
                        score_from_matches(
                            manifest,
                            report,
                            matches,
                            model=model,
                            manifest_hash=manifest_hash,
                        )
                    )

                # Select final score
                selected_run_index = repeat - 1
                if aggregate == "median" and repeat > 1:
                    selected_run_index = select_median_score_index(run_scores)
                    score = run_scores[selected_run_index]
                else:
                    score = run_scores[-1]
                scores.append(score)

                # Verbose output: show individual match decisions from selected score
                if state.verbose:
                    console.print(f"\n[dim]Match details for {reviewer_name}:[/dim]")
                    for bug_match in score.matches:
                        bug = manifest.bugs[bug_match.bug_index]
                        if bug_match.found:
                            console.print(
                                f"  [green]✓[/green] Bug {bug_match.bug_index} ({bug.file}:{bug.line}) "
                                f"found ({bug_match.confidence})"
                            )
                            console.print(f"      {bug_match.reasoning}")
                        else:
                            console.print(
                                f"  [red]✗[/red] Bug {bug_match.bug_index} ({bug.file}:{bug.line}) "
                                f"not found ({bug_match.confidence})"
                            )
                            console.print(f"      {bug_match.reasoning}")

                # Warn on match errors (always visible, not just verbose)
                if score.match_errors > 0:
                    evaluated = score.total_bugs - score.match_errors
                    err_console.print(
                        f"[yellow]  Warning: {score.match_errors} of {score.total_bugs} "
                        f"bug matches failed (LLM errors) for {reviewer_name} "
                        f"-- detection rate based on {evaluated} evaluated bugs[/yellow]"
                    )

                # Save individual score file: scores/<reviewer>.json
                output_scores_dir.mkdir(parents=True, exist_ok=True)
                score_file = output_scores_dir / f"{reviewer_name}.json"
                score_file.write_text(json.dumps(score.model_dump(), indent=2))

                # Save all repeat runs for auditability when repeat > 1.
                runs_file = output_scores_dir / f"{reviewer_name}.runs.json"
                if repeat > 1:
                    runs_payload: dict[str, object] = {
                        "reviewer": reviewer_name,
                        "model": model,
                        "repeat": repeat,
                        "aggregate": aggregate,
                        "selected_run_index": selected_run_index,
                        "runs": [run.model_dump() for run in run_scores],
                    }
                    runs_file.write_text(json.dumps(runs_payload, indent=2))
                elif runs_file.exists():
                    runs_file.unlink()

            except Exception as e:
                # Log error but continue with other reviewers
                failed_reviewers.append((reviewer_name, e))
                err_console.print(f"[red]Error scoring {reviewer_name}: {e}[/red]")
                if state.verbose:
                    import traceback

                    err_console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # Check if no reviewers could be scored
    if not scores:
        if skipped_reviewers and not failed_reviewers:
            err_console.print(
                f"[yellow]All {len(skipped_reviewers)} review(s) were skipped (failed status)[/yellow]"
            )
            raise typer.Exit(0)
        elif failed_reviewers:
            err_console.print(f"[red]All {len(failed_reviewers)} reviewer(s) failed to score[/red]")
            raise typer.Exit(1)

    # Display results
    scores_dir = output_scores_dir
    console.print()
    if failed_reviewers or skipped_reviewers:
        issues: list[str] = []
        if failed_reviewers:
            issues.append(f"{len(failed_reviewers)} error(s)")
        if skipped_reviewers:
            issues.append(f"{len(skipped_reviewers)} skipped")
        console.print(f"[yellow]Match complete with {', '.join(issues)}[/yellow]")
    else:
        console.print("[green]Match complete![/green]")
    console.print(f"  Challenge ID: {challenge_id}")
    console.print(f"  Model: {model}")
    console.print(f"  Reviewers scored: {len(scores)}")
    if failed_reviewers:
        console.print(f"  Reviewers failed: {len(failed_reviewers)}")
    console.print()
    console.print("  Results:")
    for score in scores:
        errors_note = f", {score.match_errors} match errors" if score.match_errors > 0 else ""
        console.print(
            f"    {score.reviewer}:  {score.bugs_found}/{score.total_bugs} bugs found "
            f"({score.detection_rate * 100:.1f}% detection{errors_note})"
        )
    console.print()
    console.print(f"  Output: {scores_dir}")


if __name__ == "__main__":
    app()
