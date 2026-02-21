# cheddar-bench Development Guidelines

Benchmark for evaluating CLI coding agents on bug detection. For project overview, see README.md.

Note: `CLAUDE.md` can be a local, uncommitted symlink to this file.

## Commands

```bash
uv sync                                             # Install deps
uv run cheddar list                                 # List agents, repos
uv run cheddar challenge <agent> <repo>             # Inject bugs
uv run cheddar review <agent> <repo> --challenge <id>  # Review for bugs
uv run cheddar match --challenge <id>               # Score results
uv run pytest                                       # Tests
uv run ruff format src/ tests/                      # Format
uv run ruff check src/ tests/                       # Lint
uv run mypy src/                                    # Type check
```

## Architecture

```
src/cheddar/
├── cli.py           # Typer commands (list, challenge, review, match)
├── models.py        # Pydantic: Bug, Issue, BugManifest, ReviewReport, Score
├── agents/          # BaseAgent + implementations (claude, codex, gemini)
│   ├── base.py      # Abstract base with challenge/review flow
│   └── *.py         # Per-agent CLI wrappers
├── sandbox.py       # Temp sandbox creation/cleanup
├── prompts.py       # Template loading with {variable} substitution
├── matcher.py       # Bug-to-issue matching logic
├── llm.py           # LLM-as-judge calls via litellm
└── errors.py        # AgentError, AgentTimeoutError, AgentParseError
prompts/             # Markdown prompt templates
repos/               # Target repositories to benchmark
challenges/          # Challenge data: {challenger}-{repo}/
```

**Flow**: `challenge` injects bugs via agent -> saves `bugs.json` + mutated repo -> `review` runs agents on mutated repo -> saves `reviews/*.json` -> `match` compares issues to ground truth -> `scores/<reviewer>.json`

## Key Files

When modifying agents:
- `src/cheddar/agents/base.py` - BaseAgent interface, `_run_cli()`, JSON extraction
- `src/cheddar/agents/claude.py` - Clean agent example
- `src/cheddar/models.py` - Data structures (Bug, Issue, BugManifest, ReviewReport)

When modifying CLI:
- `src/cheddar/cli.py` - All commands, `AppState` for global config
- `prompts/*.md` - Prompt templates used by commands

When debugging agent output:
- `challenges/{challenger}-{repo}/raw_challenge_output.txt` - Raw challenger stdout
- `challenges/{challenger}-{repo}/raw_review_*_output.txt` - Raw reviewer stdout

## Environment Setup

Configure provider credentials via environment variables before running benchmarks.

- Azure-backed models: use either key-based auth or token-based auth consistently.
- Gemini/Vertex models: ensure ADC and required project/location variables are set.
- Keep secrets out of git; use local shell env or a gitignored dotenv file.

## Style

- Python 3.12+, type hints required (`dict | None` not `Optional[dict]`)
- Ruff for formatting (100 char lines)
- Docstrings: explain purpose, not mechanics
- Function-based tests preferred over test classes
- Mock all agent CLI calls in tests

## Testing

```bash
uv run pytest                          # All tests
uv run pytest tests/test_agents.py -v  # Specific module
uv run pytest -k "challenge"           # Pattern match
```

- Mock subprocess calls, never hit real agents in tests
- Test both success and error paths
- Fixtures in `tests/conftest.py`

## Git

- Feature branches, rebase onto main
- Run `ruff format && ruff check && mypy src/ && pytest` before committing
- Conventional commits: `feat(cli): add X`, `fix(agents): handle Y`

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
