# Cheddar Bench

Unsupervised benchmark for evaluating CLI coding agents on bug detection.

**TL;DR**: Agents playing treasure hunt! Challenger agents hide bugs and write the ground-truth `bugs.json` manifest, reviewer agents try to find those bugs, and an LLM matcher scores assignments. No human labeling.

1. Agent injects bugs into a repo ([prompt](prompts/inject-bugs.md)); bug count scales with repo size and is capped (currently 24 per challenge)
2. Challenger writes `bugs.json` ground-truth manifest ([system prompt](prompts/extract-manifest.system.md), [user prompt](prompts/extract-manifest.user.md))
3. Different agent reviews the code blind and emits `bugs/*.json` findings ([prompt](prompts/review-bugs.md))
4. LLM judge scores bug-to-finding assignments ([prompt](prompts/match-all-bugs-report.system.md))

**Note**: This benchmark tests CLI tools (Claude Code, Codex CLI, Gemini CLI), not the underlying models directly. Each CLI has its own system prompt, tool implementations, and default settings that affect behavior.

## Results

150 challenges (3 challengers x 50 repos), 450 reviews, 2,603 injected bugs.

Scoring policy for reported results:
- matcher consumes raw `bugs/*.json` finding payloads from reviewer runs
- matcher `--repeat 5 --aggregate median`

| Reviewer | Unweighted Detection Rate | Weighted Bugs Found |
|----------|---------------------------|---------------------|
| Claude | **61.65%** | **1,511 / 2,603 (58.05%)** |
| Codex | **43.17%** | **985 / 2,603 (37.84%)** |
| Gemini | **34.64%** | **724 / 2,603 (27.81%)** |

Why two metrics:
- Unweighted = mean of per-challenge detection rates (each challenge counts equally).
- Weighted = global bug-level recall (`sum(bugs_found) / sum(total_bugs)`).

For a concise write-up, see [`REPORT.md`](REPORT.md).

## Usage

```bash
uv sync
uv run cheddar challenge claude slugify      # inject bugs
uv run cheddar review codex slugify -c <id>  # review blind
uv run cheddar match -c <id>                 # score (single run)
uv run cheddar match -c <id> --repeat 3 --aggregate median  # score (median-of-3)
```

## Dataset

50 open-source utility libraries across JavaScript, TypeScript, Python, Go, C, Ruby, Rust, Java, C#.
Source repositories are vendored under `repos/` (one directory per target project).
Use `uv run cheddar list repos` to list the active repo set recognized by the CLI.

Reference dataset release (full `challenges/` snapshot):

- S3 prefix: `s3://cheddar-bench-data-public/datasets/cheddar-bench-challenges-2026-02-21-c5a7337d55be/`
- Public HTTPS prefix: `https://cheddar-bench-data-public.s3.eu-central-1.amazonaws.com/datasets/cheddar-bench-challenges-2026-02-21-c5a7337d55be/`

Published artifacts:

| File | SHA256 | Version ID |
|------|--------|------------|
| `cheddar-bench-challenges-2026-02-21-c5a7337d55be.tar.gz` | `d479b18884530cf38cf996768a59afa1b3c334f5668feca088a44cb89a372241` | `4EcC8UgEXrSuRAbezp8ScPJzcH7_EN9d` |
| `cheddar-bench-challenges-2026-02-21-c5a7337d55be.tar.gz.sha256` | n/a | `HBbOT1vqHNX1kHfrsjdru2Hv_TJNL_2x` |
| `cheddar-bench-challenges-2026-02-21-c5a7337d55be.manifest.json` | `d5b4167318ce41275794985a4e3a45fb13c5f7c0512bc7ae3cf6a33461031659` | `VAYRp6eO272T5zidIKgS2LmEFvWdqPwC` |
| `cheddar-bench-challenges-2026-02-21-c5a7337d55be.manifest.json.sha256` | n/a | `7qCJIyaBCrdei6DtzWThCUCoEqkhFMjp` |

Retrieve and verify:

```bash
DATASET_ID=cheddar-bench-challenges-2026-02-21-c5a7337d55be
BASE_URL="https://cheddar-bench-data-public.s3.eu-central-1.amazonaws.com/datasets/${DATASET_ID}"

curl -LO "${BASE_URL}/${DATASET_ID}.tar.gz"
curl -LO "${BASE_URL}/${DATASET_ID}.tar.gz.sha256"
curl -LO "${BASE_URL}/${DATASET_ID}.manifest.json"
curl -LO "${BASE_URL}/${DATASET_ID}.manifest.json.sha256"

sha256sum -c "${DATASET_ID}.tar.gz.sha256"
sha256sum -c "${DATASET_ID}.manifest.json.sha256"
```

## Scoring

An LLM judge (`gpt-5.2`) performs one global assignment per review: all injected bugs vs reviewer `bugs/*.json` payloads. The judge is instructed to match only when file/location/mechanism align and to leave uncertain bugs unmatched. Each match includes a supporting quote and line from reviewer findings.

To reduce judge stochasticity, scoring uses repeats with median aggregation (`--repeat 5 --aggregate median`).

Scoring uses raw reviewer finding payloads as produced by agents.

## Agents

We tested official CLI tools with autonomous permission flags enabled on sandboxed challenge repositories.

| Agent | Model | CLI Flags |
|-------|-------|-----------|
| Claude Code | claude-opus-4-6 | `--dangerously-skip-permissions --output-format stream-json --verbose` |
| Codex CLI | gpt-5.3-codex | `exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --json -c model_reasoning_effort='medium'` |
| Gemini CLI | gemini-3-pro-preview | `--yolo` |

Model versions are configured in [`src/cheddar/agents/config.py`](src/cheddar/agents/config.py); CLI command flags are defined in the per-agent wrappers under `src/cheddar/agents/*.py`.

PRs welcome for additional agents.
