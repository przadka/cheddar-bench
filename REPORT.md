# Cheddar Bench Report

Cheddar Bench evaluates CLI coding agents on blind bug detection using self-play:
agents inject bugs into repos, other agents review those repos without seeing ground truth.

## Headline Results

Scale:
- 50 repositories
- 150 challenges (3 challengers x 50 repos)
- 450 reviews (3 reviewers x 150 challenges)
- 2,603 injected bugs total

Scoring policy for this report:
- matcher consumes raw reviewer `bugs/*.json` payloads
- matcher repeated 5 times per review
- median run selected (`repeat=5`, `aggregate=median`)

### Unweighted (mean of 150 challenge scores)

| Reviewer | Detection Rate |
|----------|----------------|
| Claude | **61.65%** |
| Codex | **43.17%** |
| Gemini | **34.64%** |

### Weighted (global bug count)

| Reviewer | Bugs Found |
|----------|------------|
| Claude | **1,511 / 2,603 (58.05%)** |
| Codex | **985 / 2,603 (37.84%)** |
| Gemini | **724 / 2,603 (27.81%)** |

## Key Findings

- Claude leads by a wide margin on both unweighted and weighted metrics.
- Codex is second and notably behind Claude on broad recall.
- Gemini trails both on overall bug detection.
- Extreme per-challenge variance exists, so single-challenge anecdotes are noisy.

## Methodology

1. Challenge agent injects bugs into a clean repo; the target bug count scales with repository size and is capped (currently 24 per challenge).
2. Ground truth is the challenger-produced `bugs.json` manifest (which includes the injected bug set and metadata).
3. Reviewer agent audits the mutated repo blind and emits findings as `bugs/*.json` payloads (plus optional free-form text output).
4. LLM matcher assigns reviewer raw finding payloads to injected bugs and computes score.

Notes:
- This benchmark compares CLI tools as full systems (prompting, tool/runtime behavior, and CLI flag configuration), not just base models.
- Runs execute with autonomous permissions (`--yolo` / dangerous-skip equivalents) on sandboxed challenge repositories.
- Scoring consumes raw reviewer `bugs/*.json` payloads as emitted by agents.
- Repeat+median is used to reduce matcher stochasticity.

## Dataset Publication

Reference dataset release (`challenges/` snapshot) is published at:

- `s3://cheddar-bench-data-public/datasets/cheddar-bench-challenges-2026-02-21T122452Z-cb4b7ba38c3c-r2/`
- `https://cheddar-bench-data-public.s3.eu-central-1.amazonaws.com/datasets/cheddar-bench-challenges-2026-02-21T122452Z-cb4b7ba38c3c-r2/`

Artifacts and immutable version IDs:

| Artifact | Version ID |
|----------|------------|
| `cheddar-bench-challenges-2026-02-21T122452Z-cb4b7ba38c3c-r2.tar.gz` | `tCGltP_anHikU76SYwB.3RJh8RKxh1.V` |
| `cheddar-bench-challenges-2026-02-21T122452Z-cb4b7ba38c3c-r2.tar.gz.sha256` | `Mi9SGPwLSj1igoX4avd1unK08oqwCxg2` |
| `cheddar-bench-challenges-2026-02-21T122452Z-cb4b7ba38c3c-r2.manifest.json` | `PuzCHnjSNbV9ONqZRfeV1nF70R8cmSbO` |
| `cheddar-bench-challenges-2026-02-21T122452Z-cb4b7ba38c3c-r2.manifest.json.sha256` | `9vYKiNM8lzBpsXUnOu2jzaBnol8ET5KP` |
