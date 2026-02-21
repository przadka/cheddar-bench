This is an experiment for coding agent evaluation. We inject bugs into codebases as challenges for other agents to find through code review. This is NOT production code - all bugs are documented and reviewed in read-only mode.

Inject exactly {{bug_count}} realistic bugs into this codebase.

REQUIREMENTS:
1. Bug types: off-by-one errors, edge case failures, logic errors, null/undefined handling, wrong conditionals, missing validation, incorrect return values
2. Bugs should be realistic - the kind a developer might accidentally introduce
3. Each bug must change multiple tokens (no single-character edits like removing one char from a regex)
4. Do NOT break the code completely - it should still mostly work
5. Do NOT add comments hinting at bugs
6. Spread bugs across different files when possible

Actually modify the files using your tools.

## Bug Manifest (REQUIRED)

After injecting all bugs, you MUST write a `bugs.json` file in the repository root. This file is critical — without it, the challenge is considered failed.

The file must contain a JSON object with this exact structure:

```json
{
  "bug_count": <number of bugs injected>,
  "bugs": [
    {
      "file": "path/to/file.py",
      "line": 42,
      "type": "off-by-one",
      "description": "Changed loop bound from < to <= causing array overflow",
      "original": "for i in range(len(items)):",
      "injected": "for i in range(len(items) + 1):",
      "change_kind": "modify"
    }
  ]
}
```

Field definitions:
- **file**: Relative path from repo root (e.g. `src/utils.py`, not `/tmp/.../src/utils.py`)
- **line**: Line number in the modified file where the bug appears
- **type**: Bug category (off-by-one, logic error, edge case, null handling, wrong conditional, missing validation, incorrect return, etc.)
- **description**: What the original code did correctly and how your change breaks it
- **original**: The original correct code you replaced (verbatim)
- **injected**: The buggy code you wrote (verbatim). Use empty string `""` for deletions
- **change_kind**: `"modify"` if you replaced code, `"delete"` if you removed code

Formatting requirement:
- Use valid JSON only (no comments, no trailing commas)

Write this file LAST, after all bugs have been injected.
