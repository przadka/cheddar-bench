Review the codebase in {{repo_path}} for bugs.

Output format (required):
- Create a `bugs/` directory in {{repo_path}} if it does not exist.
- Write one JSON file per finding: `bugs/<slug>.json`.
- Use lowercase kebab-case for `<slug>` (`[a-z0-9-]+`), and keep filenames unique.
- Each JSON file must follow this schema (`line` is optional if unknown):

```json
{
  "file": "relative/path/from/repo/root",
  "line": 123,
  "type": "logic error",
  "why": "Explain precisely why this is wrong.",
  "impact": "Explain concrete impact if unfixed."
}
```

Rules:
- `line` can be omitted if unknown, but include it whenever possible.
- Keep `why` and `impact` specific to this bug (no generic text).
- Do not combine multiple bugs in one file.
- Do not include non-bugs, style issues, or speculative concerns.
- Use valid JSON only (no comments, no trailing commas).
- `file` must be a repo-relative path (no absolute paths).

Ignore: style issues, documentation, type annotations, performance optimization.
