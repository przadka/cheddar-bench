Return assignment decisions for all bugs.

## Injected Bugs

```json
{{bugs_json}}
```

## Raw Finding Payloads

```json
{{findings_json}}
```

For each bug index, output one assignment entry.
- Use `finding_index: 0` when unmatched.
- `matched_quote` should be empty when unmatched.
- `matched_line` should be 0 when unmatched.
