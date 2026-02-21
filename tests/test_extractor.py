"""Tests for bug manifest extraction validation."""

from cheddar.extractor import (
    _normalize,
    _parse_diff_by_file,
    count_diff_hunks,
    filter_phantom_bugs,
    format_line_map,
    parse_diff_changes,
    validate_bugs_against_diff,
)
from cheddar.models import Bug

SAMPLE_DIFF = """\
diff --git a/src/index.js b/src/index.js
index abc1234..def5678 100644
--- a/src/index.js
+++ b/src/index.js
@@ -10,7 +10,7 @@ function process(input) {
   const result = [];
   for (let i = 0; i < input.length; i++) {
-    if (input[i] >= 0) {
+    if (input[i] > 0) {
       result.push(input[i]);
     }
   }
@@ -25,7 +25,7 @@ function format(value) {
-  return value.toString();
+  return String(value);
"""


def _make_bug(file: str, line: int, original: str, injected: str) -> Bug:
    return Bug(
        file=file,
        line=line,
        type="logic error",
        description="test bug",
        original=original,
        injected=injected,
    )


def test_parse_diff_by_file_extracts_removed_and_added():
    content = _parse_diff_by_file(SAMPLE_DIFF)
    assert "src/index.js" in content
    removed, added = content["src/index.js"]
    assert "if (input[i] >= 0) {" in removed
    assert "if (input[i] > 0) {" in added


def test_parse_diff_by_file_includes_context_lines():
    content = _parse_diff_by_file(SAMPLE_DIFF)
    removed, added = content["src/index.js"]
    # Context lines should appear in both
    assert "result.push(input[i]);" in removed
    assert "result.push(input[i]);" in added


def test_validate_keeps_correct_bugs():
    bugs = [
        _make_bug("src/index.js", 12, "if (input[i] >= 0) {", "if (input[i] > 0) {"),
        _make_bug("src/index.js", 26, "return value.toString();", "return String(value);"),
    ]
    valid, rejected = validate_bugs_against_diff(bugs, SAMPLE_DIFF)
    assert len(valid) == 2
    assert rejected == []


def test_validate_rejects_phantom_injected():
    bugs = [
        _make_bug(
            "src/index.js",
            12,
            "if (input[i] >= 0) {",
            "if (input[i] !== null) {",  # fabricated
        ),
    ]
    valid, rejected = validate_bugs_against_diff(bugs, SAMPLE_DIFF)
    assert len(valid) == 0
    assert len(rejected) == 1
    assert "'injected' not found" in rejected[0][1]


def test_validate_rejects_phantom_original():
    bugs = [
        _make_bug(
            "src/index.js",
            12,
            "if (input[i] == 0) {",  # fabricated
            "if (input[i] > 0) {",
        ),
    ]
    valid, rejected = validate_bugs_against_diff(bugs, SAMPLE_DIFF)
    assert len(valid) == 0
    assert len(rejected) == 1
    assert "'original' not found" in rejected[0][1]


def test_validate_allows_context_lines_in_original():
    """LLM often includes context lines in original/injected — not an error."""
    bugs = [
        _make_bug(
            "src/index.js",
            12,
            "if (input[i] >= 0) {\n      result.push(input[i]);",
            "if (input[i] > 0) {\n      result.push(input[i]);",
        ),
    ]
    valid, rejected = validate_bugs_against_diff(bugs, SAMPLE_DIFF)
    assert len(valid) == 1
    assert rejected == []


def test_validate_rejects_wrong_file():
    bugs = [_make_bug("nonexistent.js", 1, "foo", "bar")]
    valid, rejected = validate_bugs_against_diff(bugs, SAMPLE_DIFF)
    assert len(valid) == 0
    assert len(rejected) == 1
    assert "not found in diff" in rejected[0][1]


def test_normalize_collapses_whitespace():
    assert _normalize("  foo  \n\t bar  ") == "foo bar"
    assert _normalize("a") == "a"


def test_parse_diff_handles_ci_prefixes():
    """Git diffs from --src-prefix=c/ --dst-prefix=i/ should work."""
    diff = """\
diff --git c/lib/foo.rb i/lib/foo.rb
index 1234567..abcdef0 100644
--- c/lib/foo.rb
+++ i/lib/foo.rb
@@ -1,3 +1,3 @@
-old_code
+new_code
"""
    content = _parse_diff_by_file(diff)
    assert "lib/foo.rb" in content
    removed, added = content["lib/foo.rb"]
    assert "old_code" in removed
    assert "new_code" in added


# --- filter_phantom_bugs tests ---


def test_filter_phantom_bugs_keeps_valid():
    bugs = [
        _make_bug("a.js", 1, "x = 1", "x = 2"),
        _make_bug("b.js", 5, "return a", "return b"),
    ]
    kept, removed = filter_phantom_bugs(bugs)
    assert len(kept) == 2
    assert removed == []


def test_filter_phantom_bugs_removes_identical():
    bugs = [
        _make_bug("a.js", 1, "x = 1", "x = 1"),
        _make_bug("b.js", 5, "  y = 2  ", "y = 2"),  # whitespace-only difference stripped
    ]
    kept, removed = filter_phantom_bugs(bugs)
    assert len(kept) == 0
    assert len(removed) == 2


def test_filter_phantom_bugs_keeps_empty_injected_deletions():
    """Deletion bugs have non-empty original and empty injected — not phantom."""
    bug = Bug(
        file="a.js",
        line=1,
        type="deletion",
        description="removed line",
        original="const x = 1;",
        injected="",
    )
    kept, removed = filter_phantom_bugs([bug])
    assert len(kept) == 1
    assert removed == []


def test_filter_phantom_bugs_mixed():
    bugs = [
        _make_bug("a.js", 1, "x = 1", "x = 2"),  # valid
        _make_bug("b.js", 5, "y = 2", "y = 2"),  # phantom
        _make_bug("c.js", 10, "z = 3", "z = 4"),  # valid
    ]
    kept, removed = filter_phantom_bugs(bugs)
    assert len(kept) == 2
    assert len(removed) == 1
    assert removed[0].file == "b.js"


# --- parse_diff_changes tests ---


MULTI_HUNK_DIFF = """\
diff --git a/src/index.js b/src/index.js
index abc1234..def5678 100644
--- a/src/index.js
+++ b/src/index.js
@@ -10,7 +10,7 @@ function process(input) {
   const result = [];
   for (let i = 0; i < input.length; i++) {
-    if (input[i] >= 0) {
+    if (input[i] > 0) {
       result.push(input[i]);
     }
   }
@@ -25,7 +25,7 @@ function format(value) {
-  return value.toString();
+  return String(value);
"""

DELETION_DIFF = """\
diff --git a/color.go b/color.go
index 1234567..abcdef0 100644
--- a/color.go
+++ b/color.go
@@ -498,8 +498,5 @@ func getCachedColor(p Attribute) *Color {
     x := 1
     y := 2
-    colorsCacheMu.Lock()
-    defer colorsCacheMu.Unlock()
-
     c, ok := colorsCache[p]
     if !ok {
"""

ADDITION_DIFF = """\
diff --git a/main.py b/main.py
index 1234567..abcdef0 100644
--- a/main.py
+++ b/main.py
@@ -5,3 +5,5 @@ def main():
     x = 1
     y = 2
+    z = 3
+    w = 4
     return x + y
"""


def test_parse_diff_changes_modify():
    changes = parse_diff_changes(MULTI_HUNK_DIFF)
    assert len(changes) == 2

    c1 = changes[0]
    assert c1.file == "src/index.js"
    assert c1.change_kind == "modify"
    assert c1.new_line == 12
    assert len(c1.removed) == 1
    assert c1.removed[0][0] == 12  # old line 12
    assert "input[i] >= 0" in c1.removed[0][1]
    assert len(c1.added) == 1
    assert c1.added[0][0] == 12  # new line 12
    assert "input[i] > 0" in c1.added[0][1]

    c2 = changes[1]
    assert c2.file == "src/index.js"
    assert c2.change_kind == "modify"
    assert c2.new_line == 25
    assert c2.removed[0][0] == 25
    assert c2.added[0][0] == 25


def test_parse_diff_changes_deletion():
    changes = parse_diff_changes(DELETION_DIFF)
    assert len(changes) == 1

    c = changes[0]
    assert c.file == "color.go"
    assert c.change_kind == "delete"
    assert len(c.removed) == 3  # two code lines + one blank line
    assert c.removed[0][0] == 500
    assert "colorsCacheMu.Lock()" in c.removed[0][1]
    assert c.removed[1][0] == 501
    assert "defer colorsCacheMu.Unlock()" in c.removed[1][1]
    assert c.removed[2][0] == 502  # blank line
    assert len(c.added) == 0
    # new_line should be the new-file position after the deletion
    assert c.new_line == 500


def test_parse_diff_changes_addition():
    changes = parse_diff_changes(ADDITION_DIFF)
    assert len(changes) == 1

    c = changes[0]
    assert c.file == "main.py"
    assert c.change_kind == "add"
    assert len(c.removed) == 0
    assert len(c.added) == 2
    assert c.added[0][0] == 7  # new line 7
    assert "z = 3" in c.added[0][1]
    assert c.added[1][0] == 8  # new line 8
    assert c.new_line == 7


def test_parse_diff_changes_ci_prefixes():
    """Handles c/ and i/ git prefixes."""
    diff = """\
diff --git c/lib/foo.rb i/lib/foo.rb
index 1234567..abcdef0 100644
--- c/lib/foo.rb
+++ i/lib/foo.rb
@@ -1,3 +1,3 @@
 context
-old_code
+new_code
 context
"""
    changes = parse_diff_changes(diff)
    assert len(changes) == 1
    assert changes[0].file == "lib/foo.rb"
    assert changes[0].new_line == 2
    assert changes[0].removed[0] == (2, "old_code")
    assert changes[0].added[0] == (2, "new_code")


def test_parse_diff_changes_multiple_files():
    diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-old_a
+new_a
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,3 +5,3 @@
-old_b
+new_b
"""
    changes = parse_diff_changes(diff)
    assert len(changes) == 2
    assert changes[0].file == "a.py"
    assert changes[0].new_line == 1
    assert changes[1].file == "b.py"
    assert changes[1].new_line == 5


# --- format_line_map tests ---


def test_format_line_map_modify():
    changes = parse_diff_changes(MULTI_HUNK_DIFF)
    output = format_line_map(changes)
    assert "CHANGE 1: src/index.js:12 [modify]" in output
    assert "- (old 12)" in output
    assert "+ (new 12)" in output
    assert "CHANGE 2: src/index.js:25 [modify]" in output


def test_format_line_map_deletion():
    changes = parse_diff_changes(DELETION_DIFF)
    output = format_line_map(changes)
    assert "CHANGE 1: color.go:500 [delete]" in output
    assert "- (old 500)" in output
    assert "+ (deleted)" in output


def test_format_line_map_empty():
    assert format_line_map([]) == "(no changes detected)"


# --- count_diff_hunks tests ---


def test_count_diff_hunks():
    assert count_diff_hunks(MULTI_HUNK_DIFF) == 2
    assert count_diff_hunks(DELETION_DIFF) == 1
    assert count_diff_hunks("") == 0


# --- validate_bugs_against_diff with change_kind tests ---


def test_validate_allows_deletion_bugs():
    """Deletion bugs with change_kind='delete' should pass validation."""
    diff = DELETION_DIFF
    bug = Bug(
        file="color.go",
        line=500,
        type="deletion",
        description="removed mutex",
        original="colorsCacheMu.Lock()",
        injected="",
        change_kind="delete",
    )
    valid, rejected = validate_bugs_against_diff([bug], diff)
    assert len(valid) == 1
    assert rejected == []


def test_validate_still_rejects_bad_original_for_deletion():
    """Deletion bugs still need valid original content."""
    diff = DELETION_DIFF
    bug = Bug(
        file="color.go",
        line=500,
        type="deletion",
        description="removed mutex",
        original="nonexistent_code",
        injected="",
        change_kind="delete",
    )
    valid, rejected = validate_bugs_against_diff([bug], diff)
    assert len(valid) == 0
    assert len(rejected) == 1
