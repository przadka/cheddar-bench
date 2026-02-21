"""Claude Code CLI agent implementation."""

import json
from typing import Any, Literal

from cheddar.agents.base import BaseAgent
from cheddar.agents.config import MODELS


def _extract_text_from_stream_json(raw: str) -> str:
    """Extract plain text from Claude CLI stream-json output.

    Collects all assistant text blocks across turns. Multi-turn reviews produce
    text interleaved with tool calls, so we concatenate all text blocks rather
    than relying on the final 'result' message (which only contains the last turn).
    """
    assistant_texts: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "assistant":
            content: list[dict[str, Any]] = msg.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "text":
                    assistant_texts.append(str(block["text"]))

    return "\n\n".join(assistant_texts)


class ClaudeAgent(BaseAgent):
    """Agent implementation for Claude Code CLI."""

    CAN_CHALLENGE = True
    CAN_REVIEW = True

    @property
    def name(self) -> Literal["claude"]:
        """Agent identifier."""
        return "claude"

    def _parse_output(self, raw_stdout: str) -> str:
        """Parse stream-json JSONL output into plain text."""
        return _extract_text_from_stream_json(raw_stdout)

    def _build_challenge_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for challenge.

        Uses Claude Code CLI with full autonomous operation.
        --output-format stream-json streams JSONL for visibility and partial output on timeout.
        """
        return [
            "claude",
            "--model",
            MODELS["claude"],
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

    def _build_review_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for review.

        Uses Claude Code CLI with full autonomous operation.
        --output-format stream-json streams JSONL for visibility and partial output on timeout.
        """
        return [
            "claude",
            "--model",
            MODELS["claude"],
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
