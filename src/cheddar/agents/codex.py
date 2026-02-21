"""OpenAI Codex CLI agent implementation."""

import json
from typing import Literal

from cheddar.agents.base import BaseAgent
from cheddar.agents.config import MODELS, REASONING_EFFORT


def _extract_codex_text(raw: str) -> str:
    """Extract full agent transcript from Codex JSONL output.

    Codex --json emits JSONL with item.completed events. We capture:
    - agent_message / reasoning: text field (agent's words)
    - command_execution: the command run and its output
    """
    texts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        item_type = item.get("type", "")

        if item_type == "command_execution":
            cmd = item.get("command", "")
            output = item.get("aggregated_output", "")
            texts.append(f"$ {cmd}\n{output}")
        elif item.get("text"):
            texts.append(item["text"])

    return "\n\n".join(texts)


class CodexAgent(BaseAgent):
    """Agent implementation for OpenAI Codex CLI."""

    CAN_CHALLENGE = True
    CAN_REVIEW = True

    @property
    def name(self) -> Literal["codex"]:
        """Agent identifier."""
        return "codex"

    def _parse_output(self, raw_stdout: str) -> str:
        """Extract text from Codex JSONL output."""
        return _extract_codex_text(raw_stdout)

    def _build_challenge_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for challenge.

        Uses Codex CLI with full autonomous operation:
        --dangerously-bypass-approvals-and-sandbox: no sandbox/approval restrictions
        --skip-git-repo-check: allow running in temp sandbox without git
        --json: structured JSONL output for full visibility
        """
        return [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "-m",
            MODELS["codex"],
            "-c",
            f"model_reasoning_effort='{REASONING_EFFORT['codex']}'",
            prompt,
        ]

    def _build_review_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for review.

        Uses Codex CLI with full autonomous operation.
        """
        return [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "-m",
            MODELS["codex"],
            "-c",
            f"model_reasoning_effort='{REASONING_EFFORT['codex']}'",
            prompt,
        ]
