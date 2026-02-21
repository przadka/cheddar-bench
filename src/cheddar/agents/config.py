"""Central configuration for agent CLI parameters."""

# Model versions for each agent CLI
MODELS = {
    "claude": "claude-opus-4-6",
    "codex": "gpt-5.3-codex",
    "gemini": "gemini-3-pro-preview",
}

# Reasoning effort for agents that support it
REASONING_EFFORT = {
    "codex": "medium",
}
