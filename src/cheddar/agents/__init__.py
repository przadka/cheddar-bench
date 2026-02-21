"""Agent implementations for CLI coding agents."""

from cheddar.agents.base import BaseAgent
from cheddar.agents.claude import ClaudeAgent
from cheddar.agents.codex import CodexAgent
from cheddar.agents.gemini import GeminiAgent

# Agent registry
AGENTS: dict[str, type[BaseAgent]] = {
    "claude": ClaudeAgent,
    "codex": CodexAgent,
    "gemini": GeminiAgent,
}


def get_agent(name: str) -> BaseAgent:
    """Get agent instance by name.

    Args:
        name: Agent identifier (e.g., 'claude', 'gemini').

    Returns:
        Instantiated agent.

    Raises:
        ValueError: If agent name is unknown.
    """
    if name not in AGENTS:
        available = list(AGENTS.keys())
        raise ValueError(f"Unknown agent: {name}. Available: {available}")
    return AGENTS[name]()


def get_challengers() -> list[str]:
    """Get names of agents that can challenge.

    Returns:
        List of agent names that support bug injection.
    """
    return [name for name, cls in AGENTS.items() if cls.CAN_CHALLENGE]


def get_reviewers() -> list[str]:
    """Get names of agents that can review.

    Returns:
        List of agent names that support code review.
    """
    return [name for name, cls in AGENTS.items() if cls.CAN_REVIEW]


def register_agent(name: str, agent_class: type[BaseAgent]) -> None:
    """Register a new agent implementation.

    Args:
        name: Agent identifier.
        agent_class: Agent class to register.
    """
    AGENTS[name] = agent_class


__all__ = [
    "BaseAgent",
    "ClaudeAgent",
    "CodexAgent",
    "GeminiAgent",
    "AGENTS",
    "get_agent",
    "get_challengers",
    "get_reviewers",
    "register_agent",
]
