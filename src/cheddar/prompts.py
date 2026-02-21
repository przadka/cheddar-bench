"""Prompt template loading and variable substitution."""

import os
from pathlib import Path


class PromptNotFoundError(Exception):
    """Raised when a prompt template cannot be found."""

    def __init__(self, prompt_name: str, prompts_dir: Path) -> None:
        """Initialize PromptNotFoundError.

        Args:
            prompt_name: Name of the missing prompt.
            prompts_dir: Directory that was searched.
        """
        self.prompt_name = prompt_name
        self.prompts_dir = prompts_dir
        super().__init__(
            f"Prompt '{prompt_name}' not found in {prompts_dir}. "
            f"Expected file: {prompts_dir / f'{prompt_name}.md'}"
        )


def get_prompts_dir() -> Path:
    """Get the prompts directory path.

    Returns:
        Path to the prompts directory, from CHEDDAR_PROMPTS_DIR env var
        or prompts/ relative to the package root.
    """
    env_dir = os.environ.get("CHEDDAR_PROMPTS_DIR")
    if env_dir:
        return Path(env_dir)
    # Resolve relative to package: src/cheddar/prompts.py -> src/cheddar -> src -> repo root
    return Path(__file__).parent.parent.parent / "prompts"


def load_prompt(
    name: str,
    prompts_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> str:
    """Load a prompt template and apply variable substitution.

    Args:
        name: Name of the prompt (without .md extension).
        prompts_dir: Directory containing prompt templates.
                    Defaults to get_prompts_dir().
        variables: Dictionary of variables to substitute.
                  Keys should match {{variable_name}} in template.

    Returns:
        The prompt text with variables substituted.

    Raises:
        PromptNotFoundError: If the prompt file doesn't exist.
    """
    if prompts_dir is None:
        prompts_dir = get_prompts_dir()

    prompt_file = prompts_dir / f"{name}.md"
    if not prompt_file.exists():
        raise PromptNotFoundError(name, prompts_dir)

    content = prompt_file.read_text()

    # Apply variable substitution
    if variables:
        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            content = content.replace(placeholder, value)

    return content


def load_prompt_pair(
    name: str,
    prompts_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Load a system/user prompt pair and apply variable substitution.

    Loads {name}.system.md and {name}.user.md, applies variable substitution
    to both, and returns them as a message list for LLM APIs.

    Args:
        name: Base name of the prompt pair (without .system.md/.user.md).
        prompts_dir: Directory containing prompt templates.
        variables: Variables to substitute in both templates.

    Returns:
        List of two message dicts: [{'role': 'system', ...}, {'role': 'user', ...}]

    Raises:
        PromptNotFoundError: If either prompt file doesn't exist.
    """
    system_content = load_prompt(f"{name}.system", prompts_dir=prompts_dir, variables=variables)
    user_content = load_prompt(f"{name}.user", prompts_dir=prompts_dir, variables=variables)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def get_available_prompts(prompts_dir: Path | None = None) -> list[str]:
    """List available prompt templates.

    Args:
        prompts_dir: Directory containing prompt templates.
                    Defaults to get_prompts_dir().

    Returns:
        List of prompt names (without .md extension).
    """
    if prompts_dir is None:
        prompts_dir = get_prompts_dir()

    if not prompts_dir.exists():
        return []

    return sorted(p.stem for p in prompts_dir.glob("*.md"))
