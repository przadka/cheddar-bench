"""LLM utilities using litellm for provider-agnostic LLM calls."""

import json
import os
from collections.abc import Callable
from typing import Any, cast

import litellm
from json_repair import repair_json
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel, ValidationError

litellm.suppress_debug_info = True


def completion(**kwargs: Any) -> Any:
    """Thin wrapper to keep a patchable completion symbol for tests."""
    return litellm.completion(**kwargs)


class LLMError(Exception):
    """LLM API or parsing error."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.raw_response = raw_response


_azure_token_provider: Callable[[], str] | None = None


def _get_azure_token_provider() -> Callable[[], str]:
    """Return a cached Azure AD bearer token provider (requires azure-identity)."""
    global _azure_token_provider  # noqa: PLW0603
    if _azure_token_provider is not None:
        return _azure_token_provider

    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as e:
        raise LLMError(
            "azure-identity is required for Azure token auth. Install with: uv sync --extra azure"
        ) from e

    credential = DefaultAzureCredential()
    _azure_token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return _azure_token_provider


def _build_azure_kwargs(model: str) -> dict[str, Any]:
    """Build extra kwargs for litellm.completion() when using Azure models."""
    if not model.startswith("azure/"):
        return {}

    kwargs: dict[str, Any] = {}

    api_base = os.environ.get("AZURE_API_BASE")
    api_version = os.environ.get("AZURE_API_VERSION")
    if api_base:
        kwargs["api_base"] = api_base
    if api_version:
        kwargs["api_version"] = api_version

    if not os.environ.get("AZURE_API_KEY"):
        kwargs["azure_ad_token_provider"] = _get_azure_token_provider()

    return kwargs


def _setup_api_keys() -> None:
    """Copy project-specific API keys to standard env vars for litellm."""
    if "CHEDDAR_OPENAI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["CHEDDAR_OPENAI_API_KEY"]
    if "CHEDDAR_ANTHROPIC_API_KEY" in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CHEDDAR_ANTHROPIC_API_KEY"]


def complete_structured[T: BaseModel](
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    max_retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 16384,
    reasoning_effort: str | None = None,
) -> T:
    """Call LLM with structured output using litellm.

    Args:
        model: Model identifier (e.g., "gpt-4.1-mini", "claude-3-haiku-20240307")
        messages: Chat messages in OpenAI format
        response_model: Pydantic model for structured output
        max_retries: Number of retries on transient failures (default: 3)
        temperature: Sampling temperature (default: 0.1 for near-deterministic output)
        max_tokens: Maximum output tokens (default: 16384)
        reasoning_effort: Reasoning effort for reasoning models (e.g., "low", "medium", "high")

    Returns:
        Parsed response matching response_model schema

    Raises:
        LLMError: On API errors or parsing failures
    """
    _setup_api_keys()

    # Build response_format for native structured outputs
    schema = response_model.model_json_schema()

    # OpenAI/Azure strict mode requires additionalProperties: false
    # and all properties listed in 'required' at every object level
    def _enforce_strict_schema(obj: dict[str, Any]) -> dict[str, Any]:
        """Recursively enforce strict mode constraints on JSON schema."""
        if obj.get("type") == "object" or "properties" in obj:
            obj["additionalProperties"] = False
            if "properties" in obj:
                obj["required"] = list(obj["properties"].keys())
        for key in obj:
            child = obj[key]
            if isinstance(child, dict):
                _enforce_strict_schema(child)
            elif isinstance(child, list):
                for entry in cast(list[Any], child):
                    if isinstance(entry, dict):
                        _enforce_strict_schema(entry)
        return obj

    schema = _enforce_strict_schema(schema)

    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "strict": True,
            "schema": schema,
        },
    }

    azure_kwargs = _build_azure_kwargs(model)

    last_error: Exception | None = None
    raw_content: str | None = None

    for attempt in range(max_retries + 1):
        try:
            call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "num_retries": 3,  # litellm handles rate limit backoff
                **azure_kwargs,
            }
            if reasoning_effort is not None:
                # Reasoning models (gpt-5*) don't support temperature
                call_kwargs["reasoning_effort"] = reasoning_effort
            else:
                call_kwargs["temperature"] = temperature
            response = cast(ModelResponse, completion(**call_kwargs))
            choice = cast(Choices, response.choices[0])
            raw_content = str(choice.message.content or "")

            if not raw_content:
                raise LLMError("LLM returned empty response", raw_response=None)

            # Parse JSON and validate with Pydantic
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                # Attempt repair (handles invalid \uXXXX escapes, trailing commas, etc.)
                repaired: str = str(repair_json(raw_content))
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError as e2:
                    last_error = e2
                    if attempt < max_retries:
                        continue
                    raise LLMError(
                        f"Failed to parse LLM response after {attempt + 1} attempts: {e2}",
                        raw_response=raw_content,
                    ) from e2

            try:
                return response_model.model_validate(data)
            except ValidationError as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise LLMError(
                    f"Failed to validate LLM response after {attempt + 1} attempts: {e}",
                    raw_response=raw_content,
                ) from e

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                continue
            raise LLMError(
                f"LLM API error after {attempt + 1} attempts: {e}",
                raw_response=str(raw_content) if raw_content else None,
            ) from e

    # Should not reach here, but satisfy type checker
    raise LLMError(
        f"LLM call failed: {last_error}",
        raw_response=str(raw_content) if raw_content else None,
    )
