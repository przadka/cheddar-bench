"""Tests for LLM wrapper module."""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from cheddar.llm import LLMError, _build_azure_kwargs


class SampleResponse(BaseModel):
    """Sample response model for testing."""

    reasoning: str
    matches: bool
    confidence: Literal["high", "medium", "low"]


def test_complete_structured_success() -> None:
    """Returns parsed Pydantic model on success."""
    from cheddar.llm import complete_structured

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = '{"reasoning": "Same bug", "matches": true, "confidence": "high"}'

    with patch("cheddar.llm.completion", return_value=mock_response):
        result = complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    assert isinstance(result, SampleResponse)
    assert result.reasoning == "Same bug"
    assert result.matches is True
    assert result.confidence == "high"


def test_complete_structured_api_error() -> None:
    """Raises LLMError on API failure."""
    from cheddar.llm import LLMError, complete_structured

    with (
        patch("cheddar.llm.completion", side_effect=Exception("API rate limit exceeded")),
        pytest.raises(LLMError) as exc_info,
    ):
        complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    assert "API" in str(exc_info.value) or "rate limit" in str(exc_info.value).lower()


def test_complete_structured_parse_error() -> None:
    """Raises LLMError on invalid response format."""
    from cheddar.llm import LLMError, complete_structured

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json at all"

    with (
        patch("cheddar.llm.completion", return_value=mock_response),
        pytest.raises(LLMError) as exc_info,
    ):
        complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    assert "parse" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_complete_structured_retry_on_transient_failure() -> None:
    """Retries once on transient failure, then succeeds."""
    from cheddar.llm import complete_structured

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = '{"reasoning": "Retried", "matches": false, "confidence": "low"}'

    # First call fails, second succeeds
    call_count = 0

    def mock_completion(*args: object, **kwargs: object) -> object:
        del args, kwargs  # unused
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Transient error")
        return mock_response

    with patch("cheddar.llm.completion", side_effect=mock_completion):
        result = complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    assert call_count == 2
    assert result.reasoning == "Retried"
    assert result.matches is False


def test_build_azure_kwargs_non_azure_model() -> None:
    """Returns empty dict for non-Azure models."""
    assert _build_azure_kwargs("gpt-4.1-mini") == {}
    assert _build_azure_kwargs("claude-3-haiku-20240307") == {}


def test_build_azure_kwargs_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skips token provider when AZURE_API_KEY is set."""
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")

    result = _build_azure_kwargs("azure/gpt-4.1-mini")

    assert result["api_base"] == "https://example.openai.azure.com"
    assert result["api_version"] == "2024-10-21"
    assert "azure_ad_token_provider" not in result


def test_build_azure_kwargs_with_token_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Includes token provider when no AZURE_API_KEY."""
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_API_VERSION", "2025-03-01-preview")

    fake_provider = lambda: "fake-token"  # noqa: E731
    with patch("cheddar.llm._get_azure_token_provider", return_value=fake_provider):
        result = _build_azure_kwargs("azure/gpt-4.1-mini")

    assert result["api_base"] == "https://example.openai.azure.com"
    assert result["api_version"] == "2025-03-01-preview"
    assert result["azure_ad_token_provider"] is fake_provider


def test_get_azure_token_provider_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises LLMError when azure-identity is not installed."""
    import cheddar.llm as llm_module

    monkeypatch.setattr(llm_module, "_azure_token_provider", None)

    with (
        patch.dict("sys.modules", {"azure.identity": None}),
        pytest.raises(LLMError, match="azure-identity"),
    ):
        llm_module._get_azure_token_provider()


def test_schema_adds_additional_properties_false_in_lists() -> None:
    """Schema recursion handles anyOf/allOf lists, not just dict values."""
    from cheddar.llm import complete_structured

    class Inner(BaseModel):
        name: str

    class Outer(BaseModel):
        item: Inner | None  # Generates anyOf with list of schemas

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"item": {"name": "test"}}'

    with patch("cheddar.llm.completion", return_value=mock_response) as mock_completion:
        complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=Outer,
        )

    _, kwargs = mock_completion.call_args
    schema = kwargs["response_format"]["json_schema"]["schema"]

    # The Inner schema (in $defs or anyOf) must have additionalProperties: false
    def find_objects(obj: object) -> list[dict]:
        """Find all object-type schemas in the JSON schema tree."""
        results = []
        if isinstance(obj, dict):
            if obj.get("type") == "object" or "properties" in obj:
                results.append(obj)
            for v in obj.values():
                results.extend(find_objects(v))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(find_objects(item))
        return results

    object_schemas = find_objects(schema)
    assert len(object_schemas) >= 2  # At least Outer and Inner
    for obj_schema in object_schemas:
        assert obj_schema.get("additionalProperties") is False, (
            f"Missing additionalProperties: false in {obj_schema}"
        )


def test_complete_structured_passes_num_retries() -> None:
    """complete_structured passes num_retries=3 to litellm for rate limit backoff."""
    from cheddar.llm import complete_structured

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = '{"reasoning": "test", "matches": true, "confidence": "high"}'

    with patch("cheddar.llm.completion", return_value=mock_response) as mock_completion:
        complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    _, kwargs = mock_completion.call_args
    assert kwargs["num_retries"] == 3


def test_complete_structured_json_repair() -> None:
    """Recovers from invalid Unicode escapes via json-repair."""
    from cheddar.llm import complete_structured

    # Simulate LLM returning JSON with an invalid \uXXXX escape
    broken_json = '{"reasoning": "Has unicode \\u00zz here", "matches": true, "confidence": "high"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = broken_json

    with patch("cheddar.llm.completion", return_value=mock_response):
        result = complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    assert isinstance(result, SampleResponse)
    assert result.matches is True
    assert result.confidence == "high"


def test_complete_structured_json_repair_trailing_comma() -> None:
    """Recovers from trailing comma in JSON via json-repair."""
    from cheddar.llm import complete_structured

    broken_json = '{"reasoning": "test", "matches": false, "confidence": "low",}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = broken_json

    with patch("cheddar.llm.completion", return_value=mock_response):
        result = complete_structured(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleResponse,
        )

    assert isinstance(result, SampleResponse)
    assert result.matches is False
