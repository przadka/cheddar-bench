"""Gemini CLI agent implementation."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from cheddar.agents.base import BaseAgent
from cheddar.agents.config import MODELS

# Gemini settings without auth type - forces use of env vars for auth
GEMINI_SETTINGS: dict[str, Any] = {
    "context": {"fileName": "AGENTS.md"},
    "ui": {"theme": "Default"},
    "general": {"previewFeatures": True},
}


def _get_vertex_env() -> dict[str, str]:
    """Get Vertex AI environment variables.

    Uses GOOGLE_CLOUD_PROJECT from environment, or falls back to gcloud config.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT environment variable must be set for Gemini agent. "
            "Set it in your shell or .env file."
        )
    return {
        "GOOGLE_CLOUD_PROJECT": project,
        "GOOGLE_CLOUD_LOCATION": "global",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
    }


def _write_gemini_settings(gemini_home: Path, use_vertex: bool = True) -> None:
    """Write Gemini CLI settings to a custom directory (avoids modifying ~/.gemini)."""
    settings_dir = gemini_home / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)

    auth_type = "vertex-ai" if use_vertex else "oauth-personal"
    settings: dict[str, Any] = {
        **GEMINI_SETTINGS,
        "security": {"auth": {"selectedType": auth_type}},
    }
    (settings_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


class GeminiAgent(BaseAgent):
    """Agent implementation for Gemini CLI."""

    @property
    def name(self) -> Literal["gemini"]:
        """Agent identifier."""
        return "gemini"

    @property
    def can_challenge(self) -> bool:
        """Gemini can inject bugs."""
        return True

    @property
    def can_review(self) -> bool:
        """Gemini can review code."""
        return True

    def __init__(self) -> None:
        """Set up isolated Gemini config directory."""
        self._gemini_home = Path(tempfile.mkdtemp(prefix="cheddar-gemini-"))
        use_vertex = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        _write_gemini_settings(self._gemini_home, use_vertex=use_vertex)

    def _build_challenge_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for challenge."""
        return ["gemini", "-m", MODELS["gemini"], "--yolo", "-p", prompt]

    def _build_review_cmd(self, prompt: str) -> list[str]:
        """Build CLI command for review."""
        return ["gemini", "-m", MODELS["gemini"], "--yolo", "-p", prompt]

    def _get_env(self) -> dict[str, str]:
        """Return environment variables for Gemini CLI.

        Sets HOME to an isolated temp dir so Gemini reads settings from there
        instead of modifying ~/.gemini. Preserves ADC by pointing
        GOOGLE_APPLICATION_CREDENTIALS at the real credentials file.
        """
        env: dict[str, str] = {"HOME": str(self._gemini_home)}

        # Overriding HOME breaks GCP Application Default Credentials lookup
        # (ADC lives at $HOME/.config/gcloud/application_default_credentials.json).
        # Point GOOGLE_APPLICATION_CREDENTIALS at the real file so gcloud finds it.
        real_home = os.environ.get("HOME", "")
        adc_path = os.path.join(
            real_home, ".config", "gcloud", "application_default_credentials.json"
        )
        if real_home and os.path.exists(adc_path):
            env["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path

        if os.environ.get("GOOGLE_CLOUD_PROJECT"):
            env.update(_get_vertex_env())
        return env
