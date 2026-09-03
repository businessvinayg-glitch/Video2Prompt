from __future__ import annotations

from pathlib import Path

from video2prompt.config import LLMProviderName, Settings
from video2prompt.providers.openai_api import OpenAIProvider
from video2prompt.providers.claude_cli import ClaudeCliProvider
from video2prompt.providers.codex_cli import CodexCliProvider
from video2prompt.providers.cursor import CursorProvider
from video2prompt.providers.errors import LLMProviderError
from video2prompt.providers.protocol import LLMProvider
from video2prompt.providers.subscription import SubscriptionProvider


def create_provider(settings: Settings, project_path: Path | None = None) -> LLMProvider:
    provider = settings.llm_provider
    if provider == "cursor":
        return CursorProvider(settings, project_path)
    if provider == "claude":
        return ClaudeCliProvider(settings, project_path)
    if provider == "codex":
        return CodexCliProvider(settings, project_path)
    if provider == "openai":
        return OpenAIProvider(settings, project_path)
    if provider == "subscription":
        return SubscriptionProvider(settings, project_path)
    raise LLMProviderError(f"Unsupported LLM provider: {provider}", retryable=False)
