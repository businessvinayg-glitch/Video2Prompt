from __future__ import annotations

from pathlib import Path

from feedback_analyzer.config import LLMProviderName, Settings
from feedback_analyzer.providers.openai_api import OpenAIProvider
from feedback_analyzer.providers.claude_cli import ClaudeCliProvider
from feedback_analyzer.providers.codex_cli import CodexCliProvider
from feedback_analyzer.providers.cursor import CursorProvider
from feedback_analyzer.providers.errors import LLMProviderError
from feedback_analyzer.providers.protocol import LLMProvider
from feedback_analyzer.providers.subscription import SubscriptionProvider


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
