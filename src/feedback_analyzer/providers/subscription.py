from __future__ import annotations

import shutil
from pathlib import Path

from feedback_analyzer.config import Settings
from feedback_analyzer.providers.claude_cli import ClaudeCliProvider
from feedback_analyzer.providers.codex_cli import CodexCliProvider
from feedback_analyzer.providers.errors import LLMProviderError
from feedback_analyzer.providers.protocol import LLMProvider


class SubscriptionProvider:
    """Use the first available subscription CLI (Claude Code, then Codex)."""

    def __init__(self, settings: Settings, project_path: Path | None = None) -> None:
        self.settings = settings
        self.project_path = project_path or Path.cwd()
        self._delegate = self._select_delegate()

    def _select_delegate(self) -> LLMProvider:
        if shutil.which(self.settings.claude_cli):
            return ClaudeCliProvider(self.settings, self.project_path)
        if shutil.which(self.settings.codex_cli):
            return CodexCliProvider(self.settings, self.project_path)
        raise LLMProviderError(
            "No subscription CLI found on PATH.\n\n"
            "Install Claude Code (`claude`) or Codex (`codex`) and authenticate with "
            "`claude login` or `codex login`, or choose a specific provider with --provider.",
            retryable=False,
        )

    def prompt_text(
        self,
        prompt: str,
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> str:
        return self._delegate.prompt_text(
            prompt,
            stage=stage,
            feedback_id=feedback_id,
            images=images,
            model=model,
        )

    def prompt_json(
        self,
        prompt: str,
        schema: type,
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ):
        return self._delegate.prompt_json(
            prompt,
            schema,
            stage=stage,
            feedback_id=feedback_id,
            images=images,
            model=model,
        )
