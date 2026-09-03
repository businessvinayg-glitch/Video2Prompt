from __future__ import annotations

import logging
from pathlib import Path

from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, SDKImage, UserMessage
from cursor_sdk.errors import CursorAgentError

from video2prompt.config import MAX_IMAGES_PER_VISION_REQUEST, Settings
from video2prompt.providers.base import JsonValidatingProvider
from video2prompt.providers.errors import LLMProviderError

logger = logging.getLogger(__name__)


class CursorProvider(JsonValidatingProvider):
    def __init__(self, settings: Settings, project_path: Path | None = None) -> None:
        self.settings = settings
        self.project_path = project_path or Path.cwd()

    def prompt_text(
        self,
        prompt: str,
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> str:
        message = self._build_message(prompt, images)
        return self._run_prompt(message, stage=stage, feedback_id=feedback_id, model=model)

    def _build_message(self, prompt: str, images: list[Path] | None) -> str | UserMessage:
        if not images:
            return prompt
        limited = images[:MAX_IMAGES_PER_VISION_REQUEST]
        sdk_images = [SDKImage.from_file(path) for path in limited]
        return UserMessage(text=prompt, images=sdk_images)

    def _run_prompt(
        self,
        message: str | UserMessage,
        *,
        stage: str,
        feedback_id: str | None,
        model: str | None = None,
    ) -> str:
        if not self.settings.cursor_api_key:
            raise LLMProviderError(
                "CURSOR_API_KEY is not configured.\n\n"
                "Create a Cursor API key and add it to .env or your shell environment.",
                stage=stage,
                feedback_id=feedback_id,
            )

        selected_model = model or self.settings.cursor_model
        try:
            result = Agent.prompt(
                message,
                AgentOptions(
                    api_key=self.settings.cursor_api_key,
                    model=selected_model,
                    local=LocalAgentOptions(cwd=str(self.project_path.resolve())),
                ),
            )
        except CursorAgentError as exc:
            raise LLMProviderError(
                f"Cursor agent error during '{stage}': {exc}",
                stage=stage,
                feedback_id=feedback_id,
                retryable=True,
            ) from exc

        if result.status != "finished":
            raise LLMProviderError(
                f"Cursor run did not finish successfully (status={result.status}).",
                stage=stage,
                feedback_id=feedback_id,
                run_id=result.id,
                retryable=result.status in {"error", "expired"},
            )

        text = (result.result or "").strip()
        if not text:
            raise LLMProviderError(
                f"Cursor returned empty result for stage '{stage}'.",
                stage=stage,
                feedback_id=feedback_id,
                run_id=result.id,
                retryable=True,
            )

        if self.settings.verbose:
            logger.info(
                "Cursor stage=%s model=%s feedback_id=%s run_id=%s duration_ms=%s",
                stage,
                selected_model,
                feedback_id,
                result.id,
                result.duration_ms,
            )

        return text
