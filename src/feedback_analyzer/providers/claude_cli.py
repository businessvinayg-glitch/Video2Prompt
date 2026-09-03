from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from feedback_analyzer.config import MAX_IMAGES_PER_VISION_REQUEST, Settings
from feedback_analyzer.providers.base import JsonValidatingProvider
from feedback_analyzer.providers.cli_runner import (
    append_image_paths,
    parse_claude_json_output,
    resolve_executable,
    run_cli,
)

T = TypeVar("T", bound=BaseModel)


class ClaudeCliProvider(JsonValidatingProvider):
    def __init__(self, settings: Settings, project_path: Path | None = None) -> None:
        self.settings = settings
        self.project_path = project_path or Path.cwd()
        self.executable = resolve_executable(settings.claude_cli)

    def prompt_text(
        self,
        prompt: str,
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> str:
        return self._run(prompt, stage=stage, feedback_id=feedback_id, images=images, model=model)

    def prompt_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> T:
        text = self._run(
            prompt,
            stage=stage,
            feedback_id=feedback_id,
            images=images,
            model=model,
            json_schema=schema.model_json_schema(),
        )
        return schema.model_validate_json(text)

    def _run(
        self,
        prompt: str,
        *,
        stage: str,
        feedback_id: str | None,
        images: list[Path] | None,
        model: str | None,
        json_schema: dict | None = None,
    ) -> str:
        selected_model = model or self.settings.claude_model
        limited_images = (images or [])[:MAX_IMAGES_PER_VISION_REQUEST]
        full_prompt = append_image_paths(prompt, limited_images)

        command = [
            self.executable,
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
        ]
        if selected_model:
            command.extend(["--model", selected_model])
        if not limited_images:
            command.extend(["--tools", ""])
        if json_schema is not None:
            command.extend(["--json-schema", json.dumps(json_schema)])
        if self.project_path.exists():
            command.extend(["--add-dir", str(self.project_path.resolve())])

        raw_output = run_cli(
            command,
            stage=stage,
            feedback_id=feedback_id,
            cwd=self.project_path,
            input_text=full_prompt,
        )
        return parse_claude_json_output(raw_output)
