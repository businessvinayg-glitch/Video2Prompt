from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from video2prompt.config import MAX_IMAGES_PER_VISION_REQUEST, Settings
from video2prompt.providers.base import JsonValidatingProvider
from video2prompt.providers.cli_runner import resolve_executable, run_cli
from video2prompt.utils import codex_compatible_json_schema

T = TypeVar("T", bound=BaseModel)


class CodexCliProvider(JsonValidatingProvider):
    def __init__(self, settings: Settings, project_path: Path | None = None) -> None:
        self.settings = settings
        self.project_path = project_path or Path.cwd()
        self.executable = resolve_executable(settings.codex_cli)

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
            json_schema=codex_compatible_json_schema(schema.model_json_schema()),
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
        selected_model = model or self.settings.codex_model
        limited_images = (images or [])[:MAX_IMAGES_PER_VISION_REQUEST]

        command = [
            self.executable,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--ephemeral",
            "-C",
            str(self.project_path.resolve()),
        ]
        if self.settings.codex_reasoning_effort:
            command.extend(["-c", f'model_reasoning_effort="{self.settings.codex_reasoning_effort}"'])
        if selected_model:
            command.extend(["-m", selected_model])
        for image_path in limited_images:
            command.extend(["-i", str(image_path.resolve())])

        schema_path: Path | None = None
        output_path: Path | None = None
        try:
            if json_schema is not None:
                schema_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".schema.json",
                    delete=False,
                    encoding="utf-8",
                )
                json.dump(json_schema, schema_file)
                schema_file.close()
                schema_path = Path(schema_file.name)
                command.extend(["--output-schema", str(schema_path)])

            output_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            )
            output_file.close()
            output_path = Path(output_file.name)
            command.extend(["-o", str(output_path), "-"])

            run_cli(
                command,
                stage=stage,
                feedback_id=feedback_id,
                input_text=prompt,
                cwd=self.project_path,
            )

            if output_path.exists():
                text = output_path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        finally:
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)

        from video2prompt.providers.errors import LLMProviderError

        raise LLMProviderError(
            f"Codex CLI returned empty output for stage '{stage}'.",
            stage=stage,
            feedback_id=feedback_id,
            retryable=True,
        )
