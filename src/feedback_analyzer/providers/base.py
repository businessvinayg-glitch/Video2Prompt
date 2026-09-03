from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from feedback_analyzer.providers.errors import LLMProviderError
from feedback_analyzer.utils import validation_error_summary, validate_model_json

T = TypeVar("T", bound=BaseModel)

MAX_JSON_RETRIES = 1


class JsonValidatingProvider:
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
        text = self.prompt_text(
            prompt,
            stage=stage,
            feedback_id=feedback_id,
            images=images,
            model=model,
        )
        try:
            return validate_model_json(text, schema)
        except (ValidationError, ValueError) as first_error:
            retry_prompt = (
                f"{prompt}\n\n"
                "Your previous response did not match the required JSON schema.\n"
                f"Validation error: {first_error}\n"
                "Return corrected JSON only."
            )
            retry_text = self.prompt_text(
                retry_prompt,
                stage=stage,
                feedback_id=feedback_id,
                images=images,
                model=model,
            )
            try:
                return validate_model_json(retry_text, schema)
            except (ValidationError, ValueError) as second_error:
                summary = (
                    validation_error_summary(second_error)
                    if isinstance(second_error, ValidationError)
                    else str(second_error)
                )
                raise LLMProviderError(
                    f"LLM returned invalid JSON for stage '{stage}': {summary}",
                    stage=stage,
                    feedback_id=feedback_id,
                    retryable=False,
                ) from second_error
