from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path

from video2prompt.config import MAX_IMAGES_PER_VISION_REQUEST, Settings
from video2prompt.providers.base import JsonValidatingProvider
from video2prompt.providers.errors import LLMProviderError
from video2prompt.utils import strip_json_fence

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIProvider(JsonValidatingProvider):
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
        return self._run(prompt, stage=stage, feedback_id=feedback_id, images=images, model=model)

    def _run(
        self,
        prompt: str,
        *,
        stage: str,
        feedback_id: str | None,
        images: list[Path] | None,
        model: str | None,
    ) -> str:
        if not self.settings.openai_api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY is not configured.\n\n"
                "Add your OpenAI API key to .env when using the openai provider.",
                stage=stage,
                feedback_id=feedback_id,
            )

        selected_model = model or self.settings.openai_model
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for image_path in (images or [])[:MAX_IMAGES_PER_VISION_REQUEST]:
            content.append(self._image_content(image_path))

        payload = {
            "model": selected_model,
            "input": [{"role": "user", "content": content}],
        }
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise LLMProviderError(
                f"OpenAI API error during stage '{stage}' (HTTP {exc.code}): {details}",
                stage=stage,
                feedback_id=feedback_id,
                retryable=exc.code in {408, 409, 429, 500, 502, 503, 504},
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(
                f"OpenAI API request failed during stage '{stage}': {exc}",
                stage=stage,
                feedback_id=feedback_id,
                retryable=True,
            ) from exc

        text = self._extract_response_text(body)
        if not text:
            raise LLMProviderError(
                f"OpenAI API returned empty output for stage '{stage}'.",
                stage=stage,
                feedback_id=feedback_id,
                retryable=True,
            )
        return strip_json_fence(text)

    def _image_content(self, image_path: Path) -> dict:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{encoded}",
        }

    @staticmethod
    def _extract_response_text(body: dict) -> str:
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        chunks: list[str] = []
        for item in body.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunks).strip()
