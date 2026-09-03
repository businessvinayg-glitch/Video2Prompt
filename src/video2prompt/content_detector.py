from __future__ import annotations

import json
from pathlib import Path

from video2prompt.config import Settings
from video2prompt.content_schemas import ContentSegmentsResult
from video2prompt.providers.protocol import LLMProvider
from video2prompt.schemas import Transcript
from video2prompt.utils import load_prompt, save_json_file


def detect_content_segments(
    transcript: Transcript,
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
) -> ContentSegmentsResult:
    output_path = work_dir / "content-segments.json"
    if output_path.exists() and not force:
        return ContentSegmentsResult.model_validate_json(output_path.read_text(encoding="utf-8"))

    transcript_json = json.dumps(
        {"segments": [segment.model_dump() for segment in transcript.segments]},
        indent=2,
    )
    schema_json = json.dumps(ContentSegmentsResult.model_json_schema(), indent=2)
    prompt = load_prompt(
        "detect_content_segments.md",
        TRANSCRIPT_JSON=transcript_json,
        SCHEMA_JSON=schema_json,
    )

    result = provider.prompt_json(
        prompt,
        ContentSegmentsResult,
        stage="detect_content_segments",
        model=settings.effective_reasoning_model(),
    )
    save_json_file(output_path, result)
    return result
