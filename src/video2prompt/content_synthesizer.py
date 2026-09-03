from __future__ import annotations

import json
from pathlib import Path

from video2prompt.config import Settings
from video2prompt.content_schemas import ContentFeatureAnalysis, VisualContentAnalysis
from video2prompt.providers.protocol import LLMProvider
from video2prompt.schemas import Transcript
from video2prompt.utils import load_prompt, save_json_file


def synthesize_content_features(
    video_name: str,
    transcript: Transcript,
    visual_analyses: list[VisualContentAnalysis],
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
) -> ContentFeatureAnalysis:
    output_path = work_dir / "content-analysis.json"
    if output_path.exists() and not force:
        return ContentFeatureAnalysis.model_validate_json(output_path.read_text(encoding="utf-8"))

    payload = {
        "video": video_name,
        "visual_analyses": [item.model_dump() for item in visual_analyses],
        "transcript_excerpt": [segment.model_dump() for segment in transcript.segments],
    }
    schema_json = json.dumps(ContentFeatureAnalysis.model_json_schema(), indent=2)
    prompt = load_prompt(
        "synthesize_content.md",
        ANALYSIS_PAYLOAD_JSON=json.dumps(payload, indent=2),
        SCHEMA_JSON=schema_json,
    )

    result = provider.prompt_json(
        prompt,
        ContentFeatureAnalysis,
        stage="content_synthesis",
        model=settings.effective_synthesis_model(),
    )
    save_json_file(output_path, result)
    return result
