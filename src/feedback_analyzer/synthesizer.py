from __future__ import annotations

import json
from pathlib import Path

from feedback_analyzer.config import Settings
from feedback_analyzer.providers.protocol import LLMProvider
from feedback_analyzer.schemas import CodebaseMapping, FeedbackAnalysis, Transcript, VisualFeedbackAnalysis
from feedback_analyzer.utils import load_prompt, save_json_file


def synthesize_feedback(
    video_name: str,
    project_path: Path | None,
    transcript: Transcript,
    visual_analyses: list[VisualFeedbackAnalysis],
    codebase_mappings: dict[str, CodebaseMapping],
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
) -> FeedbackAnalysis:
    output_path = work_dir / "final-analysis.json"
    if output_path.exists() and not force:
        return FeedbackAnalysis.model_validate_json(output_path.read_text(encoding="utf-8"))

    payload = {
        "video": video_name,
        "project": str(project_path.resolve()) if project_path else None,
        "visual_analyses": [item.model_dump() for item in visual_analyses],
        "codebase_mappings": {key: value.model_dump() for key, value in codebase_mappings.items()},
        "transcript_excerpt": [
            segment.model_dump()
            for segment in transcript.segments
        ],
    }
    schema_json = json.dumps(FeedbackAnalysis.model_json_schema(), indent=2)
    prompt = load_prompt(
        "synthesize_feedback.md",
        ANALYSIS_PAYLOAD_JSON=json.dumps(payload, indent=2),
        SCHEMA_JSON=schema_json,
    )

    result = provider.prompt_json(
        prompt,
        FeedbackAnalysis,
        stage="synthesis",
        model=settings.effective_synthesis_model(),
    )
    save_json_file(output_path, result)
    return result
