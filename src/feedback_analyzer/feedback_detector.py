from __future__ import annotations

import json
from pathlib import Path

from feedback_analyzer.config import Settings
from feedback_analyzer.providers.protocol import LLMProvider
from feedback_analyzer.schemas import FeedbackCandidatesResult, Transcript
from feedback_analyzer.utils import load_prompt, save_json_file


def detect_feedback_candidates(
    transcript: Transcript,
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
) -> FeedbackCandidatesResult:
    output_path = work_dir / "feedback-candidates.json"
    if output_path.exists() and not force:
        return FeedbackCandidatesResult.model_validate_json(output_path.read_text(encoding="utf-8"))

    transcript_json = json.dumps(
        {"segments": [segment.model_dump() for segment in transcript.segments]},
        indent=2,
    )
    schema_json = json.dumps(FeedbackCandidatesResult.model_json_schema(), indent=2)
    prompt = load_prompt(
        "detect_feedback.md",
        TRANSCRIPT_JSON=transcript_json,
        SCHEMA_JSON=schema_json,
    )

    result = provider.prompt_json(
        prompt,
        FeedbackCandidatesResult,
        stage="detect_feedback",
        model=settings.effective_reasoning_model(),
    )
    save_json_file(output_path, result)
    return result
