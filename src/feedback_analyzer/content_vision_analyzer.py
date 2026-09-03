from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from feedback_analyzer.config import Settings
from feedback_analyzer.content_schemas import ContentSegmentCandidate, VisualContentAnalysis
from feedback_analyzer.providers.protocol import LLMProvider
from feedback_analyzer.schemas import Transcript
from feedback_analyzer.utils import format_timestamp, load_prompt, save_json_file
from feedback_analyzer.vision_analyzer import _timestamp_from_frame, _transcript_context

ProgressCallback = Callable[[int, int, str], None]


def analyze_content_visual(
    segment: ContentSegmentCandidate,
    transcript: Transcript,
    frame_paths: list[Path],
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
) -> VisualContentAnalysis:
    output_path = work_dir / "content-visual-analysis" / f"{segment.id}.json"
    if output_path.exists() and not force:
        return VisualContentAnalysis.model_validate_json(output_path.read_text(encoding="utf-8"))

    context_segments = _transcript_context(transcript, segment.start_seconds, segment.end_seconds)
    frame_descriptions = "\n".join(
        f"- {path.name} at {format_timestamp(_timestamp_from_frame(path))}"
        for path in frame_paths
    )
    schema_json = json.dumps(VisualContentAnalysis.model_json_schema(), indent=2)
    prompt = load_prompt(
        "analyze_content_visual.md",
        CANDIDATE_JSON=segment.model_dump_json(indent=2),
        TRANSCRIPT_CONTEXT=context_segments,
        FRAME_DESCRIPTIONS=frame_descriptions or "No frames available.",
        SCHEMA_JSON=schema_json,
    )

    analysis = provider.prompt_json(
        prompt,
        VisualContentAnalysis,
        stage="content_visual_analysis",
        feedback_id=segment.id,
        images=frame_paths,
        model=settings.effective_vision_model(),
    )
    enriched = analysis.model_copy(update={"evidence_frames": [path.name for path in frame_paths]})
    save_json_file(output_path, enriched)
    return enriched


def analyze_content_visual_parallel(
    segments: list[ContentSegmentCandidate],
    transcript: Transcript,
    frames_root: Path,
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> list[VisualContentAnalysis]:
    if not segments:
        return []

    total = len(segments)
    visual_dir = work_dir / "content-visual-analysis"
    pending = [
        segment
        for segment in segments
        if force or not (visual_dir / f"{segment.id}.json").exists()
    ]
    cached_results = {
        segment.id: VisualContentAnalysis.model_validate_json(
            (visual_dir / f"{segment.id}.json").read_text(encoding="utf-8")
        )
        for segment in segments
        if not force and (visual_dir / f"{segment.id}.json").exists()
    }

    completed = len(cached_results)
    if on_progress:
        on_progress(completed, total, f"Loaded {completed} cached content visual analyses")

    if pending:
        max_workers = min(settings.vision_concurrency, len(pending))

        def analyze_segment(segment: ContentSegmentCandidate) -> VisualContentAnalysis:
            frame_paths = sorted((frames_root / segment.id).glob("*.jpg"))
            return analyze_content_visual(
                segment,
                transcript,
                frame_paths,
                work_dir,
                provider,
                settings,
                force=force,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_segment, segment): segment for segment in pending}
            for future in as_completed(futures):
                segment = futures[future]
                result = future.result()
                cached_results[segment.id] = result
                completed += 1
                if on_progress:
                    on_progress(
                        completed,
                        total,
                        f"Content visual analysis {completed}/{total} "
                        f"({settings.effective_vision_model() or 'default'}, up to {max_workers} parallel)",
                    )

    return [cached_results[segment.id] for segment in segments]
