from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from feedback_analyzer.config import Settings
from feedback_analyzer.providers.protocol import LLMProvider
from feedback_analyzer.schemas import FeedbackCandidate, Transcript, VisualFeedbackAnalysis
from feedback_analyzer.utils import format_timestamp, load_prompt, save_json_file

ProgressCallback = Callable[[int, int, str], None]


def analyze_visual_feedback(
    candidate: FeedbackCandidate,
    transcript: Transcript,
    frame_paths: list[Path],
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
) -> VisualFeedbackAnalysis:
    output_path = work_dir / "visual-analysis" / f"{candidate.id}.json"
    if output_path.exists() and not force:
        return VisualFeedbackAnalysis.model_validate_json(output_path.read_text(encoding="utf-8"))

    context_segments = _transcript_context(transcript, candidate.start_seconds, candidate.end_seconds)
    frame_descriptions = "\n".join(
        f"- {path.name} at {format_timestamp(_timestamp_from_frame(path))}"
        for path in frame_paths
    )
    schema_json = json.dumps(VisualFeedbackAnalysis.model_json_schema(), indent=2)
    prompt = load_prompt(
        "analyze_visual_feedback.md",
        CANDIDATE_JSON=candidate.model_dump_json(indent=2),
        TRANSCRIPT_CONTEXT=context_segments,
        FRAME_DESCRIPTIONS=frame_descriptions or "No frames available.",
        SCHEMA_JSON=schema_json,
    )

    analysis = provider.prompt_json(
        prompt,
        VisualFeedbackAnalysis,
        stage="visual_analysis",
        feedback_id=candidate.id,
        images=frame_paths,
        model=settings.effective_vision_model(),
    )
    enriched = analysis.model_copy(
        update={"evidence_frames": [path.name for path in frame_paths]}
    )
    save_json_file(output_path, enriched)
    return enriched


def analyze_visual_feedback_parallel(
    candidates: list[FeedbackCandidate],
    transcript: Transcript,
    frames_root: Path,
    work_dir: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> list[VisualFeedbackAnalysis]:
    if not candidates:
        return []

    total = len(candidates)
    pending = [
        candidate
        for candidate in candidates
        if force or not (work_dir / "visual-analysis" / f"{candidate.id}.json").exists()
    ]
    cached_results = {
        candidate.id: VisualFeedbackAnalysis.model_validate_json(
            (work_dir / "visual-analysis" / f"{candidate.id}.json").read_text(encoding="utf-8")
        )
        for candidate in candidates
        if not force and (work_dir / "visual-analysis" / f"{candidate.id}.json").exists()
    }

    completed = len(cached_results)
    if on_progress:
        on_progress(completed, total, f"Loaded {completed} cached visual analyses")

    if pending:
        max_workers = min(settings.vision_concurrency, len(pending))

        def analyze_candidate(candidate: FeedbackCandidate) -> VisualFeedbackAnalysis:
            frame_paths = sorted((frames_root / candidate.id).glob("*.jpg"))
            return analyze_visual_feedback(
                candidate,
                transcript,
                frame_paths,
                work_dir,
                provider,
                settings,
                force=force,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_candidate, candidate): candidate for candidate in pending}
            for future in as_completed(futures):
                candidate = futures[future]
                result = future.result()
                cached_results[candidate.id] = result
                completed += 1
                if on_progress:
                    on_progress(
                        completed,
                        total,
                        f"Vision analysis {completed}/{total} ({settings.effective_vision_model() or 'default'}, up to {max_workers} parallel)",
                    )

    return [cached_results[candidate.id] for candidate in candidates]


def _transcript_context(transcript: Transcript, start: float, end: float, padding: float = 15.0) -> str:
    lines: list[str] = []
    window_start = max(0.0, start - padding)
    window_end = end + padding
    for segment in transcript.segments:
        if segment.end < window_start or segment.start > window_end:
            continue
        lines.append(
            f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {segment.text.strip()}"
        )
    return "\n".join(lines)


def _timestamp_from_frame(path: Path) -> float:
    parts = path.stem.split("-")
    try:
        if len(parts) == 2:
            minutes, seconds_part = parts
            seconds, millis = seconds_part.split(".", 1)
            return int(minutes) * 60 + int(seconds) + int(millis) / 1000
        if len(parts) == 3:
            hours, minutes, seconds_part = parts
            seconds, millis = seconds_part.split(".", 1)
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000
    except (ValueError, IndexError):
        pass
    return 0.0
