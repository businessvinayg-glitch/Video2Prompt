from __future__ import annotations

from pathlib import Path

from feedback_analyzer.content_schemas import ContentSegmentsResult
from feedback_analyzer.schemas import (
    FeedbackCandidatesResult,
    ProjectCacheInfo,
    SourceCacheInfo,
    Transcript,
    VideoMetadata,
    WorkCacheManifest,
)
from feedback_analyzer.utils import git_head, save_json_file


def work_dir_for_video(work_root: Path, video_path: Path) -> Path:
    return work_root / video_path.stem


def content_work_dir_for_video(work_root: Path, video_path: Path) -> Path:
    return work_root / "content" / video_path.stem


def load_manifest(work_dir: Path) -> WorkCacheManifest | None:
    path = work_dir / "cache-manifest.json"
    if not path.exists():
        return None
    return WorkCacheManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(work_dir: Path, metadata: VideoMetadata, project: ProjectCacheInfo | None) -> WorkCacheManifest:
    manifest = WorkCacheManifest(
        source=SourceCacheInfo(
            path=metadata.path,
            file_size=metadata.file_size,
            modified_at=metadata.modified_at,
            sha256=metadata.sha256,
        ),
        project=project,
    )
    save_json_file(work_dir / "cache-manifest.json", manifest)
    return manifest


def source_changed(manifest: WorkCacheManifest | None, metadata: VideoMetadata) -> bool:
    if manifest is None:
        return False
    return manifest.source.sha256 != metadata.sha256


def load_cached_transcript(work_dir: Path) -> Transcript | None:
    path = work_dir / "transcript.json"
    if not path.exists():
        return None
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def load_cached_candidates(work_dir: Path) -> FeedbackCandidatesResult | None:
    path = work_dir / "feedback-candidates.json"
    if not path.exists():
        return None
    return FeedbackCandidatesResult.model_validate_json(path.read_text(encoding="utf-8"))


def load_cached_content_segments(work_dir: Path) -> ContentSegmentsResult | None:
    path = work_dir / "content-segments.json"
    if not path.exists():
        return None
    return ContentSegmentsResult.model_validate_json(path.read_text(encoding="utf-8"))


def count_cached_frames(frames_root: Path, candidate_ids: list[str]) -> int:
    return sum(
        1
        for candidate_id in candidate_ids
        if (frames_root / candidate_id).exists() and any((frames_root / candidate_id).glob("*.jpg"))
    )


def count_cached_visual_analyses(work_dir: Path, candidate_ids: list[str]) -> int:
    visual_dir = work_dir / "visual-analysis"
    if not visual_dir.exists():
        return 0
    return sum(1 for candidate_id in candidate_ids if (visual_dir / f"{candidate_id}.json").exists())


def count_cached_content_visual_analyses(work_dir: Path, segment_ids: list[str]) -> int:
    visual_dir = work_dir / "content-visual-analysis"
    if not visual_dir.exists():
        return 0
    return sum(1 for segment_id in segment_ids if (visual_dir / f"{segment_id}.json").exists())


def describe_cached_stages(
    work_dir: Path,
    *,
    candidate_ids: list[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    transcript = load_cached_transcript(work_dir)
    if transcript is not None:
        lines.append(f"[green]✓[/green] Transcript ({len(transcript.segments)} segments)")

    candidates = load_cached_candidates(work_dir)
    if candidates is not None:
        lines.append(f"[green]✓[/green] Feedback candidates ({len(candidates.candidates)})")

    ids = candidate_ids or [candidate.id for candidate in candidates.candidates] if candidates else []
    if ids:
        frame_count = count_cached_frames(work_dir / "frames", ids)
        if frame_count:
            lines.append(f"[green]✓[/green] Screenshots ({frame_count}/{len(ids)})")

        visual_count = count_cached_visual_analyses(work_dir, ids)
        if visual_count:
            lines.append(f"[green]✓[/green] Visual analysis ({visual_count}/{len(ids)})")
        elif frame_count == len(ids):
            lines.append("[cyan]→[/cyan] Next: visual analysis")

    if (work_dir / "final-analysis.json").exists():
        lines.append("[green]✓[/green] Final synthesis")

    return lines


def describe_content_cached_stages(
    work_dir: Path,
    *,
    segment_ids: list[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    transcript = load_cached_transcript(work_dir)
    if transcript is not None:
        lines.append(f"[green]✓[/green] Transcript ({len(transcript.segments)} segments)")

    segments = load_cached_content_segments(work_dir)
    if segments is not None:
        lines.append(f"[green]✓[/green] Content segments ({len(segments.segments)})")

    ids = segment_ids or [segment.id for segment in segments.segments] if segments else []
    if ids:
        frame_count = count_cached_frames(work_dir / "frames", ids)
        if frame_count:
            lines.append(f"[green]✓[/green] Screenshots ({frame_count}/{len(ids)})")

        visual_count = count_cached_content_visual_analyses(work_dir, ids)
        if visual_count:
            lines.append(f"[green]✓[/green] Content visual analysis ({visual_count}/{len(ids)})")
        elif frame_count == len(ids):
            lines.append("[cyan]→[/cyan] Next: content visual analysis")

    if (work_dir / "content-analysis.json").exists():
        lines.append("[green]✓[/green] Content synthesis")

    return lines


def project_cache_stale(
    manifest: WorkCacheManifest | None,
    project_path: Path | None,
) -> bool:
    if project_path is None:
        return False
    if manifest is None or manifest.project is None:
        return True
    current_head = git_head(project_path)
    return manifest.project.git_head != current_head


def invalidate_from_stage(work_dir: Path, stage: str) -> None:
    patterns = {
        "metadata": ["metadata.json", "audio.wav", "transcript.json", "transcript.md"],
        "transcription": ["transcript.json", "transcript.md"],
        "candidates": ["feedback-candidates.json"],
        "frames": ["frames"],
        "visual": ["visual-analysis"],
        "codebase": ["codebase-analysis"],
        "final": ["final-analysis.json"],
    }
    for rel in patterns.get(stage, []):
        target = work_dir / rel
        if target.is_dir():
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def clear_work_cache(work_dir: Path) -> None:
    if not work_dir.exists():
        return
    import shutil

    shutil.rmtree(work_dir, ignore_errors=True)
