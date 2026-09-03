from __future__ import annotations

import subprocess
from pathlib import Path

from feedback_analyzer.config import JPEG_QUALITY, WINDOW_EXPAND_AFTER_SECONDS, WINDOW_EXPAND_BEFORE_SECONDS
from typing import Protocol

from feedback_analyzer.utils import expand_window, frame_filename, frame_timestamps_for_window, run_command
from feedback_analyzer.video import ensure_ffmpeg_available


class FrameExtractable(Protocol):
    id: str
    start_seconds: float
    end_seconds: float


def extract_frames_for_candidate(
    video_path: Path,
    candidate: FrameExtractable,
    output_dir: Path,
    *,
    duration: float,
    verbose: bool = False,
    force: bool = False,
) -> list[Path]:
    candidate_dir = output_dir / candidate.id
    if candidate_dir.exists() and not force:
        existing = sorted(candidate_dir.glob("*.jpg"))
        if existing:
            return existing

    ensure_ffmpeg_available()
    candidate_dir.mkdir(parents=True, exist_ok=True)

    window_start, window_end = expand_window(
        candidate.start_seconds,
        candidate.end_seconds,
        duration=duration,
        before=WINDOW_EXPAND_BEFORE_SECONDS,
        after=WINDOW_EXPAND_AFTER_SECONDS,
    )
    timestamps = frame_timestamps_for_window(window_start, window_end)
    extracted: list[Path] = []

    for timestamp in timestamps:
        output_path = candidate_dir / frame_filename(timestamp)
        if output_path.exists() and not force:
            extracted.append(output_path)
            continue
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            str(JPEG_QUALITY),
            str(output_path),
        ]
        try:
            run_command(command, verbose=verbose)
            extracted.append(output_path)
        except subprocess.CalledProcessError:
            continue

    return sorted(extracted)
