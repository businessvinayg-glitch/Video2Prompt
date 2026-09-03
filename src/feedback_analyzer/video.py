from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from feedback_analyzer.config import SUPPORTED_VIDEO_EXTENSIONS
from feedback_analyzer.schemas import VideoMetadata
from feedback_analyzer.utils import run_command, sha256_file


class FFmpegNotFoundError(RuntimeError):
    pass


class VideoValidationError(RuntimeError):
    pass


def ensure_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FFmpegNotFoundError(
            "FFmpeg was not found.\n\n"
            "Install FFmpeg and ensure `ffmpeg` and `ffprobe` are available on PATH."
        )


def probe_video(path: Path, *, verbose: bool = False) -> VideoMetadata:
    ensure_ffmpeg_available()
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )
    if not path.is_file():
        raise VideoValidationError(f"Video file not found: {path}")

    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = run_command(command, verbose=verbose)
    except subprocess.CalledProcessError as exc:
        raise VideoValidationError(f"Failed to probe video: {path}\n{exc.stderr}") from exc

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise VideoValidationError(f"No video stream found in: {path}")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise VideoValidationError(f"Invalid or missing duration for: {path}")

    frame_rate = None
    if video_stream.get("avg_frame_rate"):
        num, _, den = video_stream["avg_frame_rate"].partition("/")
        if den and float(den) != 0:
            frame_rate = float(num) / float(den)

    stat = path.stat()
    return VideoMetadata(
        path=str(path.resolve()),
        filename=path.name,
        duration=duration,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        frame_rate=frame_rate,
        has_audio=audio_stream is not None,
        file_size=stat.st_size,
        modified_at=stat.st_mtime,
        sha256=sha256_file(path),
    )


def extract_audio(
    video_path: Path,
    output_path: Path,
    *,
    verbose: bool = False,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path

    ensure_ffmpeg_available()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        run_command(command, verbose=verbose)
    except subprocess.CalledProcessError as exc:
        raise VideoValidationError(f"Failed to extract audio from {video_path}\n{exc.stderr}") from exc
    return output_path


def extract_audio_segment(
    audio_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    verbose: bool = False,
) -> Path:
    ensure_ffmpeg_available()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        run_command(command, verbose=verbose)
    except subprocess.CalledProcessError as exc:
        raise VideoValidationError(
            f"Failed to extract audio segment from {audio_path}\n{exc.stderr}"
        ) from exc
    return output_path
