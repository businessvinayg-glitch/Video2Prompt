from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, TypeVar

from video2prompt.config import (
    MIN_DURATION_FOR_PARALLEL_TRANSCRIPTION,
    Settings,
    TRANSCRIPTION_CHUNK_OVERLAP_SECONDS,
    parallel_whisper_on_cpu_enabled,
)
from video2prompt.schemas import Transcript, TranscriptSegment, VideoMetadata
from video2prompt.utils import format_duration, format_timestamp, save_json_file
from video2prompt.video import extract_audio_segment

ProgressCallback = Callable[[float, str], None]
T = TypeVar("T")

WHISPER_MODEL_SIZE_HINTS: dict[str, str] = {
    "tiny": "~75 MB",
    "tiny.en": "~75 MB",
    "base": "~150 MB",
    "base.en": "~150 MB",
    "small": "~500 MB",
    "small.en": "~500 MB",
    "medium": "~1.5 GB",
    "medium.en": "~1.5 GB",
    "large": "~3 GB",
    "large-v2": "~3 GB",
    "large-v3": "~3 GB",
}


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset_seconds: float
    index: int
    core_start: float
    core_end: float

    @property
    def core_duration(self) -> float:
        return max(0.0, self.core_end - self.core_start)


@dataclass
class ParallelTranscriptionProgress:
    total_chunks: int
    audio_duration: float
    completed_chunks: int = 0
    in_progress: set[int] | None = None
    chunk_fractions: dict[int, float] | None = None
    started_at: float = 0.0
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self.in_progress is None:
            self.in_progress = set()
        if self.chunk_fractions is None:
            self.chunk_fractions = {}
        if self._lock is None:
            self._lock = threading.Lock()
        if not self.started_at:
            self.started_at = time.monotonic()

    def mark_started(self, index: int) -> None:
        assert self._lock is not None and self.in_progress is not None
        with self._lock:
            self.in_progress.add(index)
            self.chunk_fractions.setdefault(index, 0.0)

    def mark_done(self, index: int) -> None:
        assert self._lock is not None and self.in_progress is not None and self.chunk_fractions is not None
        with self._lock:
            self.in_progress.discard(index)
            self.chunk_fractions[index] = 1.0
            self.completed_chunks += 1

    def mark_failed(self, index: int) -> None:
        assert self._lock is not None and self.in_progress is not None
        with self._lock:
            self.in_progress.discard(index)

    def update_chunk(self, index: int, fraction: float) -> None:
        assert self._lock is not None and self.chunk_fractions is not None
        with self._lock:
            self.chunk_fractions[index] = min(1.0, max(fraction, self.chunk_fractions.get(index, 0.0)))

    def snapshot(self) -> tuple[float, int, int, int]:
        assert self._lock is not None and self.in_progress is not None and self.chunk_fractions is not None
        with self._lock:
            elapsed = int(time.monotonic() - self.started_at)
            in_flight = len(self.in_progress)
            progress = self.completed_chunks + sum(
                self.chunk_fractions.get(index, 0.0) for index in self.in_progress
            )
            overall_fraction = min(1.0, progress / self.total_chunks)
            estimated_seconds = overall_fraction * self.audio_duration
            return estimated_seconds, self.completed_chunks, in_flight, elapsed


def render_transcript_markdown(transcript: Transcript) -> str:
    lines: list[str] = []
    for segment in transcript.segments:
        lines.append(f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}]")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def transcribe_audio(
    audio_path: Path,
    metadata: VideoMetadata,
    settings: Settings,
    *,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Transcript:
    json_path = audio_path.parent / "transcript.json"
    md_path = audio_path.parent / "transcript.md"
    if json_path.exists() and not force:
        transcript = Transcript.model_validate_json(json_path.read_text(encoding="utf-8"))
        if not md_path.exists():
            md_path.write_text(render_transcript_markdown(transcript), encoding="utf-8")
        if on_progress:
            on_progress(1.0, "Transcript loaded from cache")
        return transcript

    engine = settings.resolve_transcription_engine()
    if engine == "codex":
        from video2prompt.codex_transcription import transcribe_audio_codex

        return transcribe_audio_codex(
            audio_path,
            metadata,
            settings,
            force=True,
            on_progress=on_progress,
        )
    if engine == "openai":
        from video2prompt.openai_transcription import transcribe_audio_openai

        return transcribe_audio_openai(
            audio_path,
            metadata,
            settings,
            force=True,
            on_progress=on_progress,
        )

    device, compute_type = _resolve_whisper_runtime(settings)
    effective_concurrency = effective_whisper_concurrency(settings, device)
    size_hint = WHISPER_MODEL_SIZE_HINTS.get(settings.whisper_model, "several hundred MB")

    try:
        from faster_whisper import WhisperModel
        from faster_whisper import download_model as whisper_download_model
    except ImportError as exc:
        raise TranscriptionError("faster-whisper is not installed.") from exc

    model_path = _ensure_whisper_model(
        settings.whisper_model,
        whisper_download_model,
        on_progress,
        size_hint=size_hint,
    )

    try:
        model = _run_blocking_with_progress(
            lambda: WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
                cpu_threads=_cpu_thread_count() if device == "cpu" else 0,
                num_workers=max(1, effective_concurrency),
            ),
            on_progress,
            f"Loading Whisper '{settings.whisper_model}' into memory on {device}",
            start_fraction=0.16,
            end_fraction=0.20,
        )
        if (
            settings.whisper_concurrency > 1
            and effective_concurrency == 1
            and device == "cpu"
            and not parallel_whisper_on_cpu_enabled()
        ):
            _report_progress(
                on_progress,
                0.19,
                "Using single-pass transcription on CPU (parallel chunking is slower on CPU)",
            )
        _report_progress(
            on_progress,
            0.20,
            _transcription_start_message(metadata.duration, settings, device, effective_concurrency),
        )
        if plan_parallel_chunks(metadata.duration, effective_concurrency, device=device) > 1:
            segments, language = _transcribe_audio_parallel(
                model,
                audio_path,
                metadata,
                settings,
                device=device,
                on_progress=on_progress,
            )
        else:
            segments, language = _transcribe_audio_single(
                model,
                audio_path,
                metadata,
                on_progress=on_progress,
            )
    except Exception as exc:
        if settings.whisper_device == "auto" and device == "cuda":
            _report_progress(on_progress, 0.0, "CUDA failed — retrying transcription on CPU...")
            return transcribe_audio_with_device(
                audio_path,
                metadata,
                settings,
                device="cpu",
                force=force,
                on_progress=on_progress,
            )
        raise TranscriptionError(
            f"Whisper transcription failed.\n"
            f"Model: {settings.whisper_model}\n"
            f"Device: {device}\n"
            f"Error: {exc}"
        ) from exc

    _report_progress(on_progress, 1.0, "Transcription complete — saving transcript")
    transcript = Transcript(
        segments=segments,
        language=language,
        duration=metadata.duration,
    )
    save_json_file(json_path, transcript)
    md_path.write_text(render_transcript_markdown(transcript), encoding="utf-8")
    return transcript


def transcribe_audio_with_device(
    audio_path: Path,
    metadata: VideoMetadata,
    settings: Settings,
    *,
    device: str,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Transcript:
    overridden = replace(settings, whisper_device=device)  # type: ignore[arg-type]
    return transcribe_audio(
        audio_path,
        metadata,
        overridden,
        force=force,
        on_progress=on_progress,
    )


def _report_progress(
    callback: ProgressCallback | None,
    fraction: float,
    message: str,
) -> None:
    if callback is not None:
        callback(fraction, message)


def _ensure_whisper_model(
    model_name: str,
    whisper_download_model: Callable[..., str],
    on_progress: ProgressCallback | None,
    *,
    size_hint: str,
) -> str:
    _report_progress(on_progress, 0.0, f"Checking Whisper '{model_name}' model cache...")

    try:
        path = whisper_download_model(model_name, local_files_only=True)
        _report_progress(on_progress, 0.12, "Whisper model found in cache")
        return path
    except Exception as exc:
        if _is_incomplete_model_cache(exc):
            _report_progress(
                on_progress,
                0.02,
                f"Resuming interrupted download for Whisper '{model_name}' ({size_hint})...",
            )
        else:
            _report_progress(
                on_progress,
                0.02,
                f"Downloading Whisper '{model_name}' from Hugging Face ({size_hint})...",
            )

    return _run_blocking_with_progress(
        lambda: whisper_download_model(model_name),
        on_progress,
        f"Downloading Whisper '{model_name}'",
        start_fraction=0.02,
        end_fraction=0.14,
    )


def _is_incomplete_model_cache(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    if name == "IncompleteSnapshotError":
        return True
    message = str(exc).lower()
    return "incomplete" in message and "model.bin" in message


def _run_blocking_with_progress(
    operation: Callable[[], T],
    on_progress: ProgressCallback | None,
    message: str,
    *,
    start_fraction: float = 0.0,
    end_fraction: float = 0.2,
    poll_seconds: float = 1.0,
) -> T:
    result: list[T] = []
    error: list[BaseException] = []
    done = threading.Event()

    def worker() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    started = time.monotonic()
    while not done.is_set():
        elapsed = int(time.monotonic() - started)
        _report_progress(on_progress, start_fraction, f"{message} — {elapsed}s elapsed...")
        done.wait(poll_seconds)

    if error:
        raise error[0]
    _report_progress(on_progress, end_fraction, f"{message} — done")
    return result[0]


def _resolve_whisper_runtime(settings: Settings) -> tuple[str, str]:
    device = settings.whisper_device
    compute_type = settings.whisper_compute_type

    if device == "auto":
        device = _detect_cuda_device()

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def _detect_cuda_device() -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def plan_parallel_chunks(duration: float, concurrency: int, *, device: str) -> int:
    effective = effective_whisper_concurrency_value(concurrency, device)
    if effective <= 1:
        return 1
    if duration < MIN_DURATION_FOR_PARALLEL_TRANSCRIPTION:
        return 1
    return min(effective, max(1, int(duration // 60)))


def effective_whisper_concurrency(settings: Settings, device: str) -> int:
    return effective_whisper_concurrency_value(settings.whisper_concurrency, device)


def effective_whisper_concurrency_value(requested: int, device: str) -> int:
    if requested <= 1:
        return 1
    if device == "cpu" and not parallel_whisper_on_cpu_enabled():
        return 1
    return requested


def _cpu_thread_count() -> int:
    return max(4, os.cpu_count() or 4)


def plan_audio_chunks(duration: float, num_chunks: int) -> list[AudioChunk]:
    if num_chunks <= 1:
        return []

    core_length = duration / num_chunks
    chunks: list[AudioChunk] = []
    for index in range(num_chunks):
        core_start = index * core_length
        core_end = duration if index == num_chunks - 1 else (index + 1) * core_length
        chunks.append(
            AudioChunk(
                path=Path(),
                offset_seconds=core_start,
                index=index,
                core_start=core_start,
                core_end=core_end,
            )
        )
    return chunks


def merge_transcript_segments(
    chunk_segments: list[tuple[AudioChunk, list[TranscriptSegment]]],
) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for chunk, segments in chunk_segments:
        filtered = _segments_for_chunk(segments, chunk)
        for segment in filtered:
            merged.append(
                TranscriptSegment(
                    start=round(segment.start + chunk.offset_seconds, 3),
                    end=round(segment.end + chunk.offset_seconds, 3),
                    text=segment.text.strip(),
                )
            )
    merged.sort(key=lambda item: item.start)
    return _dedupe_segments(merged)


def _segments_for_chunk(segments: list[TranscriptSegment], chunk: AudioChunk) -> list[TranscriptSegment]:
    kept: list[TranscriptSegment] = []
    for segment in segments:
        if not segment.text.strip():
            continue
        midpoint = (segment.start + segment.end) / 2
        if midpoint < chunk.core_start or midpoint >= chunk.core_end:
            continue
        kept.append(segment)
    return kept


def _dedupe_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []
    deduped = [segments[0]]
    for segment in segments[1:]:
        previous = deduped[-1]
        if segment.text.strip() == previous.text.strip() and segment.start <= previous.end + 0.5:
            continue
        if segment.start < previous.end - 0.25:
            if segment.end > previous.end:
                deduped[-1] = TranscriptSegment(
                    start=previous.start,
                    end=max(previous.end, segment.end),
                    text=previous.text if len(previous.text) >= len(segment.text) else segment.text,
                )
            continue
        deduped.append(segment)
    return deduped


def _transcription_start_message(
    duration: float,
    settings: Settings,
    device: str,
    effective_concurrency: int,
) -> str:
    chunks = plan_parallel_chunks(duration, effective_concurrency, device=device)
    if chunks > 1:
        return (
            f"Transcribing {format_duration(duration)} on {device} "
            f"using {chunks} parallel chunks..."
        )
    return f"Transcribing {format_duration(duration)} of audio on {device}..."


def _transcribe_audio_single(
    model: Any,
    audio_path: Path,
    metadata: VideoMetadata,
    *,
    on_progress: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], str | None]:
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        if not seg.text.strip():
            continue
        segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
        if metadata.duration > 0:
            fraction = 0.20 + min(seg.end / metadata.duration, 0.99) * 0.78
        else:
            fraction = 0.5
        _report_progress(
            on_progress,
            fraction,
            f"Transcribing {format_timestamp(seg.end)} / {format_timestamp(metadata.duration)}",
        )
    return segments, getattr(info, "language", None)


def _transcribe_audio_parallel(
    model: Any,
    audio_path: Path,
    metadata: VideoMetadata,
    settings: Settings,
    *,
    device: str,
    on_progress: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], str | None]:
    effective_concurrency = effective_whisper_concurrency(settings, device)
    num_chunks = plan_parallel_chunks(metadata.duration, effective_concurrency, device=device)
    chunk_plan = plan_audio_chunks(metadata.duration, num_chunks)
    chunks_dir = audio_path.parent / "audio-chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    prepared_chunks: list[AudioChunk] = []
    overlap = TRANSCRIPTION_CHUNK_OVERLAP_SECONDS
    for chunk in chunk_plan:
        extract_start = max(0.0, chunk.core_start - overlap) if chunk.index > 0 else chunk.core_start
        extract_end = (
            metadata.duration
            if chunk.index == num_chunks - 1
            else min(metadata.duration, chunk.core_end + overlap)
        )
        chunk_path = chunks_dir / f"chunk-{chunk.index:03d}.wav"
        extract_audio_segment(
            audio_path,
            chunk_path,
            start_seconds=extract_start,
            duration_seconds=extract_end - extract_start,
            verbose=settings.verbose,
        )
        prepared_chunks.append(
            AudioChunk(
                path=chunk_path,
                offset_seconds=chunk.offset_seconds,
                index=chunk.index,
                core_start=chunk.core_start,
                core_end=chunk.core_end,
            )
        )

    completed = 0
    chunk_results: list[tuple[AudioChunk, list[TranscriptSegment]]] = []
    language: str | None = None
    tracker = ParallelTranscriptionProgress(
        total_chunks=len(prepared_chunks),
        audio_duration=metadata.duration,
    )
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(1.0):
            estimated_seconds, done_count, in_flight, elapsed = tracker.snapshot()
            if metadata.duration > 0:
                fraction = 0.20 + min(estimated_seconds / metadata.duration, 0.99) * 0.78
            else:
                fraction = 0.5
            _report_progress(
                on_progress,
                fraction,
                (
                    f"Transcribing {format_timestamp(estimated_seconds)} / "
                    f"{format_timestamp(metadata.duration)} — "
                    f"{done_count}/{len(prepared_chunks)} chunks done, "
                    f"{in_flight} running — {elapsed}s elapsed"
                ),
            )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=len(prepared_chunks)) as executor:
            futures = {
                executor.submit(_transcribe_chunk, model, chunk, settings, tracker): chunk
                for chunk in prepared_chunks
            }
            for future in as_completed(futures):
                chunk, segments, chunk_language = future.result()
                if language is None and chunk_language:
                    language = chunk_language
                chunk_results.append((chunk, segments))
                completed += 1
                estimated_seconds, done_count, in_flight, elapsed = tracker.snapshot()
                if metadata.duration > 0:
                    fraction = 0.20 + min(estimated_seconds / metadata.duration, 0.99) * 0.78
                else:
                    fraction = 0.5
                _report_progress(
                    on_progress,
                    fraction,
                    (
                        f"Transcribed chunk {completed}/{len(prepared_chunks)} "
                        f"({format_timestamp(chunk.core_end)} / {format_timestamp(metadata.duration)}) "
                        f"— {elapsed}s elapsed"
                    ),
                )
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2.0)

    chunk_results.sort(key=lambda item: item[0].index)
    return merge_transcript_segments(chunk_results), language


def _transcribe_chunk(
    model: Any,
    chunk: AudioChunk,
    settings: Settings,
    tracker: ParallelTranscriptionProgress | None = None,
) -> tuple[AudioChunk, list[TranscriptSegment], str | None]:
    if tracker is not None:
        tracker.mark_started(chunk.index)
    try:
        segments_iter, info = model.transcribe(str(chunk.path), beam_size=5, vad_filter=True)
        segments: list[TranscriptSegment] = []
        chunk_duration = max(chunk.core_duration, 1.0)
        for seg in segments_iter:
            if not seg.text.strip():
                continue
            segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
            if tracker is not None:
                tracker.update_chunk(chunk.index, min(1.0, seg.end / chunk_duration))
    except Exception:
        if tracker is not None:
            tracker.mark_failed(chunk.index)
        raise
    if tracker is not None:
        tracker.mark_done(chunk.index)
    return chunk, segments, getattr(info, "language", None)
