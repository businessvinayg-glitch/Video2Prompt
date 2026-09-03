from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from feedback_analyzer.cache import (
    clear_work_cache,
    content_work_dir_for_video,
    count_cached_content_visual_analyses,
    count_cached_frames,
    count_cached_visual_analyses,
    describe_cached_stages,
    describe_content_cached_stages,
    load_cached_transcript,
    load_manifest,
    project_cache_stale,
    save_manifest,
    source_changed,
    work_dir_for_video,
)
from feedback_analyzer.codebase_mapper import capture_project_cache_info, map_feedback_to_codebase
from feedback_analyzer.config import SUPPORTED_LLM_PROVIDERS, Settings
from feedback_analyzer.content_detector import detect_content_segments
from feedback_analyzer.content_prompt_builder import build_playbook_generation_prompt
from feedback_analyzer.content_synthesizer import synthesize_content_features
from feedback_analyzer.content_vision_analyzer import analyze_content_visual_parallel
from feedback_analyzer.interactive import prompt_continue, prompt_video_path, run_start_wizard
from feedback_analyzer.providers import LLMProviderError, create_provider
from feedback_analyzer.feedback_detector import detect_feedback_candidates
from feedback_analyzer.frames import extract_frames_for_candidate
from feedback_analyzer.prompt_builder import build_implementation_prompt
from feedback_analyzer.schemas import CodebaseMapping, ProjectCacheInfo
from feedback_analyzer.synthesizer import synthesize_feedback
from feedback_analyzer.transcription import TranscriptionError, transcribe_audio
from feedback_analyzer.utils import format_duration, is_supported_video, save_json_file
from feedback_analyzer.video import FFmpegNotFoundError, VideoValidationError, extract_audio, probe_video
from feedback_analyzer.vision_analyzer import analyze_visual_feedback_parallel

app = typer.Typer(
    name="video2prompt",
    help="Turn videos into actionable prompts — client feedback or content playbooks.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Turn videos into actionable prompts — client feedback or content playbooks."""


@app.command("analyze")
def analyze_command(
    video: Path | None = typer.Argument(
        None,
        help="Path to a specific video file. If omitted, process all videos in input/.",
        exists=False,
        readable=False,
    ),
    project: Path | None = typer.Option(
        None,
        "--project",
        help="Optional project directory for codebase-aware analysis.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help=f"LLM provider ({', '.join(SUPPORTED_LLM_PROVIDERS)}).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Reasoning model for feedback detection and codebase mapping.",
    ),
    vision_model: str | None = typer.Option(
        None,
        "--vision-model",
        help="Model for screenshot / image analysis.",
    ),
    synthesis_model: str | None = typer.Option(
        None,
        "--synthesis-model",
        help="Model for the final implementation prompt synthesis.",
    ),
    vision_concurrency: int = typer.Option(
        5,
        "--vision-concurrency",
        min=1,
        help="Number of feedback moments to analyze in parallel.",
    ),
    whisper_model: str | None = typer.Option(None, "--whisper-model", help="Local Whisper model size."),
    transcription: str | None = typer.Option(
        None,
        "--transcription",
        help="Transcription backend: auto, local, or online.",
    ),
    whisper_concurrency: int = typer.Option(
        1,
        "--whisper-concurrency",
        min=1,
        help="Parallel audio chunks for Whisper (CUDA only by default; CPU uses single-pass).",
    ),
    device: str = typer.Option("auto", "--device", help="Whisper device: auto, cpu, or cuda."),
    compute_type: str = typer.Option("auto", "--compute-type", help="Whisper compute type."),
    force: bool = typer.Option(False, "--force", help="Re-run completed stages."),
    keep_work: bool = typer.Option(True, "--keep-work/--no-keep-work", help="Keep intermediate work files."),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging."),
) -> None:
    """Analyze client feedback video(s) and generate implementation prompts."""
    root = Path.cwd()
    try:
        settings = Settings.from_env(
            project_root=root,
            llm_provider=provider,
            model=model,
            vision_model=vision_model,
            synthesis_model=synthesis_model,
            vision_concurrency=vision_concurrency,
            whisper_model=whisper_model,
            transcription_backend=transcription,
            whisper_concurrency=whisper_concurrency,
            device=device,
            compute_type=compute_type,
            verbose=verbose,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        videos = _resolve_videos(video, settings.input_dir)
    except VideoValidationError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not videos:
        console.print("[red]Error:[/red] No supported videos found in input/.")
        raise typer.Exit(code=1)

    if project is not None:
        project = project.resolve()

    _run_videos(
        videos=[video.resolve() for video in videos],
        settings=settings,
        project_path=project,
        force=force,
        keep_work=keep_work,
        verbose=verbose,
    )


@app.command("extract")
def extract_command(
    video: Path | None = typer.Argument(
        None,
        help="Path to a specific video file. If omitted, process all videos in input/.",
        exists=False,
        readable=False,
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help=f"LLM provider ({', '.join(SUPPORTED_LLM_PROVIDERS)}).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Reasoning model for segment detection.",
    ),
    vision_model: str | None = typer.Option(
        None,
        "--vision-model",
        help="Model for screenshot / image analysis.",
    ),
    synthesis_model: str | None = typer.Option(
        None,
        "--synthesis-model",
        help="Model for the final content feature synthesis.",
    ),
    vision_concurrency: int = typer.Option(
        5,
        "--vision-concurrency",
        min=1,
        help="Number of content segments to analyze in parallel.",
    ),
    whisper_model: str | None = typer.Option(None, "--whisper-model", help="Local Whisper model size."),
    transcription: str | None = typer.Option(
        None,
        "--transcription",
        help="Transcription backend: auto, local, or online.",
    ),
    whisper_concurrency: int = typer.Option(
        1,
        "--whisper-concurrency",
        min=1,
        help="Parallel audio chunks for Whisper (CUDA only by default; CPU uses single-pass).",
    ),
    device: str = typer.Option("auto", "--device", help="Whisper device: auto, cpu, or cuda."),
    compute_type: str = typer.Option("auto", "--compute-type", help="Whisper compute type."),
    force: bool = typer.Option(False, "--force", help="Re-run completed stages."),
    keep_work: bool = typer.Option(True, "--keep-work/--no-keep-work", help="Keep intermediate work files."),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging."),
) -> None:
    """Extract video content features and generate a playbook prompt for Claude/ChatGPT."""
    root = Path.cwd()
    try:
        settings = Settings.from_env(
            project_root=root,
            llm_provider=provider,
            model=model,
            vision_model=vision_model,
            synthesis_model=synthesis_model,
            vision_concurrency=vision_concurrency,
            whisper_model=whisper_model,
            transcription_backend=transcription,
            whisper_concurrency=whisper_concurrency,
            device=device,
            compute_type=compute_type,
            verbose=verbose,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        videos = _resolve_videos(video, settings.input_dir)
    except VideoValidationError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not videos:
        console.print("[red]Error:[/red] No supported videos found in input/.")
        raise typer.Exit(code=1)

    _run_content_extraction(
        videos=[video.resolve() for video in videos],
        settings=settings,
        force=force,
        keep_work=keep_work,
        verbose=verbose,
    )


@app.command("start")
def start_command(
    project: Path | None = typer.Option(
        None,
        "--project",
        help="Optional project directory for codebase-aware analysis.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    force: bool = typer.Option(False, "--force", help="Re-run completed stages."),
    keep_work: bool = typer.Option(True, "--keep-work/--no-keep-work", help="Keep intermediate work files."),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging."),
) -> None:
    """Interactive wizard to choose mode, provider, model, transcription, and video."""
    root = Path.cwd()
    try:
        mode, settings, video_path = run_start_wizard(root)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if verbose:
        settings = Settings.from_env(
            project_root=root,
            llm_provider=settings.llm_provider,
            model=settings.effective_reasoning_model(),
            vision_model=settings.effective_vision_model(),
            synthesis_model=settings.effective_synthesis_model(),
            transcription_backend=settings.transcription_backend,
            verbose=True,
        )

    project_path = project.resolve() if project is not None else None
    if mode == "extract" and project_path is not None:
        console.print("[yellow]Note:[/yellow] --project is only used for feedback mode; ignoring.")

    while True:
        try:
            if mode == "extract":
                _extract_single_video(
                    video_path=video_path,
                    settings=settings,
                    force=force,
                    keep_work=keep_work,
                )
            else:
                _analyze_single_video(
                    video_path=video_path,
                    settings=settings,
                    project_path=project_path,
                    force=force,
                    keep_work=keep_work,
                )
        except FFmpegNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except TranscriptionError as exc:
            console.print(f"[red]Transcription error:[/red] {exc}")
            if not _prompt_retry_after_failure():
                raise typer.Exit(code=1) from exc
            video_path = prompt_video_path(root)
            continue
        except LLMProviderError as exc:
            console.print(f"[red]LLM provider error[/red] stage={exc.stage} feedback_id={exc.feedback_id}")
            console.print(str(exc))
            if not _prompt_retry_after_failure():
                raise typer.Exit(code=1) from exc
            video_path = prompt_video_path(root)
            continue
        except VideoValidationError as exc:
            console.print(f"[red]Video error:[/red] {exc}")
            if not _prompt_retry_after_failure():
                raise typer.Exit(code=1) from exc
            video_path = prompt_video_path(root)
            continue
        except Exception as exc:
            console.print(f"[red]Failed for {video_path.name}:[/red] {exc}")
            if verbose:
                console.print_exception()
            if not _prompt_retry_after_failure():
                raise typer.Exit(code=1) from exc
            video_path = prompt_video_path(root)
            continue

        if not prompt_continue():
            break

        video_path = prompt_video_path(root)
        console.print(
            f"\n[bold]Next video[/bold] — "
            f"[cyan]{mode}[/cyan] / [cyan]{settings.llm_provider}[/cyan] / "
            f"[cyan]{settings.effective_synthesis_model() or 'default'}[/cyan]\n"
        )


def _prompt_retry_after_failure() -> bool:
    console.print("")
    response = typer.prompt("Try another video? [Y/n]", default="Y").strip().lower()
    return response in {"", "y", "yes"}


def _run_videos(
    *,
    videos: list[Path],
    settings: Settings,
    project_path: Path | None,
    force: bool,
    keep_work: bool,
    verbose: bool,
) -> None:
    for video_path in videos:
        try:
            _analyze_single_video(
                video_path=video_path,
                settings=settings,
                project_path=project_path,
                force=force,
                keep_work=keep_work,
            )
        except FFmpegNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except TranscriptionError as exc:
            console.print(f"[red]Transcription error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except LLMProviderError as exc:
            console.print(f"[red]LLM provider error[/red] stage={exc.stage} feedback_id={exc.feedback_id}")
            console.print(str(exc))
            raise typer.Exit(code=1) from exc
        except VideoValidationError as exc:
            console.print(f"[red]Video error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            console.print(f"[red]Analysis failed for {video_path.name}:[/red] {exc}")
            if verbose:
                console.print_exception()
            raise typer.Exit(code=1) from exc


def _run_content_extraction(
    *,
    videos: list[Path],
    settings: Settings,
    force: bool,
    keep_work: bool,
    verbose: bool,
) -> None:
    for video_path in videos:
        try:
            _extract_single_video(
                video_path=video_path,
                settings=settings,
                force=force,
                keep_work=keep_work,
            )
        except FFmpegNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except TranscriptionError as exc:
            console.print(f"[red]Transcription error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except LLMProviderError as exc:
            console.print(f"[red]LLM provider error[/red] stage={exc.stage} feedback_id={exc.feedback_id}")
            console.print(str(exc))
            raise typer.Exit(code=1) from exc
        except VideoValidationError as exc:
            console.print(f"[red]Video error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            console.print(f"[red]Extraction failed for {video_path.name}:[/red] {exc}")
            if verbose:
                console.print_exception()
            raise typer.Exit(code=1) from exc


def _extract_single_video(
    *,
    video_path: Path,
    settings: Settings,
    force: bool,
    keep_work: bool,
) -> None:
    console.print("\n[bold]Video2Prompt — Content Extraction[/bold]\n")
    console.print(f"Video: [cyan]{video_path.name}[/cyan]")
    console.print(f"Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(
        f"Vision model: [cyan]{settings.effective_vision_model() or 'provider default'}[/cyan] "
        f"([dim]{settings.vision_concurrency} parallel[/dim])"
    )
    synthesis_model = settings.effective_synthesis_model()
    reasoning_model = settings.effective_reasoning_model()
    console.print(f"Synthesis model: [cyan]{synthesis_model or 'provider default'}[/cyan]")
    if reasoning_model and reasoning_model != synthesis_model:
        console.print(f"Reasoning model: [cyan]{reasoning_model}[/cyan]")
    console.print(
        f"Transcription: [cyan]{settings.resolved_transcription_backend()}[/cyan]"
        + (
            f" ([dim]{settings.transcription_backend}[/dim])"
            if settings.transcription_backend != settings.resolved_transcription_backend()
            else ""
        )
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = content_work_dir_for_video(settings.work_dir, video_path)

    manifest = load_manifest(work_dir)
    metadata = probe_video(video_path, verbose=settings.verbose)
    console.print(f"Duration: [dim]{format_duration(metadata.duration)}[/dim]")

    if manifest is not None and source_changed(manifest, metadata):
        console.print("[yellow]Source video changed. Previous extraction will not be reused.[/yellow]")
        clear_work_cache(work_dir)
        force = True

    cached_stage_lines = describe_content_cached_stages(work_dir) if not force else []
    if cached_stage_lines:
        console.print("\n[bold]Resuming from cached work[/bold]")
        for line in cached_stage_lines:
            console.print(f"  {line}")
        console.print()

    metadata_path = work_dir / "metadata.json"
    if force or not metadata_path.exists():
        save_json_file(metadata_path, metadata)
    console.print("[green]✓[/green] Video validated")

    audio_path = work_dir / "audio.wav"
    if force or not audio_path.exists():
        extract_audio(video_path, audio_path, verbose=settings.verbose, force=True)
        console.print("[green]✓[/green] Audio extracted")
    else:
        console.print("[green]✓[/green] Audio already exists — reuse")

    transcript_cached = load_cached_transcript(work_dir) is not None and not force
    if transcript_cached:
        transcript = load_cached_transcript(work_dir)
        assert transcript is not None
        md_path = work_dir / "transcript.md"
        if not md_path.exists():
            from feedback_analyzer.transcription import render_transcript_markdown

            md_path.write_text(render_transcript_markdown(transcript), encoding="utf-8")
        console.print(
            f"[green]✓[/green] Transcript already exists — reuse "
            f"({len(transcript.segments)} segments, {format_duration(metadata.duration)})"
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Preparing transcription...", total=100)

            def transcription_progress(fraction: float, message: str) -> None:
                progress.update(
                    task_id,
                    completed=max(0.0, min(100.0, fraction * 100)),
                    description=message,
                )

            transcript = transcribe_audio(
                audio_path,
                metadata,
                settings,
                force=force,
                on_progress=transcription_progress,
            )
        console.print(
            f"[green]✓[/green] Transcript generated — "
            f"{len(transcript.segments)} segments, {format_duration(metadata.duration)}"
        )

    llm_provider = create_provider(settings, project_path=Path.cwd())

    segments_cached = (work_dir / "content-segments.json").exists() and not force
    if segments_cached:
        segments_result = detect_content_segments(
            transcript, work_dir, llm_provider, settings, force=force
        )
        console.print(
            f"[green]✓[/green] Content segments already exist — reuse ({len(segments_result.segments)})"
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Detecting content segments...", total=None)
            segments_result = detect_content_segments(
                transcript, work_dir, llm_provider, settings, force=force
            )
        console.print(f"[green]✓[/green] {len(segments_result.segments)} content segments detected")

    segment_count = len(segments_result.segments)
    segment_ids = [segment.id for segment in segments_result.segments]

    frames_root = work_dir / "frames"
    cached_frame_count = count_cached_frames(frames_root, segment_ids) if segment_count else 0
    if segment_count and cached_frame_count == segment_count and not force:
        console.print(f"[green]✓[/green] Screenshots already exist — reuse ({cached_frame_count}/{segment_count})")
    elif segment_count:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Extracting screenshots...", total=segment_count)
            for segment in segments_result.segments:
                extract_frames_for_candidate(
                    video_path,
                    segment,
                    frames_root,
                    duration=metadata.duration,
                    verbose=settings.verbose,
                    force=force,
                )
                progress.advance(task_id)
        console.print(f"[green]✓[/green] Screenshots extracted ({segment_count})")

    cached_visual_count = count_cached_content_visual_analyses(work_dir, segment_ids) if segment_count else 0
    if segment_count and cached_visual_count < segment_count and not force:
        remaining = segment_count - cached_visual_count
        console.print(
            f"[cyan]→[/cyan] Resuming content visual analysis — {cached_visual_count}/{segment_count} complete, "
            f"{remaining} remaining"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting content visual analysis...", total=segment_count)

        def vision_progress(completed: int, total: int, message: str) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=max(total, 1),
                description=message,
            )

        visual_analyses = analyze_content_visual_parallel(
            segments_result.segments,
            transcript,
            frames_root,
            work_dir,
            llm_provider,
            settings,
            force=force,
            on_progress=vision_progress,
        )
    console.print(
        f"[green]✓[/green] Content visual analysis complete — {len(visual_analyses)} segments "
        f"([dim]{settings.effective_vision_model() or settings.llm_provider}[/dim])"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Synthesizing content features...", total=None)
        content_analysis = synthesize_content_features(
            video_name=video_path.name,
            transcript=transcript,
            visual_analyses=visual_analyses,
            work_dir=work_dir,
            provider=llm_provider,
            settings=settings,
            force=force,
        )

    save_manifest(work_dir, metadata, None)

    stem = video_path.stem
    json_output = settings.output_dir / f"{stem}-content-features.json"
    md_output = settings.output_dir / f"{stem}-playbook-prompt.md"
    save_json_file(json_output, content_analysis)
    md_output.write_text(build_playbook_generation_prompt(content_analysis), encoding="utf-8")

    console.print("\n[bold green]Content extraction complete.[/bold green]\n")
    console.print(f"Hooks:                {len(content_analysis.hooks)}")
    console.print(f"Segments:             {len(content_analysis.segments)}")
    console.print(f"\nPlaybook prompt:\n[cyan]{md_output}[/cyan]")
    console.print(f"\nStructured features:\n[cyan]{json_output}[/cyan]")

    if not keep_work:
        clear_work_cache(work_dir)


def _resolve_videos(video: Path | None, input_dir: Path) -> list[Path]:
    if video is not None:
        path = video.resolve()
        if not path.exists():
            raise VideoValidationError(f"Video file not found: {video}")
        if not is_supported_video(path):
            raise VideoValidationError(f"Unsupported video format: {path.suffix}")
        return [path]

    if not input_dir.exists():
        return []
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and is_supported_video(path)
    )


def _analyze_single_video(
    *,
    video_path: Path,
    settings: Settings,
    project_path: Path | None,
    force: bool,
    keep_work: bool,
) -> None:
    console.print("\n[bold]Video2Prompt[/bold]\n")
    console.print(f"Video: [cyan]{video_path.name}[/cyan]")
    if project_path:
        console.print(f"Project: [cyan]{project_path}[/cyan]")
    console.print(f"Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(
        f"Vision model: [cyan]{settings.effective_vision_model() or 'provider default'}[/cyan] "
        f"([dim]{settings.vision_concurrency} parallel[/dim])"
    )
    synthesis_model = settings.effective_synthesis_model()
    reasoning_model = settings.effective_reasoning_model()
    console.print(f"Synthesis model: [cyan]{synthesis_model or 'provider default'}[/cyan]")
    if settings.llm_provider == "subscription":
        console.print("Subscription CLI: [cyan]auto (claude, then codex)[/cyan]")
    if reasoning_model and reasoning_model != synthesis_model:
        console.print(f"Reasoning model: [cyan]{reasoning_model}[/cyan]")
    elif settings.llm_provider == "cursor":
        console.print(f"Reasoning model: [cyan]{reasoning_model or 'provider default'}[/cyan]")
    console.print(
        f"Transcription: [cyan]{settings.resolved_transcription_backend()}[/cyan]"
        + (
            f" ([dim]{settings.transcription_backend}[/dim])"
            if settings.transcription_backend != settings.resolved_transcription_backend()
            else ""
        )
    )
    if settings.resolved_transcription_backend() == "local":
        console.print(
            f"Local Whisper: [cyan]{settings.whisper_model}[/cyan] "
            f"([dim]single-pass on CPU; parallel on CUDA[/dim])"
        )
    else:
        engine = settings.resolve_transcription_engine()
        if engine == "codex":
            console.print("Online transcription: [cyan]Codex subscription[/cyan] (`codex login`)")
        else:
            console.print(f"OpenAI Whisper model: [cyan]{settings.openai_whisper_model}[/cyan]")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir_for_video(settings.work_dir, video_path)

    manifest = load_manifest(work_dir)
    metadata = probe_video(video_path, verbose=settings.verbose)
    console.print(f"Duration: [dim]{format_duration(metadata.duration)}[/dim]")

    if manifest is not None and source_changed(manifest, metadata):
        console.print("[yellow]Source video changed. Previous analysis will not be reused.[/yellow]")
        clear_work_cache(work_dir)
        force = True

    cached_stage_lines = describe_cached_stages(work_dir) if not force else []
    if cached_stage_lines:
        console.print("\n[bold]Resuming from cached work[/bold]")
        for line in cached_stage_lines:
            console.print(f"  {line}")
        console.print()

    project_info: ProjectCacheInfo | None = None
    git_status_before = ""
    if project_path:
        head, git_status_before = capture_project_cache_info(project_path)
        project_info = ProjectCacheInfo(git_head=head, git_status_before=git_status_before)
        if project_cache_stale(manifest, project_path) and not force:
            console.print("[yellow]Project Git HEAD changed — invalidating codebase mappings and final synthesis.[/yellow]")

    metadata_path = work_dir / "metadata.json"
    if force or not metadata_path.exists():
        save_json_file(metadata_path, metadata)
    console.print("[green]✓[/green] Video validated")

    audio_path = work_dir / "audio.wav"
    if force or not audio_path.exists():
        extract_audio(video_path, audio_path, verbose=settings.verbose, force=True)
        console.print("[green]✓[/green] Audio extracted")
    else:
        console.print("[green]✓[/green] Audio already exists — reuse")

    transcript_cached = load_cached_transcript(work_dir) is not None and not force
    if transcript_cached:
        transcript = load_cached_transcript(work_dir)
        assert transcript is not None
        md_path = work_dir / "transcript.md"
        if not md_path.exists():
            from feedback_analyzer.transcription import render_transcript_markdown

            md_path.write_text(render_transcript_markdown(transcript), encoding="utf-8")
        console.print(
            f"[green]✓[/green] Transcript already exists — reuse "
            f"({len(transcript.segments)} segments, {format_duration(metadata.duration)})"
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Preparing transcription...", total=100)

            def transcription_progress(fraction: float, message: str) -> None:
                progress.update(
                    task_id,
                    completed=max(0.0, min(100.0, fraction * 100)),
                    description=message,
                )

            transcript = transcribe_audio(
                audio_path,
                metadata,
                settings,
                force=force,
                on_progress=transcription_progress,
            )
        console.print(
            f"[green]✓[/green] Transcript generated — "
            f"{len(transcript.segments)} segments, {format_duration(metadata.duration)}"
        )

    llm_provider = create_provider(settings, project_path=project_path or Path.cwd())

    invalidate_codebase = project_cache_stale(manifest, project_path)
    candidates_cached = (work_dir / "feedback-candidates.json").exists() and not force
    if candidates_cached:
        candidates_result = detect_feedback_candidates(
            transcript, work_dir, llm_provider, settings, force=force
        )
        console.print(
            f"[green]✓[/green] Feedback candidates already exist — reuse ({len(candidates_result.candidates)})"
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Detecting feedback moments...", total=None)
            candidates_result = detect_feedback_candidates(
            transcript, work_dir, llm_provider, settings, force=force
        )
        console.print(f"[green]✓[/green] {len(candidates_result.candidates)} candidate feedback moments detected")

    candidate_count = len(candidates_result.candidates)
    candidate_ids = [candidate.id for candidate in candidates_result.candidates]

    frames_root = work_dir / "frames"
    cached_frame_count = count_cached_frames(frames_root, candidate_ids) if candidate_count else 0
    if candidate_count and cached_frame_count == candidate_count and not force:
        console.print(f"[green]✓[/green] Screenshots already exist — reuse ({cached_frame_count}/{candidate_count})")
    elif candidate_count:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Extracting screenshots...", total=candidate_count)
            for candidate in candidates_result.candidates:
                extract_frames_for_candidate(
                    video_path,
                    candidate,
                    frames_root,
                    duration=metadata.duration,
                    verbose=settings.verbose,
                    force=force,
                )
                progress.advance(task_id)
        console.print(f"[green]✓[/green] Screenshots extracted ({candidate_count})")

    cached_visual_count = count_cached_visual_analyses(work_dir, candidate_ids) if candidate_count else 0
    if candidate_count and cached_visual_count < candidate_count and not force:
        remaining = candidate_count - cached_visual_count
        console.print(
            f"[cyan]→[/cyan] Resuming visual analysis — {cached_visual_count}/{candidate_count} complete, "
            f"{remaining} remaining"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting vision analysis...", total=candidate_count)

        def vision_progress(completed: int, total: int, message: str) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=max(total, 1),
                description=message,
            )

        visual_analyses = analyze_visual_feedback_parallel(
            candidates_result.candidates,
            transcript,
            frames_root,
            work_dir,
            llm_provider,
            settings,
            force=force,
            on_progress=vision_progress,
        )
    console.print(
        f"[green]✓[/green] Visual analysis complete — {len(visual_analyses)} items "
        f"([dim]{settings.effective_vision_model() or settings.llm_provider}[/dim])"
    )

    codebase_mappings: dict[str, CodebaseMapping] = {}
    if project_path:
        codebase_force = force or invalidate_codebase
        for analysis in visual_analyses:
            mapping = map_feedback_to_codebase(
                analysis,
                work_dir,
                project_path,
                llm_provider,
                settings,
                git_status_before=git_status_before,
                force=codebase_force,
            )
            codebase_mappings[analysis.id] = mapping
        console.print("[green]✓[/green] Codebase mapping complete")

    synthesis_force = force or invalidate_codebase
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Synthesizing final feedback...", total=None)
        final_analysis = synthesize_feedback(
            video_name=video_path.name,
            project_path=project_path,
            transcript=transcript,
            visual_analyses=visual_analyses,
            codebase_mappings=codebase_mappings,
            work_dir=work_dir,
            provider=llm_provider,
            settings=settings,
            force=synthesis_force,
        )

    save_manifest(work_dir, metadata, project_info)

    stem = video_path.stem
    json_output = settings.output_dir / f"{stem}-feedback.json"
    md_output = settings.output_dir / f"{stem}-implementation-prompt.md"
    save_json_file(json_output, final_analysis)
    md_output.write_text(build_implementation_prompt(final_analysis), encoding="utf-8")

    superseded = sum(1 for item in final_analysis.items if item.superseded_feedback)
    unclear = len(final_analysis.unclear_items)

    console.print("\n[bold green]Analysis complete.[/bold green]\n")
    console.print(f"Feedback items:       {len(final_analysis.items)}")
    console.print(f"Superseded items:     {superseded}")
    console.print(f"Needs clarification:  {unclear}")
    console.print(f"\nPrompt:\n[cyan]{md_output}[/cyan]")
    console.print(f"\nStructured analysis:\n[cyan]{json_output}[/cyan]")

    if not keep_work:
        clear_work_cache(work_dir)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
