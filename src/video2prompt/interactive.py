from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from video2prompt.config import (
    DEFAULT_CURSOR_MODEL,
    DEFAULT_OPENAI_MODEL,
    LLMProviderName,
    Settings,
    SUPPORTED_LLM_PROVIDERS,
    TranscriptionBackendName,
)
from video2prompt.utils import is_supported_video

console = Console()

AnalysisMode = Literal["feedback", "extract"]

MODE_LABELS: dict[AnalysisMode, str] = {
    "feedback": "Client feedback → implementation prompt (for coding agents)",
    "extract": "Content extraction → playbook prompt (for Claude/ChatGPT)",
}

PROVIDER_LABELS: dict[LLMProviderName, str] = {
    "cursor": "Cursor API (subscription / API key)",
    "claude": "Claude Code CLI (Claude subscription)",
    "codex": "Codex CLI (ChatGPT / Codex subscription)",
    "openai": "OpenAI API (pay-as-you-go)",
    "subscription": "Auto-pick subscription CLI (claude, then codex)",
}

PROVIDER_MODEL_SUGGESTIONS: dict[LLMProviderName, list[str]] = {
    "cursor": ["gpt-5.4-mini"],
    "claude": ["opus", "sonnet"],
    "codex": ["gpt-5.4-mini", "gpt-5.5"],
    "openai": ["gpt-5.4-mini", "gpt-5.5"],
    "subscription": [],
}

PROVIDER_MODEL_LABELS: dict[str, str] = {
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.5": "GPT-5.5",
    "opus": "Claude Opus (latest)",
    "sonnet": "Claude Sonnet (latest)",
}


def run_initial_setup(root: Path) -> tuple[AnalysisMode, Settings]:
    console.print("\n[bold]Video2Prompt[/bold] — interactive setup\n")

    mode = _prompt_mode()
    provider = _prompt_provider()
    vision_model, synthesis_model = _prompt_models(provider, mode)
    transcription_backend = _prompt_transcription_backend(provider, root)

    settings = Settings.from_env(
        project_root=root,
        llm_provider=provider,
        model=synthesis_model,
        vision_model=vision_model,
        synthesis_model=synthesis_model,
        transcription_backend=transcription_backend,
    )
    return mode, settings


def run_start_wizard(root: Path) -> tuple[AnalysisMode, Settings, Path]:
    mode, settings = run_initial_setup(root)
    video_path = prompt_video_path(root)
    print_run_summary(mode, settings, video_path)
    return mode, settings, video_path


def prompt_video_path(root: Path) -> Path:
    return _prompt_video_path(root)


def prompt_continue() -> bool:
    console.print("")
    response = typer.prompt("Process another video? [Y/n]", default="Y").strip().lower()
    return response in {"", "y", "yes"}


def print_run_summary(mode: AnalysisMode, settings: Settings, video_path: Path) -> None:
    mode_label = "feedback analysis" if mode == "feedback" else "content extraction"
    console.print(f"\n[bold]Starting {mode_label}[/bold]")
    console.print(f"Mode: [cyan]{mode}[/cyan]")
    console.print(f"Video: [cyan]{video_path}[/cyan]")
    console.print(f"Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(f"Vision model: [cyan]{settings.effective_vision_model()}[/cyan]")
    console.print(f"Synthesis model: [cyan]{settings.effective_synthesis_model()}[/cyan]")
    console.print(
        f"Transcription: [cyan]{settings.resolved_transcription_backend()}[/cyan]"
        + (
            f" ([dim]{settings.transcription_backend}[/dim])"
            if settings.transcription_backend != settings.resolved_transcription_backend()
            else ""
        )
        + "\n"
    )


def _prompt_mode() -> AnalysisMode:
    choices: list[tuple[AnalysisMode, str]] = [
        ("feedback", MODE_LABELS["feedback"]),
        ("extract", MODE_LABELS["extract"]),
    ]
    return _prompt_choice("Choose analysis mode", choices, default_index=1)  # type: ignore[return-value]


def _prompt_provider() -> LLMProviderName:
    choices = [(name, PROVIDER_LABELS[name]) for name in SUPPORTED_LLM_PROVIDERS]
    return _prompt_choice("Choose an LLM provider", choices, default_index=1)


def _prompt_models(provider: LLMProviderName, mode: AnalysisMode) -> tuple[str, str]:
    suggestions = PROVIDER_MODEL_SUGGESTIONS.get(provider, [])
    if provider == "subscription":
        return "", ""
    if provider == "cursor":
        default = DEFAULT_CURSOR_MODEL
    elif provider == "openai":
        default = DEFAULT_OPENAI_MODEL
    else:
        default = suggestions[0] if suggestions else ""

    if suggestions:
        console.print("\n[bold]Suggested models[/bold]")
        for index, suggestion in enumerate(suggestions, 1):
            label = PROVIDER_MODEL_LABELS.get(suggestion, suggestion)
            console.print(f"  {index}. [cyan]{suggestion}[/cyan] — {label}")
        console.print("  Or enter a custom model name.\n")

    vision_model = _resolve_model_choice(
        typer.prompt("Vision model (image analysis)", default=default).strip(),
        suggestions,
        default,
    )
    synthesis_label = (
        "Synthesis model (final implementation prompt)"
        if mode == "feedback"
        else "Synthesis model (final playbook prompt)"
    )
    synthesis_model = _resolve_model_choice(
        typer.prompt(synthesis_label, default=default).strip(),
        suggestions,
        default,
    )
    return vision_model, synthesis_model


def _prompt_transcription_backend(
    provider: LLMProviderName,
    root: Path,
) -> TranscriptionBackendName:
    settings = Settings.from_env(project_root=root, llm_provider=provider)
    has_online = settings.online_transcription_available()
    auto_resolves_to = "online" if provider in {"openai", "codex"} and has_online else "local"
    online_note = "Codex subscription (`codex login`) or OpenAI API key"
    if not has_online:
        online_note += " — not configured"

    choices: list[tuple[TranscriptionBackendName, str]] = [
        ("local", "Local faster-whisper (runs on your machine)"),
        ("online", f"Cloud transcription ({online_note})"),
        (
            "auto",
            f"Auto — online for openai/codex when available, else local (→ {auto_resolves_to})",
        ),
    ]

    return _prompt_choice("Choose transcription backend", choices, default_index=3)


def _resolve_model_choice(raw: str, suggestions: list[str], default: str) -> str:
    if raw.isdigit() and suggestions:
        index = int(raw)
        if 1 <= index <= len(suggestions):
            return suggestions[index - 1]
    return raw or default


def _prompt_video_path(root: Path) -> Path:
    input_dir = root / "input"
    candidates: list[Path] = []
    if input_dir.exists():
        candidates = sorted(
            path for path in input_dir.iterdir() if path.is_file() and is_supported_video(path)
        )

    if candidates:
        console.print("\n[bold]Videos in input/[/bold]")
        for index, path in enumerate(candidates, 1):
            console.print(f"  {index}. [cyan]{path.name}[/cyan]")
        console.print("  Or enter a full path.\n")
        raw = typer.prompt("Video number or path", default="1").strip()
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1].resolve()

    while True:
        raw = typer.prompt("Video file path").strip()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            console.print("[red]File not found. Try again.[/red]")
            continue
        if not is_supported_video(path):
            console.print("[red]Unsupported video format. Use mp4, mov, mkv, webm, or m4v.[/red]")
            continue
        return path


def _prompt_choice(
    label: str,
    choices: list[tuple[str, str]],
    *,
    default_index: int = 1,
) -> str:
    console.print(f"\n[bold]{label}[/bold]")
    for index, (value, description) in enumerate(choices, 1):
        console.print(f"  {index}. [cyan]{value}[/cyan] — {description}")

    while True:
        raw = typer.prompt("Enter number", default=str(default_index)).strip()
        if not raw.isdigit():
            console.print("[red]Enter a number from the list.[/red]")
            continue
        choice = int(raw)
        if 1 <= choice <= len(choices):
            return choices[choice - 1][0]
        console.print("[red]Invalid choice. Try again.[/red]")
