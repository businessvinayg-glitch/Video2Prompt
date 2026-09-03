from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import os

load_dotenv()

LLMProviderName = Literal["cursor", "claude", "codex", "openai", "subscription"]
TranscriptionBackendName = Literal["auto", "local", "online"]
SUPPORTED_LLM_PROVIDERS: tuple[LLMProviderName, ...] = (
    "cursor",
    "claude",
    "codex",
    "openai",
    "subscription",
)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

WINDOW_EXPAND_BEFORE_SECONDS = 3.0
WINDOW_EXPAND_AFTER_SECONDS = 4.0

MAX_IMAGES_PER_VISION_REQUEST = 5
DEFAULT_VISION_CONCURRENCY = 5
DEFAULT_CURSOR_MODEL = "gpt-5.4-mini"
DEFAULT_CURSOR_VISION_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_WHISPER_MODEL = "whisper-1"
DEFAULT_WHISPER_CONCURRENCY = 1
MIN_DURATION_FOR_PARALLEL_TRANSCRIPTION = 90.0
TRANSCRIPTION_CHUNK_OVERLAP_SECONDS = 2.0
JPEG_QUALITY = 3  # FFmpeg -q:v scale 2-31, lower is better; 3 is high quality


@dataclass
class Settings:
    llm_provider: LLMProviderName
    cursor_api_key: str
    cursor_model: str
    cursor_vision_model: str
    cursor_synthesis_model: str
    openai_api_key: str
    openai_model: str
    openai_vision_model: str
    openai_synthesis_model: str
    claude_cli: str
    claude_model: str
    claude_vision_model: str
    claude_synthesis_model: str
    codex_cli: str
    codex_model: str
    codex_vision_model: str
    codex_synthesis_model: str
    codex_reasoning_effort: str
    vision_concurrency: int
    transcription_backend: TranscriptionBackendName
    openai_whisper_model: str
    whisper_model: str
    whisper_concurrency: int
    whisper_device: Literal["auto", "cpu", "cuda"]
    whisper_compute_type: Literal["auto", "int8", "float16", "float32"]
    input_dir: Path
    output_dir: Path
    work_dir: Path
    verbose: bool = False

    def _model_provider(self) -> LLMProviderName:
        if self.llm_provider != "subscription":
            return self.llm_provider

        import shutil

        if shutil.which(self.claude_cli):
            return "claude"
        if shutil.which(self.codex_cli):
            return "codex"
        return "codex"

    def effective_reasoning_model(self) -> str | None:
        provider = self._model_provider()
        if provider == "cursor":
            return self.cursor_model
        if provider == "claude":
            return self.claude_model or None
        if provider == "codex":
            return self.codex_model or None
        if provider == "openai":
            return self.openai_model
        return None

    def effective_vision_model(self) -> str | None:
        provider = self._model_provider()
        if provider == "cursor":
            return self.cursor_vision_model
        if provider == "claude":
            return self.claude_vision_model or self.claude_model or None
        if provider == "codex":
            return self.codex_vision_model or self.codex_model or None
        if provider == "openai":
            return self.openai_vision_model or self.openai_model
        return None

    def effective_synthesis_model(self) -> str | None:
        provider = self._model_provider()
        if provider == "cursor":
            return self.cursor_synthesis_model or self.cursor_model
        if provider == "claude":
            return self.claude_synthesis_model or self.claude_model or None
        if provider == "codex":
            return self.codex_synthesis_model or self.codex_model or None
        if provider == "openai":
            return self.openai_synthesis_model or self.openai_model
        return None

    def resolve_transcription_engine(self) -> Literal["local", "codex", "openai"]:
        from video2prompt.codex_transcription import codex_auth_available

        if self.transcription_backend == "local":
            return "local"

        wants_online = self.transcription_backend == "online" or (
            self.transcription_backend == "auto"
            and self.llm_provider in {"openai", "codex"}
            and (self.openai_api_key or codex_auth_available())
        )
        if not wants_online:
            return "local"

        if self.llm_provider == "codex" and codex_auth_available():
            return "codex"
        if self.openai_api_key:
            return "openai"
        if codex_auth_available():
            return "codex"
        return "openai"

    def resolved_transcription_backend(self) -> Literal["local", "online"]:
        return "online" if self.resolve_transcription_engine() != "local" else "local"

    def online_transcription_available(self) -> bool:
        from video2prompt.codex_transcription import codex_auth_available

        return bool(self.openai_api_key) or codex_auth_available()

    def openai_transcription_available(self) -> bool:
        return bool(self.openai_api_key)

    def ensure_directories(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
        llm_provider: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        synthesis_model: str | None = None,
        cursor_model: str | None = None,
        cursor_vision_model: str | None = None,
        openai_model: str | None = None,
        claude_cli: str | None = None,
        claude_model: str | None = None,
        codex_cli: str | None = None,
        codex_model: str | None = None,
        codex_reasoning_effort: str | None = None,
        transcription_backend: str | None = None,
        openai_whisper_model: str | None = None,
        vision_concurrency: int | None = None,
        whisper_model: str | None = None,
        whisper_concurrency: int | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        verbose: bool = False,
    ) -> Settings:
        root = project_root or Path.cwd()
        api_key = os.environ.get("CURSOR_API_KEY", "").strip()
        concurrency_raw = vision_concurrency or os.environ.get("VISION_CONCURRENCY", str(DEFAULT_VISION_CONCURRENCY))
        whisper_concurrency_raw = whisper_concurrency or os.environ.get(
            "WHISPER_CONCURRENCY", str(DEFAULT_WHISPER_CONCURRENCY)
        )
        provider_raw = llm_provider or os.environ.get("LLM_PROVIDER", "cursor")
        provider = _parse_llm_provider(provider_raw)
        reasoning_override = model
        cursor_models = _resolve_provider_models(
            reasoning_override=reasoning_override or cursor_model,
            vision_override=vision_model or cursor_vision_model,
            synthesis_override=synthesis_model,
            env_reasoning="CURSOR_MODEL",
            env_vision="CURSOR_VISION_MODEL",
            env_synthesis="CURSOR_SYNTHESIS_MODEL",
            default=DEFAULT_CURSOR_MODEL,
        )
        openai_models = _resolve_provider_models(
            reasoning_override=reasoning_override or openai_model,
            vision_override=vision_model,
            synthesis_override=synthesis_model,
            env_reasoning="OPENAI_MODEL",
            env_vision="OPENAI_VISION_MODEL",
            env_synthesis="OPENAI_SYNTHESIS_MODEL",
            default=DEFAULT_OPENAI_MODEL,
        )
        claude_models = _resolve_provider_models(
            reasoning_override=reasoning_override or claude_model,
            vision_override=vision_model,
            synthesis_override=synthesis_model,
            env_reasoning="CLAUDE_MODEL",
            env_vision="CLAUDE_VISION_MODEL",
            env_synthesis="CLAUDE_SYNTHESIS_MODEL",
            default="",
        )
        codex_models = _resolve_provider_models(
            reasoning_override=reasoning_override or codex_model,
            vision_override=vision_model,
            synthesis_override=synthesis_model,
            env_reasoning="CODEX_MODEL",
            env_vision="CODEX_VISION_MODEL",
            env_synthesis="CODEX_SYNTHESIS_MODEL",
            default="",
        )
        settings = cls(
            llm_provider=provider,
            cursor_api_key=api_key,
            cursor_model=cursor_models.reasoning,
            cursor_vision_model=cursor_models.vision,
            cursor_synthesis_model=cursor_models.synthesis,
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            openai_model=openai_models.reasoning,
            openai_vision_model=openai_models.vision,
            openai_synthesis_model=openai_models.synthesis,
            claude_cli=claude_cli or os.environ.get("CLAUDE_CLI", "claude"),
            claude_model=claude_models.reasoning,
            claude_vision_model=claude_models.vision,
            claude_synthesis_model=claude_models.synthesis,
            codex_cli=codex_cli or os.environ.get("CODEX_CLI", "codex"),
            codex_model=codex_models.reasoning,
            codex_vision_model=codex_models.vision,
            codex_synthesis_model=codex_models.synthesis,
            codex_reasoning_effort=codex_reasoning_effort
            or os.environ.get("CODEX_REASONING_EFFORT", "medium"),
            transcription_backend=_parse_transcription_backend(
                transcription_backend or os.environ.get("TRANSCRIPTION_BACKEND", "auto")
            ),
            openai_whisper_model=openai_whisper_model
            or os.environ.get("OPENAI_WHISPER_MODEL", DEFAULT_OPENAI_WHISPER_MODEL),
            vision_concurrency=max(1, int(concurrency_raw)),
            whisper_model=whisper_model or os.environ.get("WHISPER_MODEL", "medium"),
            whisper_concurrency=max(1, int(whisper_concurrency_raw)),
            whisper_device=_parse_device(device or os.environ.get("WHISPER_DEVICE", "auto")),
            whisper_compute_type=_parse_compute_type(
                compute_type or os.environ.get("WHISPER_COMPUTE_TYPE", "auto")
            ),
            input_dir=Path(os.environ.get("INPUT_DIR", root / "input")),
            output_dir=Path(os.environ.get("OUTPUT_DIR", root / "output")),
            work_dir=Path(os.environ.get("WORK_DIR", root / "work")),
            verbose=verbose,
        )
        settings.ensure_directories()
        return settings


def _parse_transcription_backend(value: str) -> TranscriptionBackendName:
    normalized = value.lower().strip()
    if normalized == "openai":
        normalized = "online"
    if normalized in {"auto", "local", "online"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Invalid transcription backend: {value}. Use auto, local, or online.")


def _parse_llm_provider(value: str) -> LLMProviderName:
    normalized = value.lower().strip()
    if normalized in SUPPORTED_LLM_PROVIDERS:
        return normalized  # type: ignore[return-value]
    supported = ", ".join(SUPPORTED_LLM_PROVIDERS)
    raise ValueError(f"Invalid LLM provider: {value}. Use one of: {supported}.")


def _parse_device(value: str) -> Literal["auto", "cpu", "cuda"]:
    normalized = value.lower().strip()
    if normalized in {"auto", "cpu", "cuda"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Invalid device: {value}. Use auto, cpu, or cuda.")


def _parse_compute_type(value: str) -> Literal["auto", "int8", "float16", "float32"]:
    normalized = value.lower().strip()
    if normalized in {"auto", "int8", "float16", "float32"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Invalid compute type: {value}.")


def parallel_whisper_on_cpu_enabled() -> bool:
    return os.environ.get("WHISPER_PARALLEL_CPU", "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ProviderModels:
    reasoning: str
    vision: str
    synthesis: str


def _resolve_provider_models(
    *,
    reasoning_override: str | None,
    vision_override: str | None,
    synthesis_override: str | None,
    env_reasoning: str,
    env_vision: str,
    env_synthesis: str,
    default: str,
) -> ProviderModels:
    env_reasoning_value = os.environ.get(env_reasoning, "").strip()
    env_vision_value = os.environ.get(env_vision, "").strip()
    env_synthesis_value = os.environ.get(env_synthesis, "").strip()

    reasoning = reasoning_override or env_reasoning_value or default
    synthesis = synthesis_override or env_synthesis_value or reasoning
    vision = vision_override or env_vision_value or reasoning
    return ProviderModels(reasoning=reasoning, vision=vision, synthesis=synthesis)
