from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_range(start: float, end: float) -> str:
    return f"{format_timestamp(start)}–{format_timestamp(end)}"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def frame_filename(seconds: float) -> str:
    return f"{format_timestamp(seconds).replace(':', '-')}.jpg"


def expand_window(
    start: float,
    end: float,
    *,
    duration: float,
    before: float = 3.0,
    after: float = 4.0,
) -> tuple[float, float]:
    expanded_start = max(0.0, start - before)
    expanded_end = min(duration, end + after)
    if expanded_end < expanded_start:
        expanded_end = expanded_start
    return expanded_start, expanded_end


def frame_timestamps_for_window(start: float, end: float) -> list[float]:
    if end <= start:
        return [start]
    span = end - start
    points = [start, start + span * 0.25, start + span * 0.5, start + span * 0.75, end]
    unique: list[float] = []
    for point in points:
        rounded = round(point, 3)
        if not unique or abs(unique[-1] - rounded) > 0.05:
            unique.append(rounded)
    return unique


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = _JSON_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner
    return stripped


def parse_json_text(text: str) -> Any:
    cleaned = strip_json_fence(text)
    return json.loads(cleaned)


def validate_model_json(text: str, model: type[T]) -> T:
    data = parse_json_text(text)
    return model.model_validate(data)


def load_json_file(path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def save_json_file(path: Path, data: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        payload = data.model_dump()
    else:
        payload = data
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str], *, verbose: bool = False) -> subprocess.CompletedProcess[str]:
    if verbose:
        import shlex

        print(f"  $ {shlex.join(command)}")
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def git_porcelain_status(project_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_head(project_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_supported_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def prompts_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "prompts",
        Path.cwd() / "prompts",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Prompt templates not found. Expected a `prompts/` directory at the project root."
    )


def load_prompt(name: str, **replacements: str) -> str:
    path = prompts_dir() / name
    text = path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def validation_error_summary(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in error.errors())


def codex_compatible_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pydantic JSON Schema for Codex CLI structured output."""

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return

        for key in ("$defs", "definitions"):
            defs = node.get(key)
            if isinstance(defs, dict):
                for definition in defs.values():
                    visit(definition)

        for key in ("allOf", "anyOf", "oneOf"):
            variants = node.get(key)
            if isinstance(variants, list):
                for variant in variants:
                    visit(variant)

        properties = node.get("properties")
        if isinstance(properties, dict):
            for property_schema in properties.values():
                visit(property_schema)

        items = node.get("items")
        if isinstance(items, dict):
            visit(items)
        elif isinstance(items, list):
            for item_schema in items:
                visit(item_schema)

        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            visit(additional)

        if node.get("type") == "object" or isinstance(properties, dict):
            node["additionalProperties"] = False
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())

    result = json.loads(json.dumps(schema))
    visit(result)
    return result
