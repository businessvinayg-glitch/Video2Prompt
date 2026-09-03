from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from feedback_analyzer.providers.errors import LLMProviderError

DEFAULT_CLI_TIMEOUT_SECONDS = 900


def resolve_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise LLMProviderError(
            f"Required CLI '{name}' was not found on PATH.\n\n"
            f"Install {name} and ensure it is available before using this provider.",
            retryable=False,
        )
    return path


def run_cli(
    command: list[str],
    *,
    stage: str = "",
    feedback_id: str | None = None,
    input_text: str | None = None,
    cwd: Path | None = None,
    timeout_seconds: int = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> str:
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMProviderError(
            f"CLI timed out during stage '{stage}' after {timeout_seconds}s.",
            stage=stage,
            feedback_id=feedback_id,
            retryable=True,
        ) from exc
    except OSError as exc:
        raise LLMProviderError(
            f"Failed to run CLI for stage '{stage}': {exc}",
            stage=stage,
            feedback_id=feedback_id,
            retryable=False,
        ) from exc

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise LLMProviderError(
            f"CLI exited with code {completed.returncode} during stage '{stage}'."
            + (f"\n\n{details}" if details else ""),
            stage=stage,
            feedback_id=feedback_id,
            retryable=True,
        )

    output = (completed.stdout or "").strip()
    if not output:
        stderr = (completed.stderr or "").strip()
        raise LLMProviderError(
            f"CLI returned empty output for stage '{stage}'."
            + (f"\n\n{stderr}" if stderr else ""),
            stage=stage,
            feedback_id=feedback_id,
            retryable=True,
        )
    return output


def append_image_paths(prompt: str, images: list[Path]) -> str:
    if not images:
        return prompt
    lines = [prompt.rstrip(), "", "Attached screenshots:"]
    lines.extend(f"- {path.resolve()}" for path in images)
    return "\n".join(lines)


def parse_claude_json_output(raw_output: str) -> str:
    payload = json.loads(raw_output)
    if isinstance(payload, dict):
        structured = payload.get("structured_output")
        if structured is not None:
            return json.dumps(structured)
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()
    return raw_output.strip()
