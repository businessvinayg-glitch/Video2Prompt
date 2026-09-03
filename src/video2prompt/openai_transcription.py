from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from video2prompt.schemas import Transcript, TranscriptSegment
from video2prompt.transcription import ProgressCallback, TranscriptionError, render_transcript_markdown
from video2prompt.utils import save_json_file

if TYPE_CHECKING:
    from video2prompt.config import Settings

OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_WHISPER_MAX_BYTES = 25 * 1024 * 1024


def transcribe_audio_openai(
    audio_path: Path,
    metadata,
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

    if not settings.openai_api_key:
        raise TranscriptionError(
            "OpenAI transcription requires OPENAI_API_KEY.\n\n"
            "Add your API key to .env or choose local transcription instead."
        )

    file_size = audio_path.stat().st_size
    if file_size > OPENAI_WHISPER_MAX_BYTES:
        raise TranscriptionError(
            f"Audio file is {file_size / (1024 * 1024):.1f} MB, which exceeds OpenAI Whisper's 25 MB limit.\n"
            "Use local transcription or shorten the recording."
        )

    if on_progress:
        on_progress(0.15, "Uploading audio to OpenAI Whisper API...")

    response_body = _call_openai_whisper(audio_path, settings)
    transcript = _parse_openai_verbose_response(response_body, metadata.duration)

    if on_progress:
        on_progress(1.0, "OpenAI transcription complete — saving transcript")

    save_json_file(json_path, transcript)
    md_path.write_text(render_transcript_markdown(transcript), encoding="utf-8")
    return transcript


def _call_openai_whisper(audio_path: Path, settings: Settings) -> dict:
    audio_bytes = audio_path.read_bytes()
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/wav"
    body, content_type = _encode_multipart(
        fields={
            "model": settings.openai_whisper_model,
            "response_format": "verbose_json",
        },
        files={"file": (audio_path.name, audio_bytes, mime_type)},
    )
    request = Request(
        OPENAI_TRANSCRIPTIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise TranscriptionError(
            f"OpenAI Whisper API failed (HTTP {exc.code}): {details}"
        ) from exc
    except URLError as exc:
        raise TranscriptionError(f"OpenAI Whisper API request failed: {exc}") from exc


def _parse_openai_verbose_response(body: dict, duration: float) -> Transcript:
    segments: list[TranscriptSegment] = []
    for item in body.get("segments", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=text,
            )
        )

    if not segments:
        full_text = str(body.get("text", "")).strip()
        if full_text:
            segments = [TranscriptSegment(start=0.0, end=duration, text=full_text)]

    language = body.get("language")
    if isinstance(language, str):
        language = language.strip() or None
    else:
        language = None

    return Transcript(
        segments=segments,
        language=language,
        duration=float(body.get("duration", duration)),
    )


def _encode_multipart(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())
    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"
