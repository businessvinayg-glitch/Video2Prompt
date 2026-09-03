from __future__ import annotations

import json
import os
import re
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

CHATGPT_TRANSCRIBE_URL = "https://chatgpt.com/backend-api/transcribe"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def codex_auth_available(codex_home: Path | None = None) -> bool:
    return load_codex_auth(codex_home) is not None


def load_codex_auth(codex_home: Path | None = None) -> dict | None:
    home = codex_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth_path = home / "auth.json"
    if not auth_path.exists():
        return None
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("auth_mode") != "chatgpt":
        return None
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return None
    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not access_token or not account_id:
        return None
    return {"access_token": access_token, "account_id": account_id}


def transcribe_audio_codex(
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

    auth = load_codex_auth()
    if auth is None:
        raise TranscriptionError(
            "Codex transcription requires a ChatGPT login.\n\n"
            "Run `codex login` and try again, or choose local transcription."
        )

    if on_progress:
        on_progress(0.15, "Uploading audio via Codex subscription...")

    response_body = _call_codex_transcribe(audio_path, auth)
    transcript = _parse_codex_response(response_body, metadata.duration)

    if on_progress:
        on_progress(1.0, "Codex transcription complete — saving transcript")

    save_json_file(json_path, transcript)
    md_path.write_text(render_transcript_markdown(transcript), encoding="utf-8")
    return transcript


def _call_codex_transcribe(audio_path: Path, auth: dict) -> dict:
    audio_bytes = audio_path.read_bytes()
    boundary = uuid.uuid4().hex
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(audio_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = Request(
        CHATGPT_TRANSCRIBE_URL,
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {auth['access_token']}",
            "ChatGPT-Account-Id": auth["account_id"],
            "originator": "Codex Desktop",
            "User-Agent": "Video2Prompt",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code in {401, 403}:
            raise TranscriptionError(
                "Codex transcription auth failed. Run `codex login` and try again."
            ) from exc
        raise TranscriptionError(
            f"Codex transcription failed (HTTP {exc.code}): {details}"
        ) from exc
    except URLError as exc:
        raise TranscriptionError(f"Codex transcription request failed: {exc}") from exc


def _parse_codex_response(body: dict, duration: float) -> Transcript:
    text = str(body.get("text", "")).strip()
    if not text:
        raise TranscriptionError("Codex transcription returned empty text.")

    segments = _segment_text_evenly(text, duration)
    return Transcript(segments=segments, language=None, duration=duration)


def _segment_text_evenly(text: str, duration: float) -> list[TranscriptSegment]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        return [TranscriptSegment(start=0.0, end=duration, text=text)]

    total_chars = sum(len(sentence) for sentence in sentences)
    if total_chars == 0:
        return [TranscriptSegment(start=0.0, end=duration, text=text)]

    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for sentence in sentences:
        share = len(sentence) / total_chars
        span = max(0.5, duration * share)
        end = min(duration, cursor + span)
        segments.append(TranscriptSegment(start=cursor, end=end, text=sentence))
        cursor = end
    if segments:
        segments[-1] = segments[-1].model_copy(update={"end": duration})
    return segments
