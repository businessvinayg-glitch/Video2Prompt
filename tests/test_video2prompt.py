from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from video2prompt.cache import load_manifest, save_manifest, source_changed, work_dir_for_video
from video2prompt.cli import app
from video2prompt.config import Settings
from video2prompt.content_prompt_builder import build_playbook_generation_prompt
from video2prompt.content_schemas import (
    ContentFeatureAnalysis,
    FinalContentSegment,
    HookFeature,
)
from video2prompt.prompt_builder import build_implementation_prompt
from video2prompt.schemas import (
    FeedbackAnalysis,
    FeedbackCandidatesResult,
    FinalFeedbackItem,
    ProjectCacheInfo,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
    VisualFeedbackAnalysis,
)
from video2prompt.transcription import (
    merge_transcript_segments,
    plan_audio_chunks,
    plan_parallel_chunks,
    render_transcript_markdown,
)
from video2prompt.transcription import AudioChunk
from video2prompt.utils import (
    codex_compatible_json_schema,
    expand_window,
    format_timestamp,
    frame_timestamps_for_window,
    is_supported_video,
    parse_json_text,
    strip_json_fence,
    validate_model_json,
)
from video2prompt.providers.cli_runner import append_image_paths, parse_claude_json_output
from video2prompt.providers.factory import create_provider
from video2prompt.video import FFmpegNotFoundError


runner = CliRunner()


def test_supported_video_extensions():
    assert is_supported_video(Path("clip.mp4"))
    assert is_supported_video(Path("clip.MOV"))
    assert not is_supported_video(Path("clip.avi"))


def test_format_timestamp():
    assert format_timestamp(32.4) == "00:32.400"
    assert format_timestamp(3723.5) == "01:02:03.500"


def test_expand_window_clamps_to_bounds():
    start, end = expand_window(2.0, 5.0, duration=10.0, before=3.0, after=4.0)
    assert start == 0.0
    assert end == 9.0


def test_frame_timestamps_for_window():
    points = frame_timestamps_for_window(10.0, 20.0)
    assert points[0] == 10.0
    assert points[-1] == 20.0
    assert len(points) == 5


def test_plan_parallel_chunks_short_audio_uses_single_pass():
    assert plan_parallel_chunks(45.0, 5, device="cuda") == 1


def test_plan_parallel_chunks_long_audio_uses_parallelism_on_cuda():
    assert plan_parallel_chunks(516.0, 5, device="cuda") == 5


def test_plan_parallel_chunks_cpu_defaults_to_single_pass():
    assert plan_parallel_chunks(516.0, 5, device="cpu") == 1


def test_merge_transcript_segments_offsets_and_dedupes():
    chunks = plan_audio_chunks(100.0, 2)
    merged = merge_transcript_segments(
        [
            (
                chunks[0],
                [TranscriptSegment(start=1.0, end=2.0, text="hello world")],
            ),
            (
                chunks[1],
                [TranscriptSegment(start=0.5, end=1.5, text="hello world")],
            ),
        ]
    )
    assert len(merged) == 1
    assert merged[0].text == "hello world"


def test_transcript_markdown_rendering():
    transcript = Transcript(
        segments=[
            TranscriptSegment(start=32.4, end=37.8, text="Move this underneath the results section."),
            TranscriptSegment(start=39.0, end=46.0, text="Remove this button."),
        ]
    )
    rendered = render_transcript_markdown(transcript)
    assert "[00:32.400 - 00:37.800]" in rendered
    assert "Move this underneath the results section." in rendered


def test_strip_json_fence_raw_and_fenced():
    raw = '{"ok": true}'
    assert strip_json_fence(raw) == raw
    fenced = '```json\n{"ok": true}\n```'
    assert parse_json_text(fenced) == {"ok": True}


def test_parse_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json_text("not json")


def test_validate_model_json_retry_behavior():
  class Sample(Transcript):
      pass

  with pytest.raises(ValidationError):
      validate_model_json('{"segments": "bad"}', Transcript)


def test_prompt_builder_renders_sections():
    analysis = FeedbackAnalysis(
        video="client-review.mp4",
        project="/tmp/project",
        summary="Two UI updates requested.",
        global_constraints=["Do not modify scraping behavior."],
        items=[
            FinalFeedbackItem(
                id="FB-001",
                title="Rename processing button",
                category="ui",
                timestamp_start=218.0,
                timestamp_end=227.0,
                client_request="Rename the button to Scrape Details.",
                client_quote="Instead of Process All Businesses, let's call this Scrape Details.",
                location="Side Panel → Scrape Details card → Primary action",
                current_state="Button shows Process All Businesses.",
                desired_state="Button shows Scrape Details.",
                implementation_requirements=["Change visible text.", "Preserve click handler."],
                constraints=["Preserve processing behavior."],
                relevant_files=["src/sidepanel/SidePanel.tsx"],
                relevant_symbols=["ScrapeDetailsCard"],
                do_not_modify=["scraping logic"],
                confidence=0.97,
            )
        ],
        unclear_items=['00:08:23 — "Make this section cleaner"'],
    )
    markdown = build_implementation_prompt(analysis)
    assert "# Client Feedback Implementation Prompt" in markdown
    assert "## FB-001 — Rename processing button" in markdown
    assert "# Global Constraints" in markdown
    assert "# Items Requiring Judgment or Clarification" in markdown
    assert "# Implementation Procedure" in markdown


def test_cache_source_changed_without_manifest_does_not_invalidate(tmp_path: Path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    metadata = VideoMetadata(
        path=str(video),
        filename=video.name,
        duration=10.0,
        width=1920,
        height=1080,
        has_audio=True,
        file_size=video.stat().st_size,
        modified_at=video.stat().st_mtime,
        sha256="abc",
    )
    assert not source_changed(None, metadata)


def test_cache_source_changed_detects_hash_mismatch(tmp_path: Path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    metadata = VideoMetadata(
        path=str(video),
        filename=video.name,
        duration=10.0,
        width=1920,
        height=1080,
        has_audio=True,
        file_size=video.stat().st_size,
        modified_at=video.stat().st_mtime,
        sha256="abc",
    )
    work_dir = work_dir_for_video(tmp_path / "work", video)
    save_manifest(work_dir, metadata, ProjectCacheInfo())
    manifest = load_manifest(work_dir)
    changed_metadata = metadata.model_copy(update={"sha256": "different"})
    assert source_changed(manifest, changed_metadata)


def test_cache_unchanged_video_reused(tmp_path: Path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    metadata = VideoMetadata(
        path=str(video),
        filename=video.name,
        duration=10.0,
        width=1920,
        height=1080,
        has_audio=True,
        file_size=video.stat().st_size,
        modified_at=video.stat().st_mtime,
        sha256="same-hash",
    )
    work_dir = work_dir_for_video(tmp_path / "work", video)
    save_manifest(work_dir, metadata, None)
    manifest = load_manifest(work_dir)
    assert not source_changed(manifest, metadata)


def test_content_prompt_builder_renders_sections():
    analysis = ContentFeatureAnalysis(
        video="sample-video.mp4",
        title_guess="How I grew to 100k",
        topic="YouTube growth strategy",
        overall_direction="Educational talking-head with screen demos",
        summary="A step-by-step breakdown of a growth strategy.",
        hooks=[
            HookFeature(
                id="HOOK-001",
                timestamp_start=0.0,
                timestamp_end=8.0,
                hook_type="curiosity_gap",
                transcript="Nobody talks about this part of growing on YouTube.",
                technique="Opens with a contrarian claim",
                effect="Creates immediate curiosity",
            )
        ],
        scripting_structure="Hook → problem → 3 steps → recap → CTA",
        pacing_flow="Fast hook, slower explanation, accelerates at CTA",
        phrasing_language="Conversational, second person, short sentences",
        visual_narration_alignment="Talking head for setup, screen recording for demos",
        speaking_style="Energetic but controlled, emphasis on key numbers",
        segments=[
            FinalContentSegment(
                id="SEG-001",
                title="Opening hook",
                segment_type="hook",
                timestamp_start=0.0,
                timestamp_end=8.0,
                what_happens="Creator challenges a common belief",
                on_screen="Close-up talking head",
                narration="Nobody talks about this part...",
                visual_audio_relationship="Face fills frame to create intimacy",
                pacing_role="Grab attention immediately",
                speaking_notes="Slightly faster pace, raised energy",
                notable_phrases=["Nobody talks about this"],
                confidence=0.92,
            )
        ],
        recreation_principles=["Open with a contrarian hook", "Use 3 numbered steps"],
    )
    markdown = build_playbook_generation_prompt(analysis)
    assert "# Video Content Playbook Generation Prompt" in markdown
    assert "### Hooks" in markdown
    assert "### Scripting & Pacing" in markdown
    assert "### Speaking Style" in markdown
    assert "## Playbook Output Requirements" in markdown
    assert "HOOK-001" in markdown


def test_cli_extract_help():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "playbook" in result.stdout.lower()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.stdout
    assert "extract" in result.stdout


def test_cli_missing_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "work").mkdir()
    video = tmp_path / "input" / "clip.mp4"
    video.write_bytes(b"not-a-real-video")
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)

    with patch("video2prompt.cli.probe_video") as probe_mock, patch(
        "video2prompt.cli.extract_audio"
    ), patch("video2prompt.cli.transcribe_audio") as transcribe_mock:
        probe_mock.return_value = VideoMetadata(
            path=str(video),
            filename=video.name,
            duration=1.0,
            width=100,
            height=100,
            has_audio=True,
            file_size=1,
            modified_at=1.0,
            sha256="hash",
        )
        transcribe_mock.return_value = Transcript(segments=[])
        result = runner.invoke(app, ["analyze", str(video), "--force"])
    assert result.exit_code == 1
    assert "CURSOR_API_KEY" in result.stdout


def test_cli_missing_ffmpeg(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with patch(
        "video2prompt.cli.probe_video",
        side_effect=FFmpegNotFoundError(
            "FFmpeg was not found.\n\nInstall FFmpeg and ensure `ffmpeg` and `ffprobe` are available on PATH."
        ),
    ):
        result = runner.invoke(app, ["analyze", str(video)])
    assert result.exit_code == 1
    assert "FFmpeg" in result.stdout


def test_cli_nonexistent_video(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["analyze", "missing.mp4"])
    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()


def test_cli_empty_input_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 1
    assert "No supported videos" in result.stdout


@patch("video2prompt.cli.analyze_visual_feedback_parallel")
@patch("video2prompt.providers.cursor.Agent.prompt")
def test_mocked_end_to_end_pipeline(mock_prompt, mock_parallel_vision, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    video = tmp_path / "input" / "sample.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")

    candidates_json = json.dumps(
        {
            "candidates": [
                {
                    "id": "FB-001",
                    "start_seconds": 1.0,
                    "end_seconds": 3.0,
                    "transcript_quote": "Rename this button.",
                    "category": "rename",
                    "summary": "Rename button",
                    "needs_visual_context": True,
                    "possible_superseded_instruction": False,
                }
            ]
        }
    )
    visual_json = json.dumps(
        {
            "id": "FB-001",
            "title": "Rename button",
            "timestamp_start": 1.0,
            "timestamp_end": 3.0,
            "category": "rename",
            "client_quote": "Rename this button.",
            "implementation_intent": "Rename the primary action button.",
            "constraints": [],
            "likely_behavior_change": False,
            "confidence": 0.9,
            "evidence_frames": [],
            "supersedes": [],
        }
    )
    final_json = json.dumps(
        {
            "video": "sample.mp4",
            "project": None,
            "summary": "Rename a button.",
            "global_constraints": [],
            "items": [
                {
                    "id": "FB-001",
                    "title": "Rename button",
                    "category": "rename",
                    "timestamp_start": 1.0,
                    "timestamp_end": 3.0,
                    "client_request": "Rename the button.",
                    "client_quote": "Rename this button.",
                    "implementation_requirements": ["Update button label."],
                    "constraints": [],
                    "relevant_files": [],
                    "relevant_symbols": [],
                    "do_not_modify": [],
                    "confidence": 0.9,
                    "evidence_frames": [],
                    "superseded_feedback": [],
                }
            ],
            "unclear_items": [],
        }
    )

    mock_result = MagicMock()
    mock_result.status = "finished"
    mock_result.result = candidates_json
    mock_result.id = "run-1"
    mock_result.duration_ms = 10

    visual_result = MagicMock(status="finished", result=visual_json, id="run-2", duration_ms=10)
    final_result = MagicMock(status="finished", result=final_json, id="run-3", duration_ms=10)
    mock_prompt.side_effect = [mock_result, final_result]
    mock_parallel_vision.return_value = [VisualFeedbackAnalysis.model_validate_json(visual_json)]

    with patch("video2prompt.cli.probe_video") as probe_mock, patch(
        "video2prompt.cli.extract_audio"
    ), patch("video2prompt.cli.transcribe_audio") as transcribe_mock, patch(
        "video2prompt.cli.extract_frames_for_candidate"
    ) as frames_mock:
        probe_mock.return_value = VideoMetadata(
            path=str(video),
            filename=video.name,
            duration=5.0,
            width=1280,
            height=720,
            has_audio=True,
            file_size=10,
            modified_at=1.0,
            sha256="hash",
        )
        transcribe_mock.return_value = Transcript(
            segments=[TranscriptSegment(start=1.0, end=3.0, text="Rename this button.")]
        )
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"jpg")
        frames_mock.return_value = [frame]

        result = runner.invoke(app, ["analyze", str(video), "--force"])

    assert result.exit_code == 0, result.stdout
    prompt_path = tmp_path / "output" / "sample-implementation-prompt.md"
    json_path = tmp_path / "output" / "sample-feedback.json"
    assert prompt_path.exists()
    assert json_path.exists()
    assert "FB-001" in prompt_path.read_text(encoding="utf-8")


def test_create_provider_cursor(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    settings = Settings.from_env(project_root=tmp_path, llm_provider="cursor")
    provider = create_provider(settings, tmp_path)
    assert provider.__class__.__name__ == "CursorProvider"


def test_create_provider_invalid(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid LLM provider"):
        Settings.from_env(project_root=tmp_path, llm_provider="unknown")


def test_append_image_paths(tmp_path: Path):
    image = tmp_path / "shot.jpg"
    image.write_bytes(b"jpg")
    prompt = append_image_paths("Analyze this", [image])
    assert "Attached screenshots:" in prompt
    assert str(image.resolve()) in prompt


def test_parse_claude_json_output_structured():
    raw = '{"structured_output": {"color": "white"}, "result": ""}'
    assert parse_claude_json_output(raw) == '{"color": "white"}'


def test_provider_model_resolution(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "reasoning-model")
    monkeypatch.setenv("CODEX_VISION_MODEL", "vision-model")
    monkeypatch.setenv("CODEX_SYNTHESIS_MODEL", "synthesis-model")
    settings = Settings.from_env(project_root=tmp_path, llm_provider="codex")
    assert settings.effective_reasoning_model() == "reasoning-model"
    assert settings.effective_vision_model() == "vision-model"
    assert settings.effective_synthesis_model() == "synthesis-model"


def test_provider_model_cli_overrides(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "env-reasoning")
    settings = Settings.from_env(
        project_root=tmp_path,
        llm_provider="codex",
        vision_model="cli-vision",
        synthesis_model="cli-synthesis",
    )
    assert settings.effective_vision_model() == "cli-vision"
    assert settings.effective_synthesis_model() == "cli-synthesis"


def test_codex_reasoning_effort_defaults_to_medium(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CODEX_REASONING_EFFORT", raising=False)
    settings = Settings.from_env(project_root=tmp_path, llm_provider="codex")
    assert settings.codex_reasoning_effort == "medium"


def test_codex_compatible_json_schema_adds_additional_properties_false():
    schema = codex_compatible_json_schema(FeedbackCandidatesResult.model_json_schema())
    assert schema["additionalProperties"] is False
    candidate = schema["$defs"]["FeedbackCandidate"]
    assert candidate["additionalProperties"] is False
    assert "possible_superseded_instruction" in candidate["required"]


def test_resolved_transcription_backend_auto_codex(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "token",
                    "account_id": "acct",
                    "refresh_token": "refresh",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    settings = Settings.from_env(project_root=tmp_path, llm_provider="codex", transcription_backend="auto")
    assert settings.resolve_transcription_engine() == "codex"


def test_resolved_transcription_backend_auto_openai(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings.from_env(project_root=tmp_path, llm_provider="openai", transcription_backend="auto")
    assert settings.resolve_transcription_engine() == "openai"


def test_resolved_transcription_backend_local_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env(project_root=tmp_path, llm_provider="cursor", transcription_backend="auto")
    assert settings.resolve_transcription_engine() == "local"


def test_parse_openai_verbose_response():
    from video2prompt.openai_transcription import _parse_openai_verbose_response

    body = {
        "language": "english",
        "duration": 12.5,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " Hello"},
            {"start": 2.0, "end": 5.0, "text": " world"},
        ],
    }
    transcript = _parse_openai_verbose_response(body, 12.5)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Hello"


def test_settings_creates_project_directories(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env(project_root=tmp_path)
    assert settings.input_dir.is_dir()
    assert settings.output_dir.is_dir()
    assert settings.work_dir.is_dir()


def test_prompt_continue_accepts_defaults():
    from video2prompt.interactive import prompt_continue

    with patch("video2prompt.interactive.typer.prompt", return_value=""):
        assert prompt_continue() is True
    with patch("video2prompt.interactive.typer.prompt", return_value="n"):
        assert prompt_continue() is False


def test_cli_start_help():
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
    assert "Interactive wizard" in result.stdout
    assert "mode" in result.stdout.lower()


def test_cli_transcription_option_in_help():
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--transcription" in result.stdout
    assert "--provider" in result.stdout
