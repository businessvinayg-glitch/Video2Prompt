from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FEEDBACK_CATEGORIES = [
    "ui",
    "ux",
    "bug",
    "behavior",
    "content",
    "layout",
    "addition",
    "removal",
    "rename",
    "performance",
    "constraint",
    "question",
    "unclear",
    "other",
]


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str | None = None
    duration: float | None = None


class VideoMetadata(BaseModel):
    path: str
    filename: str
    duration: float
    width: int
    height: int
    frame_rate: float | None = None
    has_audio: bool
    file_size: int
    modified_at: float
    sha256: str


class FeedbackCandidate(BaseModel):
    id: str
    start_seconds: float
    end_seconds: float
    transcript_quote: str
    category: str
    summary: str
    needs_visual_context: bool
    possible_superseded_instruction: bool = False


class FeedbackCandidatesResult(BaseModel):
    candidates: list[FeedbackCandidate]


class VisualFeedbackAnalysis(BaseModel):
    id: str
    title: str
    timestamp_start: float
    timestamp_end: float
    category: str
    client_quote: str
    location: str | None = None
    screen: str | None = None
    section: str | None = None
    element: str | None = None
    current_state: str | None = None
    requested_state: str | None = None
    implementation_intent: str
    constraints: list[str] = Field(default_factory=list)
    likely_behavior_change: bool = False
    ambiguity: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)


class CodebaseMapping(BaseModel):
    relevant_files: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    do_not_modify: list[str] = Field(default_factory=list)


class FinalFeedbackItem(BaseModel):
    id: str
    title: str
    category: str
    timestamp_start: float
    timestamp_end: float
    client_request: str
    client_quote: str
    location: str | None = None
    current_state: str | None = None
    desired_state: str | None = None
    implementation_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    do_not_modify: list[str] = Field(default_factory=list)
    ambiguity: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[str] = Field(default_factory=list)
    superseded_feedback: list[str] = Field(default_factory=list)


class FeedbackAnalysis(BaseModel):
    video: str
    project: str | None = None
    summary: str
    global_constraints: list[str] = Field(default_factory=list)
    items: list[FinalFeedbackItem] = Field(default_factory=list)
    unclear_items: list[str] = Field(default_factory=list)


class SourceCacheInfo(BaseModel):
    path: str
    file_size: int
    modified_at: float
    sha256: str


class ProjectCacheInfo(BaseModel):
    git_head: str | None = None
    git_status_before: str | None = None


class WorkCacheManifest(BaseModel):
    source: SourceCacheInfo
    project: ProjectCacheInfo | None = None
