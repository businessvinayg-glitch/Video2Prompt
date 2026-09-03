from __future__ import annotations

from pydantic import BaseModel, Field


CONTENT_SEGMENT_TYPES = [
    "hook",
    "intro",
    "context_setup",
    "main_point",
    "demonstration",
    "story",
    "transition",
    "recap",
    "cta",
    "outro",
    "b_roll",
    "montage",
    "other",
]


class ContentSegmentCandidate(BaseModel):
    id: str
    start_seconds: float
    end_seconds: float
    segment_type: str
    transcript_quote: str
    summary: str
    needs_visual_context: bool = True


class ContentSegmentsResult(BaseModel):
    segments: list[ContentSegmentCandidate]


class VisualContentAnalysis(BaseModel):
    id: str
    title: str
    timestamp_start: float
    timestamp_end: float
    segment_type: str
    on_screen: str
    narration_summary: str
    visual_audio_relationship: str
    hook_or_retention_technique: str | None = None
    pacing_notes: str
    speaking_delivery: str
    notable_phrases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[str] = Field(default_factory=list)


class HookFeature(BaseModel):
    id: str
    timestamp_start: float
    timestamp_end: float
    hook_type: str
    transcript: str
    technique: str
    effect: str


class FinalContentSegment(BaseModel):
    id: str
    title: str
    segment_type: str
    timestamp_start: float
    timestamp_end: float
    what_happens: str
    on_screen: str
    narration: str
    visual_audio_relationship: str
    pacing_role: str
    speaking_notes: str
    notable_phrases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ContentFeatureAnalysis(BaseModel):
    video: str
    title_guess: str | None = None
    topic: str
    overall_direction: str
    summary: str
    hooks: list[HookFeature] = Field(default_factory=list)
    scripting_structure: str
    pacing_flow: str
    phrasing_language: str
    visual_narration_alignment: str
    speaking_style: str
    segments: list[FinalContentSegment] = Field(default_factory=list)
    recreation_principles: list[str] = Field(default_factory=list)
