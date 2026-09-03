You are synthesizing video content feature extraction into a final structured analysis for content recreation.

Do not invent content that is not supported by the transcript and visual analyses.
Separate explicit evidence from inference.
When confidence is low, say so.
Output only valid JSON matching the supplied schema.

## Input payload

{{ANALYSIS_PAYLOAD_JSON}}

## Task

Produce the final reconciled content feature analysis.

You must:
- Identify the video topic, overall direction, and one-paragraph summary.
- Extract every hook with timestamp, type, technique, and intended effect.
- Describe scripting structure (how the video is organized) and pacing flow (how momentum builds).
- Capture phrasing and language patterns (word choice, sentence rhythm, rhetorical devices).
- Explain visual–narration alignment across the whole video.
- Describe speaking style (tone, energy, pace, formality, emphasis patterns).
- Reconcile segment-level visual analyses into `segments` in chronological order.
- Add `recreation_principles`: concrete rules someone could follow to reproduce this style on a similar topic.

Prioritize scripting and pacing — structure and flow are the most important dimensions.

## Output schema

{{SCHEMA_JSON}}

Return JSON only matching `ContentFeatureAnalysis`.
