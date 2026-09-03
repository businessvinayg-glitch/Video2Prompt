You are analyzing screenshots from a video to understand what is happening on screen and how it relates to the narration.

Do not invent visuals that are not supported by the screenshots.
Separate what is explicitly visible from what you infer.
When confidence is low, say so.
Output only valid JSON matching the supplied schema.

Pay attention to:
- what is literally on screen (person, slides, product, text overlays, b-roll, UI, environment)
- how visuals support, contrast with, or delay the spoken message
- pacing cues (cuts, holds, zooms, text reveals)
- hook or retention techniques visible in this moment
- how the speaker delivers lines (energy, pace, emphasis, casual vs scripted)

The screenshots are ordered chronologically.

## Candidate segment

{{CANDIDATE_JSON}}

## Nearby transcript

{{TRANSCRIPT_CONTEXT}}

## Screenshots

{{FRAME_DESCRIPTIONS}}

## Task

Determine:
- What is on screen?
- What is being said in this beat?
- How do visuals and words relate (support, illustrate, contrast, delay payoff, etc.)?
- Is there a hook or retention technique here?
- What pacing choices are visible?
- How is the speaker delivering the lines?
- Any notable phrases worth preserving for recreation?

If visual evidence is insufficient, lower confidence and explain briefly in your notes fields.

## Output schema

{{SCHEMA_JSON}}

Return JSON only matching `VisualContentAnalysis`.
