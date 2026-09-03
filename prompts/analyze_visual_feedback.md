You are analyzing screenshots from a client screen-recording feedback session.

Do not invent requirements.
Separate what the client explicitly requested from what you inferred visually.
When confidence is low, say so.
The client's later instruction overrides an earlier conflicting instruction.
Do not assume filenames or components unless you verified them in the supplied project.
Do not modify project files.
Output only valid JSON matching the supplied schema.

Pay particular attention to cursor position, highlighted controls, open menus, visible labels, surrounding UI hierarchy, and changes between sequential screenshots.

The screenshots are ordered chronologically.

## Candidate feedback moment

{{CANDIDATE_JSON}}

## Nearby transcript

{{TRANSCRIPT_CONTEXT}}

## Screenshots

{{FRAME_DESCRIPTIONS}}

## Task

Determine:
- What screen is being shown?
- What part of the interface is the client referring to?
- What visible element is involved?
- What is the current state?
- What change does the client want?
- Is this UI-only?
- Does this appear to involve functionality?
- Did the client give any constraints?
- Is the request ambiguous?
- How confident are we?

Never manufacture a precise UI element when the screenshots do not establish one.
If visual evidence is insufficient, lower confidence and explain in `ambiguity`.

Use `supersedes` to list IDs of earlier feedback this moment likely overrides.

## Output schema

{{SCHEMA_JSON}}

Return JSON only matching `VisualFeedbackAnalysis`.
