You are analyzing a client screen-recording transcript to find actionable feedback moments.

Do not invent requirements.
Separate what the client explicitly requested from what you inferred.
When confidence is low, say so.
The client's later instruction overrides an earlier conflicting instruction.
Do not assume filenames or components unless you verified them in the supplied project.
Do not modify project files.
Output only valid JSON matching the supplied schema.

## Task

Review the timestamped transcript below and identify every likely actionable feedback moment.

Find:
- explicit requests
- implicit bug reports
- UI requests
- behavior changes
- removals
- additions
- renames
- layout changes
- workflow changes
- constraints
- statements about what must NOT change
- later corrections to earlier feedback
- vague requests requiring visual context

Rules:
- Do not invent requirements.
- Preserve chronological order.
- Assign stable IDs like FB-001, FB-002, ...
- Set `needs_visual_context` true when the client uses deictic language such as "this", "that", "here", "this button", "this section", "move this", etc.
- Set `possible_superseded_instruction` true when the quote sounds like a correction to earlier feedback.
- Use one of these categories: ui, ux, bug, behavior, content, layout, addition, removal, rename, performance, constraint, question, unclear, other.

## Transcript

{{TRANSCRIPT_JSON}}

## Output schema

Return a JSON object matching this schema exactly:

{{SCHEMA_JSON}}

Return JSON only. No markdown fences unless unavoidable.
