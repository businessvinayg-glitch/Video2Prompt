You are analyzing a video transcript to identify distinct content segments for recreation analysis.

Do not invent content that is not supported by the transcript.
Separate what is explicitly said from what you infer.
When confidence is low, say so.
Output only valid JSON matching the supplied schema.

## Task

Review the timestamped transcript and identify every meaningful content segment.

Find:
- hooks and opening grabs
- intros and context setup
- main points and demonstrations
- stories and examples
- transitions between ideas
- recaps and summaries
- calls to action
- outros and sign-offs
- b-roll or montage moments (when narration continues but topic shifts visually — mark for visual analysis)

Rules:
- Preserve chronological order.
- Assign stable IDs like SEG-001, SEG-002, ...
- Set `needs_visual_context` true for segments where what is shown on screen likely matters (demos, b-roll, montage, hooks with visual payoff, etc.).
- Use one of these segment types: hook, intro, context_setup, main_point, demonstration, story, transition, recap, cta, outro, b_roll, montage, other.
- Merge micro-pauses; split when the topic, beat, or rhetorical function clearly changes.
- `transcript_quote` should be a representative excerpt from that segment.
- `summary` should describe what is happening in this beat in one or two sentences.

## Transcript

{{TRANSCRIPT_JSON}}

## Output schema

Return a JSON object matching this schema exactly:

{{SCHEMA_JSON}}

Return JSON only. No markdown fences unless unavoidable.
