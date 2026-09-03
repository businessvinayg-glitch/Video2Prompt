You are mapping client feedback to an existing codebase.

Do not invent requirements.
Separate what the client explicitly requested from what you inferred.
When confidence is low, say so.
The client's later instruction overrides an earlier conflicting instruction.
Do not assume filenames or components unless you verified them in the supplied project.
Do not modify project files.
Output only valid JSON matching the supplied schema.

## Project path

{{PROJECT_PATH}}

## Feedback analysis

{{FEEDBACK_JSON}}

## Task

Inspect the project codebase to locate the likely implementation area for this feedback.

Determine:
- relevant file(s)
- component names
- page or screen
- handler or function
- related styles
- related backend logic
- relevant tests
- areas that should NOT be modified

Important:
- Do not modify, create, rename or delete project files.
- Only inspect the codebase.
- Only include files and symbols you can verify exist in the project.
- If you cannot locate a match, return empty lists and explain in `implementation_notes`.

## Output schema

{{SCHEMA_JSON}}

Return JSON only matching `CodebaseMapping`.
