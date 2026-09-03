from __future__ import annotations

from video2prompt.schemas import FeedbackAnalysis
from video2prompt.utils import format_timestamp, format_timestamp_range


IMPLEMENTATION_PROCEDURE = """# Implementation Procedure

Before making changes:

1. Read all feedback tasks before editing.
2. Inspect the referenced files and surrounding implementation.
3. Verify that inferred code locations are correct.
4. Prefer the client's latest instruction when feedback was superseded.
5. Do not make unrelated refactors.
6. Preserve behavior not explicitly requested to change.

During implementation:

1. Implement each feedback item completely.
2. Keep changes narrowly scoped.
3. Preserve existing architecture where practical.
4. Do not delete functionality merely because it appears unused unless the feedback explicitly requests deletion.

After implementation:

1. Review every feedback item against the code changes.
2. Run relevant tests/build/lint commands.
3. Verify that existing functionality still works.
4. Report each completed feedback ID.
5. Report files changed.
6. Mention anything that could not be implemented confidently.
"""


def build_implementation_prompt(analysis: FeedbackAnalysis) -> str:
    lines: list[str] = [
        "# Client Feedback Implementation Prompt",
        "",
        "## Source",
        "",
        "Recording:",
        f"`{analysis.video}`",
        "",
    ]

    if analysis.project:
        lines.extend(["Project:", f"`{analysis.project}`", ""])
    else:
        lines.extend(["Project:", "_Not specified — locate components by description._", ""])

    lines.extend(
        [
            "## Objective",
            "",
            "Implement the client feedback described below.",
            "",
            "Inspect the existing codebase before changing anything.",
            "",
            "Preserve all existing behavior unless a task explicitly requests a behavior change.",
            "",
            "Do not make unrelated refactors or improvements.",
            "",
            "---",
            "",
        ]
    )

    if analysis.summary:
        lines.extend(["## Summary", "", analysis.summary, "", "---", ""])

    if analysis.global_constraints:
        lines.extend(["# Global Constraints", ""])
        for constraint in analysis.global_constraints:
            lines.append(f"- {constraint}")
        lines.extend(["", "---", ""])

    lines.extend(["# Feedback Tasks", ""])
    for item in analysis.items:
        lines.extend(_render_task(item))
        lines.extend(["", "---", ""])

    if analysis.unclear_items:
        lines.extend(["# Items Requiring Judgment or Clarification", ""])
        for index, unclear in enumerate(analysis.unclear_items, start=1):
            lines.extend(_render_unclear(index, unclear))
            lines.append("")

    lines.append(IMPLEMENTATION_PROCEDURE)
    return "\n".join(lines).rstrip() + "\n"


def _render_task(item) -> list[str]:
    confidence_pct = int(round(item.confidence * 100))
    lines = [
        f"## {item.id} — {item.title}",
        "",
        f"**Recording:** `{format_timestamp_range(item.timestamp_start, item.timestamp_end)}`",
        f"**Category:** {item.category.title()}",
        f"**Confidence:** {confidence_pct}%",
        "",
        "### Client Request",
        "",
        item.client_request,
        "",
        "### Client Evidence",
        "",
        f"> {item.client_quote}",
        "",
    ]

    if item.location:
        lines.extend(["### Location", "", item.location, ""])
    if item.current_state:
        lines.extend(["### Current State", "", item.current_state, ""])
    if item.desired_state:
        lines.extend(["### Desired State", "", item.desired_state, ""])

    if item.relevant_files or item.relevant_symbols:
        lines.append("### Relevant Code")
        lines.append("")
        for path in item.relevant_files:
            lines.append(f"- `{path}`")
        for symbol in item.relevant_symbols:
            lines.append(f"- Component: `{symbol}`")
        lines.append("")

    if item.implementation_requirements:
        lines.extend(["### Implementation Requirements", ""])
        for requirement in item.implementation_requirements:
            lines.append(f"- {requirement}")
        lines.append("")

    if item.constraints:
        lines.extend(["### Constraints", ""])
        for constraint in item.constraints:
            lines.append(f"- {constraint}")
        lines.append("")

    if item.do_not_modify:
        lines.extend(["### Do Not Modify", ""])
        for note in item.do_not_modify:
            lines.append(f"- {note}")
        lines.append("")

    if item.superseded_feedback:
        lines.extend(["### Superseded Instruction", ""])
        for note in item.superseded_feedback:
            lines.append(f"- {note}")
        lines.append("")

    if item.ambiguity:
        lines.extend(["### Ambiguity", "", item.ambiguity, ""])

    return lines


def _render_unclear(index: int, text: str) -> list[str]:
    timestamp = ""
    body = text
    if " — " in text and text[:2].isdigit() is False:
        parts = text.split(" — ", 1)
        if len(parts) == 2:
            timestamp, body = parts
    lines = [f"## UNCLEAR-{index:03d} — {body.splitlines()[0][:80]}", ""]
    if timestamp:
        lines.append(f"**Timestamp:** `{timestamp}`")
        lines.append("")
    lines.append(body)
    lines.append("")
    lines.append(
        "Do not invent a major redesign. If implementing without further clarification, "
        "make only conservative improvements consistent with the existing design system."
    )
    return lines
