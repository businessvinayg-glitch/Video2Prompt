from __future__ import annotations

from video2prompt.content_schemas import ContentFeatureAnalysis
from video2prompt.utils import format_timestamp_range


def build_playbook_generation_prompt(analysis: ContentFeatureAnalysis) -> str:
    lines: list[str] = [
        "# Video Content Playbook Generation Prompt",
        "",
        "## Source",
        "",
        f"Video: `{analysis.video}`",
    ]

    if analysis.title_guess:
        lines.append(f"Title (inferred): `{analysis.title_guess}`")
    lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Your Task",
            "",
            "Using the extracted video features below, produce a detailed **content playbook** that lets someone recreate this video style **1:1 on a similar topic**.",
            "",
            "The playbook must be actionable — not a summary. Include:",
            "",
            "- Exact structural blueprint (sections, order, approximate timing)",
            "- Hook formulas with fill-in-the-blank templates",
            "- Scripting and pacing rules (how ideas are introduced, developed, and landed)",
            "- Phrasing and language patterns to mirror",
            "- Visual direction (what to show when, and how it pairs with words)",
            "- Speaking and delivery notes (tone, pace, emphasis)",
            "- Segment-by-segment reproduction guide",
            "",
            "Prioritize **scripting and pacing** — structure and flow are the most important.",
            "",
            "---",
            "",
            "## Extracted Features",
            "",
            "### Topic & Direction",
            "",
            f"**Topic:** {analysis.topic}",
            "",
            f"**Overall direction:** {analysis.overall_direction}",
            "",
            f"**Summary:** {analysis.summary}",
            "",
            "---",
            "",
        ]
    )

    if analysis.hooks:
        lines.extend(["### Hooks", ""])
        for hook in analysis.hooks:
            lines.extend(
                [
                    f"#### {hook.id} — {hook.hook_type} ({format_timestamp_range(hook.timestamp_start, hook.timestamp_end)})",
                    "",
                    f"**Transcript:** \"{hook.transcript}\"",
                    "",
                    f"**Technique:** {hook.technique}",
                    "",
                    f"**Effect:** {hook.effect}",
                    "",
                ]
            )
        lines.extend(["---", ""])

    lines.extend(
        [
            "### Scripting & Pacing",
            "",
            "#### Structure",
            "",
            analysis.scripting_structure,
            "",
            "#### Flow & momentum",
            "",
            analysis.pacing_flow,
            "",
            "---",
            "",
            "### Phrasing & Language",
            "",
            analysis.phrasing_language,
            "",
            "---",
            "",
            "### Visual vs Narration",
            "",
            analysis.visual_narration_alignment,
            "",
            "---",
            "",
            "### Speaking Style",
            "",
            analysis.speaking_style,
            "",
            "---",
            "",
        ]
    )

    if analysis.segments:
        lines.extend(["### Segment-by-Segment Breakdown", ""])
        for segment in analysis.segments:
            confidence_pct = int(round(segment.confidence * 100))
            lines.extend(
                [
                    f"#### {segment.id} — {segment.title} ({format_timestamp_range(segment.timestamp_start, segment.timestamp_end)})",
                    "",
                    f"**Type:** {segment.segment_type}",
                    f"**Confidence:** {confidence_pct}%",
                    "",
                    f"**What happens:** {segment.what_happens}",
                    "",
                    f"**On screen:** {segment.on_screen}",
                    "",
                    f"**Narration:** {segment.narration}",
                    "",
                    f"**Visual–audio relationship:** {segment.visual_audio_relationship}",
                    "",
                    f"**Pacing role:** {segment.pacing_role}",
                    "",
                    f"**Speaking notes:** {segment.speaking_notes}",
                    "",
                ]
            )
            if segment.notable_phrases:
                lines.append("**Notable phrases:**")
                for phrase in segment.notable_phrases:
                    lines.append(f"- \"{phrase}\"")
                lines.append("")
        lines.extend(["---", ""])

    if analysis.recreation_principles:
        lines.extend(["### Recreation Principles", ""])
        for principle in analysis.recreation_principles:
            lines.append(f"- {principle}")
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Playbook Output Requirements",
            "",
            "When you write the playbook, include:",
            "",
            "1. **Blueprint** — section order, purpose of each section, and target duration ranges",
            "2. **Hook library** — 2–3 hook variants using the same techniques, with templates",
            "3. **Script skeleton** — beat-by-beat outline someone can fill in for a new topic",
            "4. **Pacing guide** — where to speed up, slow down, pause, or cut",
            "5. **Phrase bank** — recurring language patterns and transition phrases",
            "6. **Shot list** — what to show on screen for each beat and how it pairs with words",
            "7. **Delivery notes** — how to speak each section (energy, emphasis, cadence)",
            "8. **Checklist** — final QA list to verify the recreation matches the original style",
            "",
            "Write the playbook so a creator with no access to the original video could reproduce the format faithfully on a new topic.",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"
