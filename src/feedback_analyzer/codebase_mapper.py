from __future__ import annotations

import json
from pathlib import Path

from feedback_analyzer.config import Settings
from feedback_analyzer.providers.protocol import LLMProvider
from feedback_analyzer.schemas import CodebaseMapping, VisualFeedbackAnalysis
from feedback_analyzer.utils import git_head, git_porcelain_status, load_prompt, save_json_file


def map_feedback_to_codebase(
    analysis: VisualFeedbackAnalysis,
    work_dir: Path,
    project_path: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    git_status_before: str,
    force: bool = False,
) -> CodebaseMapping:
    output_path = work_dir / "codebase-analysis" / f"{analysis.id}.json"
    if output_path.exists() and not force:
        return CodebaseMapping.model_validate_json(output_path.read_text(encoding="utf-8"))

    schema_json = json.dumps(CodebaseMapping.model_json_schema(), indent=2)
    prompt = load_prompt(
        "map_to_codebase.md",
        PROJECT_PATH=str(project_path.resolve()),
        FEEDBACK_JSON=analysis.model_dump_json(indent=2),
        SCHEMA_JSON=schema_json,
    )

    mapping = provider.prompt_json(
        prompt,
        CodebaseMapping,
        stage="codebase_mapping",
        feedback_id=analysis.id,
        model=settings.effective_reasoning_model(),
    )
    save_json_file(output_path, mapping)

    git_status_after = git_porcelain_status(project_path)
    if git_status_after != git_status_before:
        new_changes = _diff_porcelain(git_status_before, git_status_after)
        if new_changes:
            print(
                "\n⚠ WARNING: The LLM provider introduced new changes in the project during analysis.\n"
                "Review your working tree before continuing. Existing changes were NOT discarded.\n"
                f"New changes detected:\n{new_changes}\n"
            )

    return mapping


def capture_project_cache_info(project_path: Path) -> tuple[str | None, str]:
    return git_head(project_path), git_porcelain_status(project_path)


def _diff_porcelain(before: str, after: str) -> str:
    before_lines = set(before.splitlines())
    after_lines = set(after.splitlines())
    added = sorted(after_lines - before_lines)
    return "\n".join(added)
