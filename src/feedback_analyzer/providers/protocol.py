from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    def prompt_text(
        self,
        prompt: str,
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> str: ...

    def prompt_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        stage: str = "",
        feedback_id: str | None = None,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> T: ...
