from __future__ import annotations

from feedback_analyzer.providers.cursor import CursorProvider
from feedback_analyzer.providers.errors import LLMProviderError

CursorClient = CursorProvider
CursorClientError = LLMProviderError

__all__ = ["CursorClient", "CursorClientError", "CursorProvider"]
