from __future__ import annotations

from video2prompt.providers.cursor import CursorProvider
from video2prompt.providers.errors import LLMProviderError

CursorClient = CursorProvider
CursorClientError = LLMProviderError

__all__ = ["CursorClient", "CursorClientError", "CursorProvider"]
