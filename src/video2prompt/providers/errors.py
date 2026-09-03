from __future__ import annotations


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str = "",
        feedback_id: str | None = None,
        run_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.feedback_id = feedback_id
        self.run_id = run_id
        self.retryable = retryable
