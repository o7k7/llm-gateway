from __future__ import annotations


class GuardrailError(Exception):
    def __init__(self, message: str, *, guardrail: str | None = None) -> None:
        super().__init__(message)
        self.guardrail = guardrail


class GuardrailBlockedError(GuardrailError):
    def __init__(
        self, message: str, *, guardrail: str | None = None, reason: str | None = None
    ) -> None:
        super().__init__(message, guardrail=guardrail)
        self.reason = reason
