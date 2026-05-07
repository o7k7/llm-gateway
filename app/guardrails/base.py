from enum import StrEnum
from typing import Protocol

from attr import dataclass

from app.schemas import ChatRequest, Tenant


class GuardrailOutcome(StrEnum):
    PASSED = "PASSED"
    TRANSFORMED = "TRANSFORMED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    outcome: GuardrailOutcome
    request: ChatRequest
    reason: str | None = None
    metadata: dict[str, object] | None = None


class Guardrail(Protocol):
    name: str

    async def check(self, request: ChatRequest, tenant: Tenant) -> GuardrailResult: ...
