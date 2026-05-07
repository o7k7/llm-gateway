"""Content guardrails — pre-request inspection and transformation.

Guardrails inspect a ChatRequest before it reaches the backend. They can:
- Transform the request (e.g., scrub PII from messages)
- Block the request (raise GuardrailError → 400)
- Flag metadata for observability (reason, entities detected, etc.)
"""

from __future__ import annotations

from app.guardrails.base import (
    Guardrail,
    GuardrailOutcome,
    GuardrailResult,
)
from app.guardrails.errors import GuardrailBlockedError, GuardrailError
from app.guardrails.pii import PresidioPIIGuardrail
from app.guardrails.registry import GuardrailRegistry

__all__ = [
    "Guardrail",
    "GuardrailBlockedError",
    "GuardrailError",
    "GuardrailOutcome",
    "GuardrailRegistry",
    "GuardrailResult",
    "PresidioPIIGuardrail",
]
