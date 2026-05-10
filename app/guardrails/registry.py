"""Registry that runs guardrails in order and handles short-circuit logic."""

from __future__ import annotations

import logging

from app.guardrails.base import Guardrail, GuardrailOutcome, GuardrailResult
from app.guardrails.errors import GuardrailBlockedError
from app.observability import span, set_guardrail_attrs
from app.schemas.chat import ChatRequest
from app.schemas.tenant import Tenant

logger = logging.getLogger(__name__)


class GuardrailRegistry:
    """Runs a list of guardrails in order, transforming the request as needed"""

    def __init__(self) -> None:
        self._guardrails: list[Guardrail] = []

    def register(self, guardrail: Guardrail) -> None:
        if any(g.name == guardrail.name for g in self._guardrails):
            raise ValueError(f"Guardrail {guardrail.name!r} already registered")
        self._guardrails.append(guardrail)

    def names(self) -> list[str]:
        return [g.name for g in self._guardrails]

    async def run(
        self, req: ChatRequest, tenant: Tenant
    ) -> tuple[ChatRequest, list[GuardrailResult]]:
        """Run all guardrails in order. Returns (possibly-transformed request,
        list of per-guardrail results for observability)
        """
        current = req
        results: list[GuardrailResult] = []

        for g in self._guardrails:
            async with span(f"Guardrail {g.name!r}") as g_span:
                result = await g.check(current, tenant)
                results.append(result)

                set_guardrail_attrs(
                    g_span,
                    name=g.name,
                    outcome=result.outcome.value,
                    metadata=result.metadata,
                )

                if result.outcome is GuardrailOutcome.BLOCKED:
                    logger.info(
                        "Guardrail %s blocked request for tenant %s: %s",
                        g.name,
                        tenant.id,
                        result.reason,
                    )
                    raise GuardrailBlockedError(
                        f"Blocked by {g.name} guardrail",
                        guardrail=g.name,
                        reason=result.reason,
                    )

                if result.outcome is GuardrailOutcome.TRANSFORMED:
                    logger.debug(
                        "Guardrail %s transformed request for tenant %s: %s",
                        g.name,
                        tenant.id,
                        result.metadata,
                    )
                    current = result.request

        return current, results
