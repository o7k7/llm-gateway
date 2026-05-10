"""Langfuse-specific span attribute helpers.

Langfuse's OTel ingestion recognizes specific attribute names to
populate its UI — model name, prompt/completion tokens, costs, tenant
id, session id, etc. This module centralizes those names so callers
don't hardcode string constants.

Reference: https://langfuse.com/docs/opentelemetry/example-python-sdk
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from opentelemetry.trace import Span


class LangfuseAttrs(StrEnum):
    """Attribute names Langfuse's OTel collector recognizes."""

    # Core observation fields
    USER_ID = "langfuse.user.id"
    SESSION_ID = "langfuse.session.id"
    TAGS = "langfuse.tags"

    # LLM generation fields
    MODEL = "langfuse.observation.model.name"
    INPUT = "langfuse.observation.input"
    OUTPUT = "langfuse.observation.output"
    USAGE_INPUT = "langfuse.observation.usage.input"
    USAGE_OUTPUT = "langfuse.observation.usage.output"
    USAGE_TOTAL = "langfuse.observation.usage.total"

    # Cost fields (USD)
    COST_INPUT = "langfuse.observation.cost.input"
    COST_OUTPUT = "langfuse.observation.cost.output"
    COST_TOTAL = "langfuse.observation.cost.total"

    # Span type
    OBSERVATION_TYPE = "langfuse.observation.type"


# Observation type values Langfuse recognizes
TYPE_GENERATION = "generation"
TYPE_SPAN = "span"
TYPE_EVENT = "event"


def set_tenant_attrs(span: Span, *, tenant_id: str, session_id: str | None = None) -> None:
    """Attach tenant identity to a span.

    user_id is the stable tenant id; session_id optionally groups
    multiple requests from the same conversation
    """
    span.set_attribute(LangfuseAttrs.USER_ID, tenant_id)
    if session_id:
        span.set_attribute(LangfuseAttrs.SESSION_ID, session_id)


def set_route_attrs(
    span: Span,
    *,
    client_model: str,
    resolved_backend: str,
    resolved_model: str,
    reason: str,
) -> None:
    """Attach routing decision to a span."""
    span.set_attribute("gateway.route.client_model", client_model)
    span.set_attribute("gateway.route.backend", resolved_backend)
    span.set_attribute("gateway.route.model", resolved_model)
    span.set_attribute("gateway.route.reason", reason)


def set_guardrail_attrs(
    span: Span,
    *,
    name: str,
    outcome: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Attach guardrail decision + metadata to a span."""
    span.set_attribute("gateway.guardrail.name", name)
    span.set_attribute("gateway.guardrail.outcome", outcome)
    if metadata:
        for k, v in metadata.items():
            # OTel attributes must be primitive types; stringify complex ones
            if isinstance(v, (str, bool, int, float)):
                span.set_attribute(f"gateway.guardrail.{k}", v)
            else:
                span.set_attribute(f"gateway.guardrail.{k}", str(v))


def set_cache_attrs(
    span: Span,
    *,
    outcome: str,
    distance: float | None = None,
    tenant_id: str | None = None,
) -> None:
    """Attach cache outcome to a span. outcome is 'hit' or 'miss'."""
    span.set_attribute("gateway.cache.outcome", outcome)
    if distance is not None:
        span.set_attribute("gateway.cache.distance", distance)
    if tenant_id is not None:
        span.set_attribute("gateway.cache.tenant", tenant_id)


def set_llm_attrs(
    span: Span,
    *,
    model: str,
    prompt_text: str | None = None,
    completion_text: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_input_usd: float | None = None,
    cost_output_usd: float | None = None,
) -> None:
    """Attach LLM generation fields (Langfuse 'generation' span type).

    Prompt/completion text is truncated to 2048 chars.
    """
    span.set_attribute(LangfuseAttrs.OBSERVATION_TYPE, TYPE_GENERATION)
    span.set_attribute(LangfuseAttrs.MODEL, model)

    if prompt_text is not None:
        span.set_attribute(LangfuseAttrs.INPUT, _truncate(prompt_text))
    if completion_text is not None:
        span.set_attribute(LangfuseAttrs.OUTPUT, _truncate(completion_text))

    if prompt_tokens is not None:
        span.set_attribute(LangfuseAttrs.USAGE_INPUT, prompt_tokens)
    if completion_tokens is not None:
        span.set_attribute(LangfuseAttrs.USAGE_OUTPUT, completion_tokens)
    if prompt_tokens is not None and completion_tokens is not None:
        span.set_attribute(
            LangfuseAttrs.USAGE_TOTAL, prompt_tokens + completion_tokens
        )

    if cost_input_usd is not None:
        span.set_attribute(LangfuseAttrs.COST_INPUT, cost_input_usd)
    if cost_output_usd is not None:
        span.set_attribute(LangfuseAttrs.COST_OUTPUT, cost_output_usd)
    if cost_input_usd is not None and cost_output_usd is not None:
        span.set_attribute(
            LangfuseAttrs.COST_TOTAL, cost_input_usd + cost_output_usd
        )


_MAX_TEXT_LEN = 2048


def _truncate(text: str) -> str:
    if len(text) <= _MAX_TEXT_LEN:
        return text
    return text[: _MAX_TEXT_LEN - 20] + "... [truncated]"
