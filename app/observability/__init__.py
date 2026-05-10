"""Observability — OpenTelemetry tracing + Langfuse integration.

What this module provides
-------------------------
- `tracer`: module-global OTel tracer for creating spans
- `traced()` decorator for wrapping async functions in spans
- `span()` context manager for inline spans with attributes
- Langfuse-specific attribute helpers (cost, tokens, model, tenant)
- Context propagation helpers that survive async generators
"""

from __future__ import annotations

from app.observability.langfuse import (
    LangfuseAttrs,
    set_cache_attrs,
    set_guardrail_attrs,
    set_llm_attrs,
    set_route_attrs,
    set_tenant_attrs,
)
from app.observability.startup import configure_observability, shutdown_observability
from app.observability.tracing import (
    get_current_span,
    get_tracer,
    span,
    traced,
)

__all__ = [
    "LangfuseAttrs",
    "configure_observability",
    "get_current_span",
    "get_tracer",
    "set_cache_attrs",
    "set_guardrail_attrs",
    "set_llm_attrs",
    "set_route_attrs",
    "set_tenant_attrs",
    "shutdown_observability",
    "span",
    "traced",
]
