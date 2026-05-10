"""Tests for app.observability.langfuse attribute helpers."""

from __future__ import annotations

from collections.abc import Iterator

import opentelemetry.trace as trace_api
import pytest
from app.observability.langfuse import (
    LangfuseAttrs,
    set_cache_attrs,
    set_guardrail_attrs,
    set_llm_attrs,
    set_route_attrs,
    set_tenant_attrs,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]

    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    trace_api.set_tracer_provider(provider)

    try:
        yield exp
    finally:
        provider.shutdown()


def _emit(exporter: InMemorySpanExporter, fn) -> ReadableSpan:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("test") as s:
        fn(s)
    return list(exporter.get_finished_spans())[-1]


class TestTenantAttrs:
    def test_user_id_attached(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(exporter, lambda sp: set_tenant_attrs(sp, tenant_id="acme"))
        assert s.attributes[LangfuseAttrs.USER_ID] == "acme"
        assert LangfuseAttrs.SESSION_ID not in s.attributes

    def test_session_id_attached_when_provided(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_tenant_attrs(sp, tenant_id="acme", session_id="sess-1"),
        )
        assert s.attributes[LangfuseAttrs.USER_ID] == "acme"
        assert s.attributes[LangfuseAttrs.SESSION_ID] == "sess-1"


class TestRouteAttrs:
    def test_all_fields_set(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_route_attrs(
                sp,
                client_model="auto",
                resolved_backend="small",
                resolved_model="Qwen2.5-7B",
                reason="auto_short",
            ),
        )
        assert s.attributes["gateway.route.client_model"] == "auto"
        assert s.attributes["gateway.route.backend"] == "small"
        assert s.attributes["gateway.route.model"] == "Qwen2.5-7B"
        assert s.attributes["gateway.route.reason"] == "auto_short"


class TestGuardrailAttrs:
    def test_basic_fields(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_guardrail_attrs(sp, name="pii", outcome="transformed"),
        )
        assert s.attributes["gateway.guardrail.name"] == "pii"
        assert s.attributes["gateway.guardrail.outcome"] == "transformed"

    def test_metadata_primitives_attached(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_guardrail_attrs(
                sp,
                name="jailbreak",
                outcome="blocked",
                metadata={"similarity": 0.92, "threshold": 0.75},
            ),
        )
        assert s.attributes["gateway.guardrail.similarity"] == 0.92
        assert s.attributes["gateway.guardrail.threshold"] == 0.75

    def test_complex_metadata_stringified(self, exporter: InMemorySpanExporter) -> None:
        """OTel attributes are primitives; complex values get str()."""
        s = _emit(
            exporter,
            lambda sp: set_guardrail_attrs(
                sp,
                name="pii",
                outcome="transformed",
                metadata={"entity_types": ["EMAIL_ADDRESS", "PHONE_NUMBER"]},
            ),
        )
        # List gets stringified
        val = s.attributes["gateway.guardrail.entity_types"]
        assert isinstance(val, str)
        assert "EMAIL_ADDRESS" in val


class TestCacheAttrs:
    def test_hit_outcome(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_cache_attrs(sp, outcome="hit", distance=0.08, tenant_id="acme"),
        )
        assert s.attributes["gateway.cache.outcome"] == "hit"
        assert s.attributes["gateway.cache.distance"] == 0.08
        assert s.attributes["gateway.cache.tenant"] == "acme"

    def test_miss_without_distance(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_cache_attrs(sp, outcome="miss", tenant_id="acme"),
        )
        assert s.attributes["gateway.cache.outcome"] == "miss"
        assert "gateway.cache.distance" not in s.attributes


class TestLLMAttrs:
    def test_model_and_observation_type_set(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(exporter, lambda sp: set_llm_attrs(sp, model="Qwen2.5-7B"))
        assert s.attributes[LangfuseAttrs.MODEL] == "Qwen2.5-7B"
        assert s.attributes[LangfuseAttrs.OBSERVATION_TYPE] == "generation"

    def test_usage_totals_computed(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_llm_attrs(
                sp,
                model="m",
                prompt_tokens=100,
                completion_tokens=50,
            ),
        )
        assert s.attributes[LangfuseAttrs.USAGE_INPUT] == 100
        assert s.attributes[LangfuseAttrs.USAGE_OUTPUT] == 50
        assert s.attributes[LangfuseAttrs.USAGE_TOTAL] == 150

    def test_cost_totals_computed(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_llm_attrs(
                sp,
                model="m",
                cost_input_usd=0.001,
                cost_output_usd=0.002,
            ),
        )
        assert s.attributes[LangfuseAttrs.COST_INPUT] == 0.001
        assert s.attributes[LangfuseAttrs.COST_OUTPUT] == 0.002
        assert s.attributes[LangfuseAttrs.COST_TOTAL] == pytest.approx(0.003)

    def test_prompt_truncation(self, exporter: InMemorySpanExporter) -> None:
        long_prompt = "x" * 5_000
        s = _emit(
            exporter,
            lambda sp: set_llm_attrs(sp, model="m", prompt_text=long_prompt),
        )
        val = s.attributes[LangfuseAttrs.INPUT]
        assert len(val) < len(long_prompt)
        assert "[truncated]" in val

    def test_short_prompts_not_truncated(self, exporter: InMemorySpanExporter) -> None:
        s = _emit(
            exporter,
            lambda sp: set_llm_attrs(sp, model="m", prompt_text="hello"),
        )
        assert s.attributes[LangfuseAttrs.INPUT] == "hello"
