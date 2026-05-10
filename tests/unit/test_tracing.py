"""Tests for app.observability.tracing — decorator, span CM, attach_span."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import opentelemetry.trace as trace_api
import pytest
from app.observability.tracing import (
    attach_span,
    get_current_span,
    get_tracer,
    span,
    traced,
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


def _by_name(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, f"Expected exactly one {name!r}, got {len(matches)}"
    return matches[0]


class TestSpanContextManager:
    async def test_span_emits_with_given_name(self, exporter: InMemorySpanExporter) -> None:
        async with span("my.operation"):
            pass
        spans = exporter.get_finished_spans()
        assert _by_name(list(spans), "my.operation") is not None

    async def test_span_attaches_initial_attributes(self, exporter: InMemorySpanExporter) -> None:
        async with span("op", attributes={"tenant": "acme", "count": 5}):
            pass
        s = _by_name(list(exporter.get_finished_spans()), "op")
        assert s.attributes["tenant"] == "acme"
        assert s.attributes["count"] == 5

    async def test_span_records_exceptions_and_sets_error_status(
        self, exporter: InMemorySpanExporter
    ) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with span("op"):
                raise ValueError("boom")

        s = _by_name(list(exporter.get_finished_spans()), "op")
        assert s.status.status_code == trace.StatusCode.ERROR
        assert any(evt.name == "exception" for evt in s.events)

    async def test_nested_spans_have_parent_child_relationship(
        self, exporter: InMemorySpanExporter
    ) -> None:
        async with span("parent"), span("child"):
            pass

        spans = list(exporter.get_finished_spans())
        parent = _by_name(spans, "parent")
        child = _by_name(spans, "child")
        assert child.parent is not None
        assert child.parent.span_id == parent.context.span_id


class TestTracedDecorator:
    async def test_function_wrapped_in_span(self, exporter: InMemorySpanExporter) -> None:
        @traced("my.op")
        async def my_func() -> int:
            return 42

        assert await my_func() == 42
        spans = list(exporter.get_finished_spans())
        assert _by_name(spans, "my.op") is not None

    async def test_default_name_uses_qualified_function_name(
        self, exporter: InMemorySpanExporter
    ) -> None:
        @traced()
        async def my_unique_function() -> None:
            pass

        await my_unique_function()
        spans = list(exporter.get_finished_spans())
        # Default name is module.qualname
        assert any("my_unique_function" in s.name for s in spans)

    async def test_exception_propagates_and_records(self, exporter: InMemorySpanExporter) -> None:
        @traced("failing.op")
        async def failing() -> None:
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            await failing()

        s = _by_name(list(exporter.get_finished_spans()), "failing.op")
        assert s.status.status_code == trace.StatusCode.ERROR

    async def test_return_value_preserved(self, exporter: InMemorySpanExporter) -> None:
        @traced("op")
        async def returning() -> dict[str, int]:
            return {"x": 1}

        result = await returning()
        assert result == {"x": 1}


class TestGetCurrentSpan:
    async def test_returns_active_span_inside_context(self, exporter: InMemorySpanExporter) -> None:
        async with span("outer") as outer:
            current = get_current_span()
            assert current.get_span_context().span_id == outer.get_span_context().span_id

    async def test_returns_noop_span_outside_context(self, exporter: InMemorySpanExporter) -> None:
        current = get_current_span()
        # INVALID_SPAN's span_id is 0
        assert current.get_span_context().span_id == 0
        # Should not raise
        current.set_attribute("x", "y")

    async def test_attributes_set_via_current_appear_on_span(
        self, exporter: InMemorySpanExporter
    ) -> None:
        async with span("op"):
            get_current_span().set_attribute("runtime.flag", True)

        s = _by_name(list(exporter.get_finished_spans()), "op")
        assert s.attributes["runtime.flag"] is True


class TestAttachSpan:
    async def test_children_under_attached_span_have_correct_parent(
        self, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer()

        # Simulate: span started by "request handler"
        parent_span = tracer.start_span("handler.root")

        async def generator() -> AsyncIterator[None]:
            async with attach_span(parent_span), span("child.inside.generator"):
                yield

        async for _ in generator():
            pass

        parent_span.end()

        spans = list(exporter.get_finished_spans())
        root = _by_name(spans, "handler.root")
        child = _by_name(spans, "child.inside.generator")
        assert child.parent is not None
        assert child.parent.span_id == root.context.span_id

    async def test_attach_span_does_not_end_on_exit(self, exporter: InMemorySpanExporter) -> None:
        tracer = get_tracer()
        outer = tracer.start_span("outer")

        async with attach_span(outer):
            pass

        # Outer is still active at this point
        assert outer.is_recording()
        outer.end()
        assert not outer.is_recording()
