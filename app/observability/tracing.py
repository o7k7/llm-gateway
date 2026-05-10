import functools
import logging
from contextlib import asynccontextmanager

from opentelemetry import trace
from typing import TypeVar, Type, Any, AsyncIterator, Callable, Awaitable

from opentelemetry.trace import Span, SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

_TRACER_NAME = "app.gateway"

T = TypeVar("T")

def get_tracer() -> trace.Tracer:
    """Returns the gateway's OTel tracer."""
    return trace.get_tracer(__name__)

def get_current_span() -> Span:
    return trace.get_current_span()


@asynccontextmanager
async def span(
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
) -> AsyncIterator[Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as s:
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(Status(StatusCode.ERROR, description=str(exc)))
            raise


def traced(
        name: str | None = None,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:

    def decorator(
            fn: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with span(span_name, kind=kind, attributes=attributes or {}) as s:
                return await fn(*args, **kwargs)

        return wrapper

    return decorator

@asynccontextmanager
async def attach_span(span_: Span) -> AsyncIterator[None]:
    _ = trace.set_span_in_context(span_)
    token = trace.use_span(span_, end_on_exit=False).__enter__()
    try:
        yield
    finally:
        try:
            token.__exit__(None, None, None)
        except Exception:
            logger.debug("Span detach failed")

