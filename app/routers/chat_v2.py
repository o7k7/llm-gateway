import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from app.backends import (
    Backend,
    BackendAuthError,
    BackendError,
    BackendRateLimitError,
    BackendRegistry,
    BackendTimeoutError,
    BackendUnavailableError,
)
from app.dependencies import CurrentBackends
from app.routing.routing import resolve_backend as _routing_resolve
from app.schemas import ChatChunk, ChatRequest, Usage
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

chat_route_v2 = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@chat_route_v2.post("/completions", tags=["chat"], response_model=None)
async def chat_completions(
    req: ChatRequest, backends: CurrentBackends
) -> StreamingResponse | JSONResponse:
    backend, route_reason = _resolve_backend(backends, req)

    logger.info(
        "route: model=%s → backend=%s (reason=%s)",
        req.model,
        backend.name,
        route_reason,
    )

    headers = {
        "X-Gateway-Backend": backend.name,
        "X-Gateway-Model": backend.model,
        "X-Gateway-Route-Reason": route_reason,
    }

    if req.stream:
        return StreamingResponse(
            _sse_stream(backend.stream(req), backend_name=backend.name),
            media_type="text/event-stream",
            headers={
                **headers,
                "Cache-Control": "no-cache",
                # https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    return JSONResponse(
        await _collect_non_streaming(backend.stream(req), backend.model),
    )


def _resolve_backend(backends: BackendRegistry, req: ChatRequest) -> tuple[Backend, str]:
    try:
        name, reason = _routing_resolve(req, backends)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Unknown model: {req.model!r}",
                    "type": "unknown_backend",
                }
            },
        ) from None
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(e), "type": "no_backends"}},
        ) from e

    return backends.get(name), reason


async def _sse_stream(chunks: AsyncIterator[ChatChunk], *, backend_name: str) -> AsyncIterator[str]:
    try:
        async for chunk in chunks:
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n]"
    except BackendError as e:
        logger.warning("Backend %s stream error: %s", backend_name, e)
        error_payload = {
            "error": {
                "message": str(e),
                "type": _error_type(e),
                "backend": backend_name,
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n]"


async def _collect_non_streaming(chunks: AsyncIterator[ChatChunk], model: str) -> dict[str, object]:
    content_parts: list[str] = []
    usage: Usage | None = None
    finish_with_reason: str | None = None

    try:
        async for chunk in chunks:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    content_parts.append(delta_content)
                if chunk.choices[0].finish_reason:
                    finish_with_reason = chunk.choices[0].finish_reason
            if chunk.usage:
                usage = chunk.usage
    except BackendError as e:
        raise _backend_error_to_http(e) from e

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_with_reason or "stop",
            }
        ],
        "usage": usage.model_dump() if usage else None,
    }


def _error_type(e: BackendError) -> str:
    mapping = {
        BackendAuthError: "backend_auth_error",
        BackendRateLimitError: "backend_rate_limit",
        BackendTimeoutError: "backend_timeout",
        BackendUnavailableError: "backend_unavailable",
    }
    return mapping.get(type(e), "backend_error")


def _backend_error_to_http(e: BackendError) -> HTTPException:
    if isinstance(e, BackendAuthError):
        return HTTPException(500, detail={"error": {"message": "Upstream auth failed"}})
    if isinstance(e, BackendRateLimitError):
        return HTTPException(429, detail={"error": {"message": str(e)}})
    if isinstance(e, BackendTimeoutError):
        return HTTPException(504, detail={"error": {"message": "Upstream timeout"}})
    if isinstance(e, BackendUnavailableError):
        return HTTPException(502, detail={"error": {"message": "Upstream unavailable"}})
    return HTTPException(500, detail={"error": {"message": "Internal backend error"}})
