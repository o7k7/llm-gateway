import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from app.accounting import Ledger, PricingTable, TokenBucket
from app.auth import CurrentTenant
from app.backends import (
    Backend,
    BackendAuthError,
    BackendError,
    BackendRateLimitError,
    BackendRegistry,
    BackendTimeoutError,
    BackendUnavailableError,
)
from app.cache import CachedEntry, CacheKey, cache_key_hash
from app.dependencies import (
    CurrentBackends,
    CurrentBucket,
    CurrentCache,
    CurrentEstimator,
    CurrentGuardrails,
    CurrentLedger,
    CurrentPricing,
)
from app.guardrails import GuardrailBlockedError
from app.routing.routing import resolve_backend as _routing_resolve
from app.schemas import ChatChunk, ChatRequest, ChoiceChunk, Delta, Tenant, Usage
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette import status

logger = logging.getLogger(__name__)

chat_route_v2 = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


_DEFAULT_PREFLIGHT_MAX_TOKENS = 1024


@chat_route_v2.post("/completions", tags=["chat"], response_model=None)
async def chat_completions(
    req: ChatRequest,
    tenant: CurrentTenant,
    backends: CurrentBackends,
    bucket: CurrentBucket,
    guardrails: CurrentGuardrails,
    ledger: CurrentLedger,
    estimator: CurrentEstimator,
    pricing: CurrentPricing,
    sm_cache: CurrentCache,
) -> StreamingResponse | JSONResponse:
    try:
        transformed_req, guardrail_results = await guardrails.run(req, tenant)
    except GuardrailBlockedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Request blocked by content guardrail",
                    "type": "guardrail_blocked",
                    "guardrail": e.guardrail,
                    "reason": e.reason,
                }
            },
        ) from e

    backend, route_reason = _resolve_backend(backends, transformed_req)
    current_spend = await ledger.current_spend_usd(tenant_id=tenant.id)

    if current_spend >= tenant.limits.daily_budget_usd > 0:
        raise _backend_error_to_http(BackendRateLimitError("Daily budget exhausted"))

    await _enforce_rpm(bucket, tenant)
    key = cache_key_hash(transformed_req, tenant_id=tenant.id, model=backend.model)
    prompt_text = transformed_req.text_for_routing()

    cached = None
    if sm_cache is not None and transformed_req.stream_only_hint():
        cached = await sm_cache.get(key=key, prompt=prompt_text)

    common_headers = _base_headers(
        tenant=tenant,
        backend=backend,
        route_reason=route_reason,
        guardrail_results=guardrail_results,
    )

    if cached is not None:
        common_headers["X-Gateway-Cache"] = "HIT"
        return _cache_hit_response(req=transformed_req, cached=cached, headers=common_headers)

    common_headers["X-Gateway-Cache"] = "MISS"

    estimated_cost = estimator.estimate_budget(
        transformed_req, default_max_tokens=_DEFAULT_PREFLIGHT_MAX_TOKENS
    )

    common_headers["X-Gateway-Estimated-Cost"] = str(estimated_cost)
    # Consume - Pre-flight
    await _enforce_tpm(bucket, tenant, estimated_cost)

    logger.info(
        "route: model=%s → backend=%s (reason=%s)",
        req.model,
        backend.name,
        route_reason,
    )

    if transformed_req.stream:
        return StreamingResponse(
            _sse_stream_accounted(
                backend=backend,
                req=transformed_req,
                est_cost=estimated_cost,
                pricing=pricing,
                tenant=tenant,
                bucket=bucket,
                ledger=ledger,
            ),
            media_type="text/event-stream",
            headers={
                **common_headers,
                "Cache-Control": "no-cache",
                # https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    return JSONResponse(
        await _collect_non_streaming_accounted_with_cache(
            backend=backend,
            req=transformed_req,
            tenant=tenant,
            bucket=bucket,
            ledger=ledger,
            pricing=pricing,
            cache=sm_cache,
            cache_key=key,
            prompt_text=prompt_text,
            estimated_cost=estimated_cost,
        ),
        headers=common_headers,
    )


async def _enforce_rpm(bucket: TokenBucket, tenant: Tenant) -> None:
    """Rejects if the tenant exceeds requests per min"""
    result = await bucket.consume(
        tenant_id=tenant.id,
        suffix="rpm",
        capacity=tenant.limits.requests_per_min,
        refill_per_sec=tenant.limits.requests_per_min / 60.0,
        cost=1,
    )
    if not result.allowed:
        raise _backend_error_to_http(
            BackendRateLimitError("Request rate limit exceeded"), headers={"Retry-After": "60"}
        )


async def _enforce_tpm(bucket: TokenBucket, tenant: Tenant, est_cost: int) -> None:
    result = await bucket.consume(
        tenant_id=tenant.id,
        suffix="tpm",
        capacity=tenant.limits.tokens_per_min,
        refill_per_sec=tenant.limits.tokens_per_min / 60.0,
        cost=est_cost,
    )
    if not result.allowed:
        raise _backend_error_to_http(
            BackendRateLimitError("Token rate limit exceeded"), headers={"Retry-After": "60"}
        )


def _base_headers(
    *,
    tenant: Tenant,
    backend: Backend,
    route_reason: str,
    guardrail_results: list,
) -> dict[str, str]:
    """Base set of X-Gateway-* headers for every response."""
    headers = {
        "X-Gateway-Tenant": tenant.id,
        "X-Gateway-Backend": backend.name,
        "X-Gateway-Model": backend.model,
        "X-Gateway-Route-Reason": route_reason,
    }
    applied = [r for r in guardrail_results if r.outcome.value != "passed"]
    if applied:
        headers["X-Gateway-Guardrails"] = ",".join(f"{type(r).__name__.lower()}" for r in applied)
    return headers


async def _sse_stream_accounted(
    *,
    backend: Backend,
    req: ChatRequest,
    tenant: Tenant,
    ledger: Ledger,
    bucket: TokenBucket,
    pricing: PricingTable,
    est_cost: int,
) -> AsyncIterator[str]:
    usage_finalized: Usage | None = None
    try:
        async for chunk in backend.stream(req):
            if chunk.usage is not None:
                usage_finalized = chunk.usage
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n]"
    except BackendError as e:
        logger.warning("Backend %s stream error: %s", backend.name, e)
        error_payload = {
            "error": {
                "message": str(e),
                "type": _error_type(e),
                "backend": backend.name,
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n]"
    finally:
        if usage_finalized is not None:
            await _post_flight_reconcile(
                tenant=tenant,
                bucket=bucket,
                usage=usage_finalized,
                ledger=ledger,
                pricing=pricing,
                backend=backend,
                est_cost=est_cost,
            )
        else:
            logger.warning("No usage reported for tenant=%s backend=%s", tenant.id, backend.name)


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


def _cache_hit_response(
    *, req: ChatRequest, cached: object, headers: dict[str, str]
) -> StreamingResponse | JSONResponse:
    """Build either a synthetic SSE stream or a JSON body from a cache hit."""
    assert isinstance(cached, CachedEntry)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not req.stream:
        return JSONResponse(
            content={
                "id": chunk_id,
                "object": "chat.completion",
                "created": created,
                "model": cached.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": cached.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": cached.usage.model_dump() if cached.usage else None,
            },
            headers=headers,
        )

    async def _synthetic_stream() -> AsyncIterator[str]:
        chunk1 = ChatChunk(
            id=chunk_id,
            created=created,
            model=cached.model,
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=Delta(role="assistant", content=cached.content),
                )
            ],
        )
        yield f"data: {chunk1.model_dump_json(exclude_none=True)}\n\n"

        chunk2 = ChatChunk(
            id=chunk_id,
            created=created,
            model=cached.model,
            choices=[ChoiceChunk(index=0, delta=Delta(), finish_reason="stop")],
            usage=cached.usage,
        )
        yield f"data: {chunk2.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _synthetic_stream(),
        media_type="text/event-stream",
        headers={
            **headers,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _collect_non_streaming_accounted_with_cache(
    *,
    backend: Backend,
    req: ChatRequest,
    tenant: Tenant,
    bucket: TokenBucket,
    ledger: Ledger,
    pricing: PricingTable,
    cache: CurrentCache | None,
    cache_key: CacheKey,
    prompt_text: str,
    estimated_cost: int,
) -> dict[str, object]:
    content_parts: list[str] = []
    usage: Usage | None = None
    finish_with_reason: str | None = None

    try:
        async for chunk in backend.stream(req):
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
    finally:
        if usage is not None:
            await _post_flight_reconcile(
                tenant=tenant,
                backend=backend,
                bucket=bucket,
                ledger=ledger,
                pricing=pricing,
                usage=usage,
                est_cost=estimated_cost,
            )
            if cache is not None and content_parts:
                try:
                    await cache.put(
                        key=cache_key,
                        prompt=prompt_text,
                        response="".join(content_parts),
                        usage=usage,
                    )
                except Exception:
                    logger.exception(
                        "Cache put failed for tenant=%s backend=%s", tenant.id, backend.name
                    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": backend.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_with_reason or "stop",
            }
        ],
        "usage": usage.model_dump() if usage else None,
    }


async def _post_flight_reconcile(
    *,
    tenant: Tenant,
    backend: Backend,
    bucket: TokenBucket,
    ledger: Ledger,
    pricing: PricingTable,
    usage: Usage,
    est_cost: int,
) -> None:
    actual_total = usage.prompt_tokens + usage.completion_tokens
    cost_usd = pricing.cost_usd(
        backend.model, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
    )

    try:
        await ledger.record(
            tenant_id=tenant.id,
            cost_usd=cost_usd,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            daily_cap_usd=tenant.limits.daily_budget_usd,
        )
    except Exception:
        logger.exception(
            "Ledger record failed (tenant=%s, cost=%f USD",
            tenant.id,
            cost_usd,
        )

    overestimate = max(0, est_cost - actual_total)
    if overestimate > 0:
        try:
            await bucket.refund(
                tenant_id=tenant.id,
                capacity=tenant.limits.tokens_per_min,
                amount=overestimate,
                suffix="tpm",
            )
        except Exception:
            logger.exception(
                "Bucket refund failed (tenant=%s, amount=%d)",
                tenant.id,
                overestimate,
            )


def _error_type(e: BackendError) -> str:
    mapping = {
        BackendAuthError: "backend_auth_error",
        BackendRateLimitError: "backend_rate_limit",
        BackendTimeoutError: "backend_timeout",
        BackendUnavailableError: "backend_unavailable",
    }
    return mapping.get(type(e), "backend_error")


def _backend_error_to_http(e: BackendError, headers: dict[str, str] | None = None) -> HTTPException:
    if isinstance(e, BackendAuthError):
        return HTTPException(
            500, detail={"error": {"message": "Upstream auth failed"}}, headers=headers
        )
    if isinstance(e, BackendRateLimitError):
        return HTTPException(429, detail={"error": {"message": str(e)}}, headers=headers)
    if isinstance(e, BackendTimeoutError):
        return HTTPException(
            504, detail={"error": {"message": "Upstream timeout"}}, headers=headers
        )
    if isinstance(e, BackendUnavailableError):
        return HTTPException(
            502, detail={"error": {"message": "Upstream unavailable"}}, headers=headers
        )
    return HTTPException(
        500, detail={"error": {"message": "Internal backend error"}}, headers=headers
    )


def stream_only_hint(self) -> bool:
    """Hook for request shapes that should never be served from cache.

    For now its always False; future PRs may override for tool calls or
    other non-deterministic shapes where caching would be misleading.
    """
    return False
