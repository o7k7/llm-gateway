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
from app.observability import (
    get_current_span,
    set_cache_attrs,
    set_llm_attrs,
    set_route_attrs,
    set_tenant_attrs,
    span,
)
from app.observability.tracing import attach_span
from app.routing.routing import resolve_backend as _routing_resolve
from app.schemas import ChatChunk, ChatRequest, ChoiceChunk, Delta, Tenant, Usage
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace import Span, SpanKind
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
    root_span = get_current_span()
    set_tenant_attrs(root_span, tenant_id=tenant.id)
    root_span.set_attribute("gateway.request.model", req.model)
    root_span.set_attribute("gateway.request.stream", req.stream)

    async with span("guardrails.run") as g_span:
        try:
            transformed_req, guardrail_results = await guardrails.run(req, tenant)
        except GuardrailBlockedError as e:
            g_span.set_attribute("gateway.guardrail.blocked_by", e.guardrail or "unknown")
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
        g_span.set_attribute("gateway.guardrail.count", len(guardrail_results))

    async with span("routing.resolve") as r_span:
        backend, route_reason = _resolve_backend(backends, transformed_req)
        set_route_attrs(
            r_span,
            client_model=req.model,
            resolved_backend=backend.name,
            resolved_model=backend.model,
            reason=route_reason,
        )
        set_route_attrs(
            root_span,
            client_model=req.model,
            resolved_backend=backend.name,
            resolved_model=backend.model,
            reason=route_reason,
        )
    # Budget guard
    async with span("accounting.budget_guard") as b_span:
        current_spend = await ledger.current_spend_usd(tenant_id=tenant.id)
        b_span.set_attribute("gateway.budget.current_spend_usd", current_spend)
        b_span.set_attribute("gateway.budget.cap_usd", tenant.limits.daily_budget_usd)
        if current_spend >= tenant.limits.daily_budget_usd > 0:
            b_span.set_attribute("gateway.budget.exceeded", True)
            raise _backend_error_to_http(BackendRateLimitError("Daily budget exhausted"))

    await _enforce_rpm(bucket, tenant)
    key = cache_key_hash(transformed_req, tenant_id=tenant.id, model=backend.model)
    prompt_text = transformed_req.text_for_routing()

    cached = None
    if sm_cache is not None:
        async with span("cache.lookup") as c_span:
            cached = await sm_cache.get(key=key, prompt=prompt_text)
            set_cache_attrs(c_span, outcome="HIT" if cached else "MISS", tenant_id=tenant.id)

    common_headers = _base_headers(
        tenant=tenant,
        backend=backend,
        route_reason=route_reason,
        guardrail_results=guardrail_results,
    )

    if cached is not None:
        common_headers["X-Gateway-Cache"] = "HIT"
        set_cache_attrs(root_span, outcome="hit", tenant_id=tenant.id)
        # Record LLM attrs on root for cache hits
        set_llm_attrs(
            root_span,
            model=cached.model,
            prompt_text=prompt_text,
            completion_text=cached.content,
            prompt_tokens=cached.usage.prompt_tokens if cached.usage else None,
            completion_tokens=cached.usage.completion_tokens if cached.usage else None,
        )
        return _cache_hit_response(req=transformed_req, cached=cached, headers=common_headers)

    common_headers["X-Gateway-Cache"] = "MISS"
    set_cache_attrs(root_span, outcome="miss", tenant_id=tenant.id)
    # TPM pre-flight
    estimated_cost = estimator.estimate_budget(
        transformed_req, default_max_tokens=_DEFAULT_PREFLIGHT_MAX_TOKENS
    )

    common_headers["X-Gateway-Estimated-Cost"] = str(estimated_cost)
    root_span.set_attribute("gateway.token.estimated_cost", estimated_cost)
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
            _sse_stream_with_cache_and_accounting(
                backend=backend,
                req=transformed_req,
                est_cost=estimated_cost,
                pricing=pricing,
                tenant=tenant,
                bucket=bucket,
                ledger=ledger,
                prompt_text=prompt_text,
                sm_cache=sm_cache,
                cache_key=key,
                parent_span=root_span,
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


async def _sse_stream_with_cache_and_accounting(
    *,
    backend: Backend,
    req: ChatRequest,
    tenant: Tenant,
    ledger: Ledger,
    bucket: TokenBucket,
    pricing: PricingTable,
    est_cost: int,
    prompt_text: str,
    sm_cache: CurrentCache | None,
    cache_key: CacheKey,
    parent_span: Span,
) -> AsyncIterator[str]:
    async with attach_span(parent_span):
        usage_finalized: Usage | None = None
        content_parts: list[str] = []
        ttft_recorded = False
        async with span("backend.stream", kind=SpanKind.CLIENT) as b_span:
            b_span.set_attribute("gateway.backend.name", backend.name)
            b_span.set_attribute("gateway.backend.model", backend.model)

            try:
                async for chunk in backend.stream(req):
                    if not ttft_recorded and chunk.choices and chunk.choices[0].delta.content:
                        ttft_recorded = True

                        b_span.add_event("ttft")
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_parts.append(chunk.choices[0].delta.content)
                    if chunk.usage is not None:
                        usage_finalized = chunk.usage
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"

                if usage_finalized is not None:
                    cost_in = pricing.get(backend.model).cost_usd(usage_finalized.prompt_tokens, 0)
                    cost_out = pricing.get(backend.model).cost_usd(
                        0, usage_finalized.completion_tokens
                    )
                    set_llm_attrs(
                        b_span,
                        model=backend.model,
                        prompt_text=prompt_text,
                        completion_text="".join(content_parts),
                        prompt_tokens=usage_finalized.prompt_tokens,
                        completion_tokens=usage_finalized.completion_tokens,
                        cost_input_usd=cost_in,
                        cost_output_usd=cost_out,
                    )
            except BackendError as e:
                logger.warning("Backend %s stream error: %s", backend.name, e)
                b_span.record_exception(e)
                error_payload = {
                    "error": {
                        "message": str(e),
                        "type": _error_type(e),
                        "backend": backend.name,
                    }
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                yield "data: [DONE]\n\n]"
                # Post-stream: reconcile and cache, under the parent span
        if usage_finalized is not None:
            async with span("accounting.reconcile"):
                await _post_flight_reconcile(
                    tenant=tenant,
                    backend=backend,
                    bucket=bucket,
                    ledger=ledger,
                    pricing=pricing,
                    usage=usage_finalized,
                    est_cost=est_cost,
                )
            if sm_cache is not None and content_parts:
                async with span("cache.put") as cp_span:
                    cp_span.set_attribute("gateway.cache.tenant", tenant.id)
                    try:
                        await sm_cache.put(
                            key=cache_key,
                            prompt=prompt_text,
                            response="".join(content_parts),
                            usage=usage_finalized,
                        )
                    except Exception as e:
                        cp_span.record_exception(e)
                        logger.exception(
                            "Cache put failed for tenant=%s backend=%s",
                            tenant.id,
                            backend.name,
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
