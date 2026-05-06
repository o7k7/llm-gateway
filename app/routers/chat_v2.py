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
from app.dependencies import (
    CurrentBackends,
    CurrentBucket,
    CurrentEstimator,
    CurrentLedger,
    CurrentPricing,
)
from app.routing.routing import resolve_backend as _routing_resolve
from app.schemas import ChatRequest, Tenant, Usage
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

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
    ledger: CurrentLedger,
    estimator: CurrentEstimator,
    pricing: CurrentPricing,
) -> StreamingResponse | JSONResponse:
    estimated_cost = estimator.estimate_budget(
        req, default_max_tokens=_DEFAULT_PREFLIGHT_MAX_TOKENS
    )

    current_spend = await ledger.current_spend_usd(tenant_id=tenant.id)

    if current_spend >= tenant.limits.daily_budget_usd > 0:
        raise _backend_error_to_http(BackendRateLimitError("Daily budget exhausted"))

    # Consume - Pre-flight
    await _enforce_rpm(bucket, tenant)
    await _enforce_tpm(bucket, tenant, estimated_cost)

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
        "X-Gateway-Tenant": tenant.id,
        "X-Gateway-Estimated-Cost": str(estimated_cost),
    }

    if req.stream:
        return StreamingResponse(
            _sse_stream_accounted(
                backend=backend,
                req=req,
                est_cost=estimated_cost,
                pricing=pricing,
                tenant=tenant,
                bucket=bucket,
                ledger=ledger,
            ),
            media_type="text/event-stream",
            headers={
                **headers,
                "Cache-Control": "no-cache",
                # https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    return JSONResponse(
        await _collect_non_streaming_accounted(
            ledger=ledger,
            bucket=bucket,
            tenant=tenant,
            pricing=pricing,
            backend=backend,
            estimated_cost=estimated_cost,
            req=req,
        ),
        headers=headers,
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


async def _collect_non_streaming_accounted(
    *,
    backend: Backend,
    req: ChatRequest,
    tenant: Tenant,
    bucket: TokenBucket,
    ledger: Ledger,
    pricing: PricingTable,
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
