"""End-to-end integration tests for the rate-limited chat route.

These exercise the full pipeline: tenant → estimator → bucket → ledger,
using fake backends and fakeredis.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import fakeredis.aioredis
import httpx
import pytest
from app.accounting import Ledger, PricingTable, TokenBucket, TokenEstimator
from app.app_state import AppState
from app.auth import get_current_tenant
from app.backends import BackendRegistry
from app.config import get_config
from app.routers.chat_v2 import chat_route_v2
from app.schemas.chat import ChatChunk, ChatRequest, ChoiceChunk, Delta, Usage
from app.schemas.tenant import Pricing, Tenant, TenantLimits
from fastapi import FastAPI
from httpx import ASGITransport

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeBackend:
    def __init__(
        self,
        name: str = "small",
        model: str = "fake-model",
        usage: Usage | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._usage = usage or Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        yield ChatChunk(
            id="chatcmpl-test",
            created=0,
            model=self.model,
            choices=[ChoiceChunk(index=0, delta=Delta(role="assistant", content="hi"))],
        )
        yield ChatChunk(
            id="chatcmpl-test",
            created=0,
            model=self.model,
            choices=[ChoiceChunk(index=0, delta=Delta(), finish_reason="stop")],
            usage=self._usage,
        )

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


# --------------------------------------------------------------------------
# App builder with all accounting wired up
# --------------------------------------------------------------------------


async def _build_app(
    *,
    backend: _FakeBackend,
    tenant_limits: TenantLimits,
    pricing_entries: tuple[Pricing, ...] = (
        Pricing(model="fake-model", input_per_1m=1.0, output_per_1m=3.0),
    ),
) -> tuple[FastAPI, fakeredis.aioredis.FakeRedis]:
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    await redis_client.flushall()

    registry = BackendRegistry()
    registry.register(backend)

    state = AppState(
        config=get_config(),
        backends=registry,
        redis=redis_client,
        bucket=TokenBucket(redis_client),
        ledger=Ledger(redis_client),
        estimator=TokenEstimator(),
        pricing=PricingTable(entries=pricing_entries),
    )

    app = FastAPI()
    app.state.app_state = state
    app.include_router(chat_route_v2)

    async def _fixed_tenant() -> Tenant:
        return Tenant(id="test-tenant", limits=tenant_limits)

    app.dependency_overrides[get_current_tenant] = _fixed_tenant
    return app, redis_client


def _payload(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "small",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


class TestSuccessfulRequest:
    async def test_200_when_under_limits(self) -> None:
        backend = _FakeBackend()
        limits = TenantLimits(
            requests_per_min=60,
            tokens_per_min=100_000,
            daily_budget_usd=10.0,
        )
        app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
        try:
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post("/chat/completions", json=_payload())
            assert r.status_code == 200
            body = r.json()
            assert body["usage"]["prompt_tokens"] == 10
            assert body["usage"]["completion_tokens"] == 5
        finally:
            await redis_client.aclose()

    async def test_ledger_records_spend(self) -> None:
        backend = _FakeBackend(
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        )
        limits = TenantLimits(
            requests_per_min=60,
            tokens_per_min=100_000,
            daily_budget_usd=10.0,
        )
        app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
        try:
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post("/chat/completions", json=_payload())

            # Ledger should reflect the exact cost
            # 1000 in @ $1/1M + 500 out @ $3/1M = 0.001 + 0.0015 = 0.0025
            ledger = Ledger(redis_client)
            spend = await ledger.current_spend_usd("test-tenant")
            assert spend == pytest.approx(0.0025, abs=1e-6)
        finally:
            await redis_client.aclose()


# --------------------------------------------------------------------------
# Rate-limit rejection
# --------------------------------------------------------------------------


class TestRateLimits:
    async def test_rpm_exhaustion_returns_429(self) -> None:
        backend = _FakeBackend()
        limits = TenantLimits(
            requests_per_min=2,  # only 2 allowed per minute
            tokens_per_min=100_000,
            daily_budget_usd=10.0,
        )
        app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
        try:
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r1 = await client.post("/chat/completions", json=_payload(stream=False))
                r2 = await client.post("/chat/completions", json=_payload(stream=False))
                r3 = await client.post("/chat/completions", json=_payload(stream=False))

            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429
            assert r3.json()["detail"]["error"]["message"] == "Request rate limit exceeded"
            assert r3.headers["retry-after"] == "60"
        finally:
            await redis_client.aclose()

    async def test_tpm_exhaustion_returns_429(self) -> None:
        backend = _FakeBackend()
        limits = TenantLimits(
            requests_per_min=60,
            tokens_per_min=100,  # tiny budget; single request exceeds
            daily_budget_usd=10.0,
        )
        app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
        try:
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post(
                    "/chat/completions",
                    json=_payload(max_tokens=500),  # guarantees >100 tokens
                )
            assert r.status_code == 429
            assert r.json()["detail"]["error"]["message"] == "Token rate limit exceeded"
        finally:
            await redis_client.aclose()

        async def test_daily_budget_exhausted_returns_429(self) -> None:
            backend = _FakeBackend()
            limits = TenantLimits(
                requests_per_min=60,
                tokens_per_min=100_000,
                daily_budget_usd=0.0001,  # essentially no budget
            )
            app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
            try:
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # First request drives the ledger above the 0.0001 USD cap
                    # Fake backend reports usage=10 in + 5 out at $1/$3 per 1M
                    # = 10e-6 + 15e-6 = 25e-6 = $0.000025
                    # Still under 0.0001, so first request succeeds.
                    # But it records spend, and we set cap low enough to exceed
                    # after a few requests.
                    r1 = await client.post("/chat/completions", json=_payload())
                    assert r1.status_code == 200

                    # Drive the ledger past the cap
                    for _ in range(5):
                        await client.post("/chat/completions", json=_payload())

                    # Now the pre-flight budget guard should reject
                    r_final = await client.post("/chat/completions", json=_payload())
                    assert r_final.status_code == 429
                    assert r_final.json()["detail"]["error"]["type"] == "budget_exceeded"
                    assert r_final.json()["detail"]["error"]["daily_cap_usd"] == 0.0001
            finally:
                await redis_client.aclose()

    # --------------------------------------------------------------------------
    # Refund behavior — the big selling point of the pre-flight+post-flight design
    # --------------------------------------------------------------------------

    class TestRefund:
        async def test_overestimate_is_refunded_after_streaming(self) -> None:
            """Pre-flight charges ~1024 tokens (default max_tokens).
            Actual usage is 15 tokens. Bucket should be refunded ~1009 tokens.
            """
            # Backend reports 10 in + 5 out = 15 actual tokens
            backend = _FakeBackend()

            # Give the tenant a just-right TPM budget: can do one request but
            # not two back-to-back if the full pre-charge wasn't refunded.
            limits = TenantLimits(
                requests_per_min=60,
                tokens_per_min=1_200,  # roughly pre-flight estimate + a bit
                daily_budget_usd=10.0,
            )
            app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
            try:
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # First request: pre-charges ~1030 tokens (input + default max)
                    # After stream, actual is 15; refund ~1015.
                    r1 = await client.post("/chat/completions", json=_payload())
                    assert r1.status_code == 200

                    # Second request immediately: only possible if the refund worked.
                    # Without refund, the bucket would be at ~170 remaining (1200 - 1030),
                    # not enough for another 1030-token pre-charge.
                    r2 = await client.post("/chat/completions", json=_payload())
                    assert r2.status_code == 200
            finally:
                await redis_client.aclose()

        async def test_streaming_response_refunds_correctly(self) -> None:
            """Same refund contract for stream=True."""
            backend = _FakeBackend()
            limits = TenantLimits(
                requests_per_min=60,
                tokens_per_min=1_200,
                daily_budget_usd=10.0,
            )
            app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
            try:
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # Streaming request — fully consume the SSE to trigger finally block
                    async with client.stream(
                        "POST", "/chat/completions", json=_payload(stream=True)
                    ) as r1:
                        assert r1.status_code == 200
                        async for _ in r1.aiter_lines():
                            pass  # drain

                    # Second request should work, proving the refund happened
                    async with client.stream(
                        "POST", "/chat/completions", json=_payload(stream=True)
                    ) as r2:
                        assert r2.status_code == 200
                        async for _ in r2.aiter_lines():
                            pass
            finally:
                await redis_client.aclose()

    # --------------------------------------------------------------------------
    # Response headers
    # --------------------------------------------------------------------------

    class TestResponseHeaders:
        async def test_gateway_headers_set_on_success(self) -> None:
            backend = _FakeBackend()
            limits = TenantLimits(
                requests_per_min=60, tokens_per_min=100_000, daily_budget_usd=10.0
            )
            app, redis_client = await _build_app(backend=backend, tenant_limits=limits)
            try:
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    r = await client.post("/chat/completions", json=_payload())
                assert r.status_code == 200
                assert r.headers["x-gateway-tenant"] == "test-tenant"
                assert r.headers["x-gateway-backend"] == "small"
                assert r.headers["x-gateway-model"] == "fake-model"
                assert r.headers["x-gateway-route-reason"] == "explicit"
                # Estimated cost header exists and is numeric
                assert int(r.headers["x-gateway-estimated-cost"]) > 0
            finally:
                await redis_client.aclose()

    # --------------------------------------------------------------------------
    # Auth interaction (dev mode)
    # --------------------------------------------------------------------------

    class TestAuthRequired:
        async def test_no_tenant_override_requires_x_tenant_id_header(self) -> None:
            """When we don't override the tenant dep, dev-mode auth kicks in."""
            # Build without the tenant override by using a different fixture path
            redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
            await redis_client.flushall()

            backend = _FakeBackend()
            registry = BackendRegistry()
            registry.register(backend)

            state = AppState(
                config=get_config(),
                backends=registry,
                redis=redis_client,
                bucket=TokenBucket(redis_client),
                ledger=Ledger(redis_client),
                estimator=TokenEstimator(),
                pricing=PricingTable(
                    entries=(Pricing(model="fake-model", input_per_1m=1.0, output_per_1m=3.0),)
                ),
            )

            app = FastAPI()
            app.state.app_state = state
            app.include_router(chat_route_v2)

            try:
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # No X-Tenant-Id header + no bearer token (dev mode with no JWT key)
                    r = await client.post("/chat/completions", json=_payload())
                assert r.status_code == 401
                assert r.json()["detail"]["error"]["type"] == "auth"
            finally:
                await redis_client.aclose()
