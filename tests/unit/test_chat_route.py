"""Integration tests for the v2 chat route with a fake backend."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import fakeredis.aioredis
import httpx
import pytest
from sentence_transformers import SentenceTransformer

from app.accounting import Ledger, PricingTable, TokenBucket, TokenEstimator
from app.app_state import AppState
from app.auth import get_current_tenant
from app.backends import BackendRegistry
from app.backends.errors import BackendRateLimitError, BackendUnavailableError
from app.cache import Embedder
from app.config import get_config
from app.guardrails import GuardrailRegistry
from app.routers.chat_v2 import chat_route_v2
from app.schemas import Pricing, Tenant, TenantLimits
from app.schemas.chat import ChatChunk, ChatRequest, ChoiceChunk, Delta, Usage
from fastapi import FastAPI
from httpx import ASGITransport


class _FakeBackend:
    def __init__(
        self,
        name: str = "small",
        model: str = "fake-model",
        chunks: list[ChatChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._chunks = chunks or []
        self._error = error

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        if self._error is not None:
            raise self._error
        for c in self._chunks:
            yield c

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


def _content_chunk(text: str, *, model: str = "fake-model") -> ChatChunk:
    return ChatChunk(
        id="chatcmpl-test",
        created=0,
        model=model,
        choices=[ChoiceChunk(index=0, delta=Delta(role="assistant", content=text))],
    )


def _final_chunk(*, model: str = "fake-model") -> ChatChunk:
    return ChatChunk(
        id="chatcmpl-test",
        created=0,
        model=model,
        choices=[ChoiceChunk(index=0, delta=Delta(), finish_reason="stop")],
        usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


async def _app_with_backend(
    backend: Any, tenant_limits: TenantLimits
) -> tuple[FastAPI, fakeredis.aioredis.FakeRedis]:
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    await redis_client.flushall()

    registry = BackendRegistry()
    registry.register(backend)
    config = get_config()
    model = SentenceTransformer(config.cache_embedder_model)
    embedder = Embedder(model, lru_capacity=config.cache_embedder_lru_capacity)

    state = AppState(
        config=config,
        backends=registry,
        redis=redis_client,
        bucket=TokenBucket(redis_client),
        ledger=Ledger(redis_client),
        estimator=TokenEstimator(),
        embedder=embedder,
        cache=None,
        guardrails=GuardrailRegistry(),
        pricing=PricingTable(
            entries=(Pricing(model="fake-model", input_per_1m=1.0, output_per_1m=3.0),)
        ),
    )

    app = FastAPI()
    app.state.app_state = state
    app.include_router(chat_route_v2)

    async def _fixed_tenant() -> Tenant:
        return Tenant(id="test-tenant", limits=tenant_limits)

    app.dependency_overrides[get_current_tenant] = _fixed_tenant
    return app, redis_client


@pytest.fixture
def client_factory():
    async def _make(backend: Any) -> httpx.AsyncClient:
        app, _ = await _app_with_backend(
            backend,
            TenantLimits(
                requests_per_min=2,  # only 2 allowed per minute
                tokens_per_min=100_000,
                daily_budget_usd=10.0,
            ),
        )
        return httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    return _make


def _payload(stream: bool, model: str = "small") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": stream,
    }


class TestStreaming:
    async def test_streams_content_then_done(self, client_factory: Any) -> None:
        backend = _FakeBackend(chunks=[_content_chunk("Hel"), _content_chunk("lo"), _final_chunk()])

        async with (
            await client_factory(backend) as client,
            client.stream(
                "POST",
                "/chat/completions",
                json=_payload(stream=True),
            ) as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = [line async for line in response.aiter_lines()]

        data_lines = [ln for ln in lines if ln.startswith("data: ")]
        payloads = [ln.removeprefix("data: ") for ln in data_lines]

        # Last payload is [DONE]
        assert payloads[-1] == "[DONE]"

        # First two chunks carry content
        chunk_0 = json.loads(payloads[0])
        chunk_1 = json.loads(payloads[1])
        assert chunk_0["choices"][0]["delta"]["content"] == "Hel"
        assert chunk_1["choices"][0]["delta"]["content"] == "lo"

        # Final usage chunk
        chunk_final = json.loads(payloads[2])
        assert chunk_final["usage"]["prompt_tokens"] == 4
        assert chunk_final["usage"]["completion_tokens"] == 2

    async def test_streams_emits_error_event_on_backend_failure(self, client_factory: Any) -> None:
        """A backend error raised mid-stream must surface as an SSE error event."""
        backend = _FakeBackend(error=BackendUnavailableError("down", backend="small"))

        async with (
            await client_factory(backend) as client,
            client.stream(
                "POST",
                "/chat/completions",
                json=_payload(stream=True),
            ) as response,
        ):
            # HTTP 200 because we already committed to streaming
            assert response.status_code == 200
            lines = [line async for line in response.aiter_lines()]

        data_lines = [ln.removeprefix("data: ") for ln in lines if ln.startswith("data: ")]
        assert data_lines[-1] == "[DONE]"

        error_payload = json.loads(data_lines[-2])
        assert error_payload["error"]["type"] == "backend_unavailable"
        assert error_payload["error"]["backend"] == "small"


class TestNonStreaming:
    async def test_returns_collected_json(self, client_factory: Any) -> None:
        backend = _FakeBackend(chunks=[_content_chunk("Hel"), _content_chunk("lo"), _final_chunk()])

        async with await client_factory(backend) as client:
            response = await client.post("/chat/completions", json=_payload(stream=False))

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "fake-model"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hello"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 4
        assert body["usage"]["completion_tokens"] == 2

    async def test_backend_rate_limit_maps_to_429(self, client_factory: Any) -> None:
        backend = _FakeBackend(error=BackendRateLimitError("slow down", backend="small"))

        async with await client_factory(backend) as client:
            response = await client.post("/chat/completions", json=_payload(stream=False))

        assert response.status_code == 429
        assert response.json()["detail"]["error"]["message"]

    async def test_backend_unavailable_maps_to_502(self, client_factory: Any) -> None:
        backend = _FakeBackend(error=BackendUnavailableError("down", backend="small"))

        async with await client_factory(backend) as client:
            response = await client.post("/chat/completions", json=_payload(stream=False))

        assert response.status_code == 502


class TestRequestValidation:
    async def test_rejects_empty_messages(self, client_factory: Any) -> None:
        backend = _FakeBackend()

        async with await client_factory(backend) as client:
            response = await client.post(
                "/chat/completions",
                json={"model": "small", "messages": []},
            )

        assert response.status_code == 422

    async def test_rejects_temperature_out_of_range(self, client_factory: Any) -> None:
        backend = _FakeBackend()

        async with await client_factory(backend) as client:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": "small",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 5.0,
                },
            )

        assert response.status_code == 422
