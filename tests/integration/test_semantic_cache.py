"""Integration tests for SemanticCache with real Redis Stack."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis
from app.cache import Embedder, SemanticCache, cache_key_hash
from app.schemas.chat import ChatRequest, Usage
from sentence_transformers import SentenceTransformer
from testcontainers.redis import RedisContainer

# Real Redis Stack via testcontainers


@pytest.fixture(scope="module")
async def redis_stack_url() -> AsyncIterator[str]:
    """Spin up a real Redis Stack container for the module."""
    container = RedisContainer("redis/redis-stack-server:latest")
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"
    finally:
        container.stop()


@pytest.fixture
async def redis_client(redis_stack_url: str) -> AsyncIterator[aioredis.Redis]:
    client = aioredis.from_url(redis_stack_url, decode_responses=False)
    try:
        await client.flushall()
        yield client
    finally:
        await client.aclose()


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Real MiniLM for integration tests. Loads once per module (~1-2s)."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return Embedder(model, lru_capacity=128)


@pytest.fixture
def cache(redis_client: aioredis.Redis, embedder: Embedder) -> SemanticCache:
    return SemanticCache(redis_client, embedder, distance_threshold=0.15, ttl_s=60)


def _req(content: str, **extra: object) -> ChatRequest:
    base: dict[str, object] = {
        "model": "small",
        "messages": [{"role": "user", "content": content}],
    }
    base.update(extra)
    return ChatRequest.model_validate(base)


class TestBasicCache:
    async def test_miss_returns_none_when_empty(self, cache: SemanticCache) -> None:
        req = _req("What is the capital of France?")
        key = cache_key_hash(req, tenant_id="t1", model="stub-model")
        result = await cache.get(key=key, prompt=req.text_for_routing())
        assert result is None

    async def test_put_then_identical_prompt_hits(self, cache: SemanticCache) -> None:
        req = _req("What is the capital of France?")
        key = cache_key_hash(req, tenant_id="t1", model="stub-model")
        await cache.put(
            key=key,
            prompt=req.text_for_routing(),
            response="Paris is the capital of France.",
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )
        result = await cache.get(key=key, prompt=req.text_for_routing())
        assert result is not None
        assert result.content == "Paris is the capital of France."
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10

    async def test_semantically_similar_prompt_hits(self, cache: SemanticCache) -> None:
        """Core value prop: near-matches should hit."""
        req_stored = _req("What is the capital of France?")
        req_lookup = _req("Tell me the capital city of France")

        key_stored = cache_key_hash(req_stored, tenant_id="t1", model="stub-model")
        key_lookup = cache_key_hash(req_lookup, tenant_id="t1", model="stub-model")
        # Keys must be identical for lookup to work (same tenant, same params)
        assert key_stored == key_lookup

        await cache.put(
            key=key_stored,
            prompt=req_stored.text_for_routing(),
            response="Paris.",
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        result = await cache.get(key=key_lookup, prompt=req_lookup.text_for_routing())
        assert result is not None
        assert result.content == "Paris."

    async def test_unrelated_prompt_misses(self, cache: SemanticCache) -> None:
        req_stored = _req("What is the capital of France?")
        req_lookup = _req("Explain quantum entanglement")

        key_stored = cache_key_hash(req_stored, tenant_id="t1", model="stub-model")
        key_lookup = cache_key_hash(req_lookup, tenant_id="t1", model="stub-model")

        await cache.put(
            key=key_stored,
            prompt=req_stored.text_for_routing(),
            response="Paris.",
            usage=None,
        )
        result = await cache.get(key=key_lookup, prompt=req_lookup.text_for_routing())
        assert result is None


class TestTenantIsolation:
    async def test_different_tenants_cannot_share_cache_entries(self, cache: SemanticCache) -> None:
        """Tenant A's cached answer must never surface for Tenant B."""
        prompt = "What is my account balance?"
        req = _req(prompt)

        key_a = cache_key_hash(req, tenant_id="tenant-a", model="stub-model")
        key_b = cache_key_hash(req, tenant_id="tenant-b", model="stub-model")

        await cache.put(key=key_a, prompt=prompt, response="Tenant A's balance: $5000", usage=None)

        # Tenant B asks the identical question — must miss
        result_b = await cache.get(key=key_b, prompt=prompt)
        assert result_b is None

        # Tenant A still gets their entry
        result_a = await cache.get(key=key_a, prompt=prompt)
        assert result_a is not None
        assert "Tenant A's balance" in result_a.content

    async def test_index_names_are_per_tenant(self, cache: SemanticCache) -> None:
        req = _req("hi")
        key_a = cache_key_hash(req, tenant_id="t-a", model="m")
        key_b = cache_key_hash(req, tenant_id="t-b", model="m")
        assert key_a.index_name() != key_b.index_name()
        assert key_a.doc_prefix() != key_b.doc_prefix()


class TestParameterAwareness:
    async def test_different_temperatures_are_different_cache_keys(
        self, cache: SemanticCache
    ) -> None:
        prompt = "Write a haiku about rain."
        req_precise = _req(prompt, temperature=0.1)
        req_creative = _req(prompt, temperature=1.5)

        key_p = cache_key_hash(req_precise, tenant_id="t1", model="m")
        key_c = cache_key_hash(req_creative, tenant_id="t1", model="m")
        assert key_p.param_hash != key_c.param_hash

    async def test_different_max_tokens_different_keys(self, cache: SemanticCache) -> None:
        prompt = "Summarize the history of jazz."
        req_short = _req(prompt, max_tokens=50)
        req_long = _req(prompt, max_tokens=500)

        key_s = cache_key_hash(req_short, tenant_id="t1", model="m")
        key_l = cache_key_hash(req_long, tenant_id="t1", model="m")
        assert key_s.param_hash != key_l.param_hash

    async def test_same_prompt_different_temp_isolated_in_cache(self, cache: SemanticCache) -> None:
        """Put with temp=0.1, lookup with temp=1.5 must miss."""
        prompt = "Generate a product name."
        req_t01 = _req(prompt, temperature=0.1)
        req_t15 = _req(prompt, temperature=1.5)

        key_t01 = cache_key_hash(req_t01, tenant_id="t1", model="m")
        key_t15 = cache_key_hash(req_t15, tenant_id="t1", model="m")

        await cache.put(key=key_t01, prompt=prompt, response="FocusPro", usage=None)
        # Identical prompt, different temp → miss
        result = await cache.get(key=key_t15, prompt=prompt)
        assert result is None
