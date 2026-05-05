"""Integration tests for TokenBucket against an in-process fake Redis."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
import redis.asyncio as aioredis
from app.accounting.token_bucket import TokenBucket


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    try:
        await client.flushall()
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def bucket(redis_client: aioredis.Redis) -> TokenBucket:
    return TokenBucket(redis_client, ttl_ms=3_600_000)


class TestConsume:
    async def test_first_consume_succeeds_with_full_bucket(self, bucket: TokenBucket) -> None:
        result = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=30,
        )
        assert result.allowed is True
        assert result.remaining == 70

    async def test_sequential_consumes_deplete_bucket(self, bucket: TokenBucket) -> None:
        """Multiple consumes in quick succession should deduct cumulatively."""
        # Freeze "now" so refill doesn't add tokens between calls
        now = 1_000_000_000_000
        r1 = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=30,
            now_ms=now,
        )
        r2 = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=40,
            now_ms=now,
        )
        r3 = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=25,
            now_ms=now,
        )
        assert r1.allowed and r1.remaining == 70
        assert r2.allowed and r2.remaining == 30
        assert r3.allowed and r3.remaining == 5

    async def test_insufficient_tokens_denied_without_deduction(self, bucket: TokenBucket) -> None:
        """When tokens < cost, the bucket is left unchanged."""
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=80,
            now_ms=now,
        )
        denied = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=50,
            now_ms=now,
        )
        assert denied.allowed is False
        assert denied.remaining == 20  # unchanged from previous state

    async def test_exact_cost_equal_to_remaining_is_allowed(self, bucket: TokenBucket) -> None:
        """Boundary: tokens == cost is allowed (the >= comparison in Lua)."""
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=80,
            now_ms=now,
        )
        # Remaining is 20; cost 20 should be allowed
        result = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=20,
            now_ms=now,
        )
        assert result.allowed is True
        assert result.remaining == 0

    async def test_zero_cost_consume_is_allowed_and_readlike(self, bucket: TokenBucket) -> None:
        """A zero-cost consume acts as a read without deducting."""
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=40,
            now_ms=now,
        )
        probe = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=0,
            now_ms=now,
        )
        assert probe.allowed is True
        assert probe.remaining == 60

    async def test_negative_cost_raises_value_error(self, bucket: TokenBucket) -> None:
        """Defensive: negative costs must never reach Redis."""
        with pytest.raises(ValueError):
            await bucket.consume(
                tenant_id="t1",
                suffix="tpm",
                capacity=100,
                refill_per_sec=0,
                cost=-5,
            )


class TestRefill:
    async def test_refill_accumulates_over_elapsed_time(self, bucket: TokenBucket) -> None:
        """After 1s at 10 tok/s, 10 tokens should be replenished."""
        t0 = 1_000_000_000_000
        # Drain to 0
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=100,
            now_ms=t0,
        )
        # 1 second later
        t1 = t0 + 1_000
        result = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=5,
            now_ms=t1,
        )
        assert result.allowed is True
        # Started at 0, refilled ~10, consumed 5 → ~5 remaining
        assert 4 <= result.remaining <= 6

    async def test_refill_caps_at_capacity(self, bucket: TokenBucket) -> None:
        """After a long idle, the bucket refills only up to capacity."""
        t0 = 1_000_000_000_000
        # Drain to 0
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=100,
            now_ms=t0,
        )
        # Wait 60 seconds — would refill 600 tokens if uncapped
        t1 = t0 + 60_000
        result = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=0,
            now_ms=t1,
        )
        assert result.remaining == 100

    async def test_clock_going_backwards_does_not_refund(self, bucket: TokenBucket) -> None:
        """Defensive: if the client clock jumps backwards, no negative refill."""
        t0 = 1_000_000_000_000
        r1 = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=50,
            now_ms=t0,
        )
        assert r1.remaining == 50

        # Now send an "earlier" timestamp
        r2 = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=10,
            cost=10,
            now_ms=t0 - 60_000,
        )
        # No refill (elapsed_ms <= 0 branch), just the deduct
        assert r2.allowed is True
        assert r2.remaining == 40

    async def test_fractional_refill_rates_work(self, bucket: TokenBucket) -> None:
        """A TPM of 100000 → ~1666.67 tokens/sec; must not integer-truncate."""
        t0 = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100_000,
            refill_per_sec=100_000 / 60,
            cost=100_000,
            now_ms=t0,
        )
        # 100ms later → should refill ~166 tokens
        t1 = t0 + 100
        result = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100_000,
            refill_per_sec=100_000 / 60,
            cost=150,
            now_ms=t1,
        )
        assert result.allowed is True
        # Around 16 remaining
        assert 0 <= result.remaining <= 50


class TestTenantIsolation:
    async def test_one_tenant_draining_does_not_affect_another(self, bucket: TokenBucket) -> None:
        now = 1_000_000_000_000
        # Drain tenant A
        await bucket.consume(
            tenant_id="a",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=100,
            now_ms=now,
        )
        # Tenant B still has a full bucket
        result = await bucket.consume(
            tenant_id="b",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=100,
            now_ms=now,
        )
        assert result.allowed is True
        assert result.remaining == 0

    async def test_different_suffixes_are_independent_buckets(self, bucket: TokenBucket) -> None:
        """Per-tenant RPM vs TPM live in separate keys."""
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="a",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=100,
            now_ms=now,
        )
        result = await bucket.consume(
            tenant_id="a",
            suffix="rpm",
            capacity=60,
            refill_per_sec=0,
            cost=1,
            now_ms=now,
        )
        assert result.allowed is True
        assert result.remaining == 59


class TestRefund:
    async def test_refund_returns_tokens_to_bucket(self, bucket: TokenBucket) -> None:
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=80,
            now_ms=now,
        )
        remaining = await bucket.refund(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            amount=50,
            now_ms=now,
        )
        assert remaining == 70  # had 20, refunded 50

    async def test_refund_caps_at_capacity(self, bucket: TokenBucket) -> None:
        """Refunding more than can fit clamps to capacity."""
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=10,
            now_ms=now,
        )
        remaining = await bucket.refund(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            amount=500,
            now_ms=now,
        )
        assert remaining == 100

    async def test_zero_amount_refund_is_noop(self, bucket: TokenBucket) -> None:
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=40,
            now_ms=now,
        )
        remaining = await bucket.refund(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            amount=0,
            now_ms=now,
        )
        assert remaining == 60

    async def test_negative_amount_refund_clamped_to_zero(self, bucket: TokenBucket) -> None:
        """Caller shouldn't pass negative, but if they do it's a noop."""
        now = 1_000_000_000_000
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=40,
            now_ms=now,
        )
        remaining = await bucket.refund(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            amount=-20,
            now_ms=now,
        )
        # Short-circuited in Python before touching Redis
        assert remaining == 60

    async def test_refund_on_missing_key_is_noop_returns_capacity(
        self, bucket: TokenBucket
    ) -> None:
        """A refund against a never-existed bucket MUST NOT create it."""
        remaining = await bucket.refund(
            tenant_id="never-existed",
            suffix="tpm",
            capacity=100,
            amount=50,
        )
        assert remaining == 100


class TestScriptReload:
    async def test_evalsha_falls_back_to_eval_after_flush(
        self, bucket: TokenBucket, redis_client: aioredis.Redis
    ) -> None:
        """After SCRIPT FLUSH, the next call must recover transparently.

        EVALSHA raises NoScriptError when the server-side cache is empty;
        our wrapper should catch that and retry with EVAL.
        """
        # Prime the bucket so the state is well-defined
        await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=30,
        )
        # Flush the Redis script cache
        await redis_client.script_flush()

        # Next call must still succeed (falls back to EVAL)
        result = await bucket.consume(
            tenant_id="t1",
            suffix="tpm",
            capacity=100,
            refill_per_sec=0,
            cost=10,
        )
        assert result.allowed is True
