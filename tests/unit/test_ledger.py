"""Integration tests for Ledger against fakeredis."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import pytest
import redis.asyncio as aioredis
from app.accounting.ledger import Ledger


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    try:
        await client.flushall()
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def ledger(redis_client: aioredis.Redis) -> Ledger:
    return Ledger(redis_client)


class TestRecord:
    async def test_first_record_reports_totals(self, ledger: Ledger) -> None:
        entry = await ledger.record(
            tenant_id="t1",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            daily_cap_usd=10.0,
        )
        assert entry.total_tokens_in == 100
        assert entry.total_tokens_out == 50
        assert entry.total_usd_micros == 1_000  # 0.001 * 1_000_000
        assert entry.under_budget is True

    async def test_multiple_records_accumulate(self, ledger: Ledger) -> None:
        for _ in range(3):
            await ledger.record(
                tenant_id="t1",
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.001,
                daily_cap_usd=10.0,
            )
        entry = await ledger.record(
            tenant_id="t1",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            daily_cap_usd=10.0,
        )
        assert entry.total_tokens_in == 400
        assert entry.total_tokens_out == 200
        assert entry.total_usd_micros == 4_000

    async def test_tenants_are_isolated(self, ledger: Ledger) -> None:
        await ledger.record(
            tenant_id="a",
            tokens_in=1000,
            tokens_out=500,
            cost_usd=0.01,
            daily_cap_usd=10.0,
        )
        entry_b = await ledger.record(
            tenant_id="b",
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0001,
            daily_cap_usd=10.0,
        )
        assert entry_b.total_tokens_in == 10

    async def test_negative_values_raise(self, ledger: Ledger) -> None:
        with pytest.raises(ValueError):
            await ledger.record(
                tenant_id="t1",
                tokens_in=-1,
                tokens_out=0,
                cost_usd=0.0,
                daily_cap_usd=10.0,
            )


class TestBudget:
    async def test_under_budget_reported_correctly(self, ledger: Ledger) -> None:
        entry = await ledger.record(
            tenant_id="t1",
            tokens_in=0,
            tokens_out=0,
            cost_usd=5.0,  # well under cap
            daily_cap_usd=10.0,
        )
        assert entry.under_budget is True

    async def test_exceeding_budget_flips_flag(self, ledger: Ledger) -> None:
        # First record: under
        e1 = await ledger.record(
            tenant_id="t1",
            tokens_in=0,
            tokens_out=0,
            cost_usd=9.5,
            daily_cap_usd=10.0,
        )
        assert e1.under_budget is True
        # Second push us over
        e2 = await ledger.record(
            tenant_id="t1",
            tokens_in=0,
            tokens_out=0,
            cost_usd=1.0,
            daily_cap_usd=10.0,
        )
        assert e2.under_budget is False

    async def test_zero_cap_means_unlimited(self, ledger: Ledger) -> None:
        """A daily_cap_usd of 0 means no cap (not 'zero allowed')."""
        entry = await ledger.record(
            tenant_id="t1",
            tokens_in=0,
            tokens_out=0,
            cost_usd=9999.0,
            daily_cap_usd=0.0,
        )
        assert entry.under_budget is True


class TestDailyRollover:
    async def test_different_days_have_separate_keys(self, ledger: Ledger) -> None:
        yesterday = datetime.now(UTC) - timedelta(days=1)
        today = datetime.now(UTC)

        await ledger.record(
            tenant_id="t1",
            tokens_in=1000,
            tokens_out=500,
            cost_usd=1.0,
            daily_cap_usd=10.0,
            now_utc=yesterday,
        )
        today_entry = await ledger.record(
            tenant_id="t1",
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.01,
            daily_cap_usd=10.0,
            now_utc=today,
        )
        # Today's ledger starts fresh
        assert today_entry.total_tokens_in == 10
        assert today_entry.total_usd_micros == 10_000


class TestCurrentSpend:
    async def test_current_spend_before_any_record(self, ledger: Ledger) -> None:
        assert await ledger.current_spend_usd("t1") == 0.0

    async def test_current_spend_after_records(self, ledger: Ledger) -> None:
        await ledger.record(
            tenant_id="t1",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.123456,
            daily_cap_usd=10.0,
        )
        spend = await ledger.current_spend_usd("t1")
        assert spend == pytest.approx(0.123456, abs=1e-6)
