"""Async token-bucket rate limiter backed by Redis Lua.

Design - Atomicity
------
* Two Lua scripts: consume (reject if insufficient) and refund (never reject).
* Time is provided by the client, not Redis, so all buckets share a consistent
  clock.

Usage
-----
    bucket = TokenBucket(redis_client, key_prefix="tb")
    result = await bucket.consume(
        tenant_id="foe",
        suffix="tpm",
        capacity=100_000,
        refill_per_sec=100_000 / 60,
        cost=1_234,
    )
    if not result.allowed:
        HTTP.Status(429)
    #  do work, observe actual cost
    await bucket.refund(
        tenant_id="acme",
        suffix="tpm",
        capacity=100_000,
        amount=over_estimate,
    )
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import redis.asyncio as redis
from redis.exceptions import NoScriptError

logger = logging.getLogger(__name__)


_SCRIPT_DIR = Path(__file__).parent
_CONSUME_SCRIPT = (_SCRIPT_DIR / "token_bucket.lua").read_text()
_REFUND_SCRIPT = (_SCRIPT_DIR / "token_bucket_refund.lua").read_text()

_DEFAULT_TTL_MS = 60 * 60 * 1000  # 1 hour idle before the key expires


@dataclass(frozen=True, slots=True)
class BucketResult:
    """Result of a consume operation."""

    allowed: bool
    remaining: int


def _sha(script: str) -> str:
    return hashlib.sha1(script.encode()).hexdigest()


class TokenBucket:
    """Redis Lua token bucket.

    One instance is shared across all tenants; tenant isolation is handled
    by the Redis key (`<prefix>:<tenant_id>:<suffix>`).
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        key_prefix: str = "tb",
        ttl_ms: int = _DEFAULT_TTL_MS,
        consume_script: str = _CONSUME_SCRIPT,
        refund_script: str = _REFUND_SCRIPT,
    ) -> None:
        self._client = client
        self._prefix = key_prefix
        self._ttl_ms = ttl_ms
        self._consume_script = consume_script
        self._refund_script = refund_script
        self._consume_sha = _sha(consume_script)
        self._refund_sha = _sha(refund_script)

    async def consume(
        self,
        *,
        tenant_id: str,
        suffix: str,
        capacity: int,
        refill_per_sec: float,
        cost: int,
        now_ms: int | None = None,
    ) -> BucketResult:
        """Atomically refill + consume. Returns (allowed, remaining).

        Cost must be >= 0. A zero-cost consume is effectively a read (never denied).
        """
        if cost < 0:
            raise ValueError("cost must be non-negative")
        key = self._key(tenant_id, suffix)
        args = [
            capacity,
            refill_per_sec,
            now_ms if now_ms is not None else _now_ms(),
            cost,
            self._ttl_ms,
        ]
        raw = await self._run(self._consume_script, self._consume_sha, key, args)
        allowed_raw, remaining_raw = raw
        return BucketResult(allowed=bool(int(allowed_raw)), remaining=int(remaining_raw))

    async def refund(
        self,
        *,
        tenant_id: str,
        suffix: str,
        capacity: int,
        amount: int,
        now_ms: int | None = None,
    ) -> int:
        """Return unused tokens to the bucket. Returns remaining tokens.

        No-op when amount <= 0. Caps at capacity.
        """
        if amount <= 0:
            return await self._peek_or_capacity(tenant_id, suffix, capacity)
        key = self._key(tenant_id, suffix)
        args = [
            capacity,
            amount,
            now_ms if now_ms is not None else _now_ms(),
            self._ttl_ms,
        ]
        raw = await self._run(self._refund_script, self._refund_sha, key, args)
        return int(raw)

    def _key(self, tenant_id: str, suffix: str) -> str:
        return f"{self._prefix}:{tenant_id}:{suffix}"

    async def _peek_or_capacity(self, tenant_id: str, suffix: str, capacity: int) -> int:
        """Cheap best-effort read used by no-op refunds."""
        try:
            raw = await self._client.hget(self._key(tenant_id, suffix), "tokens")
        except Exception:
            return capacity
        if raw is None:
            return capacity
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return capacity

    async def _run(self, script: str, sha: str, key: str, args: list[object]) -> list[object]:
        """EVALSHA with automatic EVAL fallback on NoScriptError."""
        try:
            return await self._client.evalsha(sha, 1, key, *args)
        except NoScriptError:
            logger.info("Script cache miss; falling back to EVAL and reloading")
            return await self._client.eval(script, 1, key, *args)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
