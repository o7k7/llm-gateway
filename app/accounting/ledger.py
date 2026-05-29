import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import redis.asyncio as redis
from redis.exceptions import NoScriptError

from app.observability import span

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).parent / "ledger.lua"
_LEDGER_SCRIPT = _SCRIPT_PATH.read_text()
_DEFAULT_TTL_S = 90_000  # ~25h


def _sha(script: str) -> str:
    return hashlib.sha1(script.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """Result of recording a request againts the ledger"""

    under_budget: bool
    total_usd_micros: int
    total_tokens_in: int
    total_tokens_out: int

    @property
    def total_usd(self) -> float:
        return self.total_usd_micros / 1e6


class Ledger:
    """Per-tenant daily cost tracking"""

    def __init__(
        self,
        client: redis.Redis,
        *,
        key_prefix: str = "ledger",
        ttl_s: int = _DEFAULT_TTL_S,
        script: str = _LEDGER_SCRIPT,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_s = ttl_s
        self._script = script
        self._sha = _sha(script)

    async def record(
        self,
        *,
        tenant_id: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        daily_cap_usd: float,
        now_utc: datetime | None = None,
    ):
        if tokens_in < 0 or tokens_out < 0 or cost_usd < 0 or daily_cap_usd < 0:
            raise ValueError("Negative values are not allowed in ledger.records")

        async with span("ledger.record") as s:
            s.set_attribute("gateway.ledger.tenant", tenant_id)
            s.set_attribute("gateway.ledger.tokens_in", tokens_in)
            s.set_attribute("gateway.ledger.tokens_out", tokens_out)
            s.set_attribute("gateway.ledger.cost_usd", cost_usd)

            now = now_utc or datetime.now(UTC)
            key = self._key(tenant_id, now)
            args: list[object] = [
                tokens_in,
                tokens_out,
                round(cost_usd * 1_000_000),
                round(daily_cap_usd * 1_000_000),
                self._ttl_s,
            ]

            try:
                raw = await self._client.evalsha(self._sha, 1, key, *args)
            except NoScriptError:
                logger.info("Ledger script not loaded; falling back to EVAL")
                raw = await self._client.eval(self._script, 1, key, *args)

            under, total_usd, total_in, total_out = raw
            entry = LedgerEntry(
                under_budget=bool(int(under)),
                total_usd_micros=int(total_usd),
                total_tokens_in=int(total_in),
                total_tokens_out=int(total_out),
            )
            s.set_attribute("gateway.ledger.under_budget", entry.under_budget)
            s.set_attribute("gateway.ledger.total_spend_usd", entry.total_usd)
            return entry

    async def current_spend_usd(
        self,
        tenant_id: str,
        *,
        now_utc: datetime | None = None,
    ) -> float:
        now = now_utc if now_utc else datetime.now(UTC)
        key = self._key(tenant_id, now)
        raw = await self._client.hget(key, "usd_micros")

        if raw is None:
            return 0.0
        else:
            try:
                return int(raw) / 1e6
            except (TypeError, ValueError):
                return 0.0

    def _key(self, tenant_id: str, now: datetime) -> str:
        date_str = now.strftime("%Y-%m-%d")
        return f"{self._key_prefix}:{tenant_id}:{date_str}"
