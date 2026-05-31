from dataclasses import dataclass

import redis.asyncio as aioredis

from app.accounting import Ledger, PricingTable, TokenBucket, TokenEstimator
from app.backends import BackendRegistry
from app.cache import Embedder, SemanticCache
from app.config import Config
from app.guardrails import GuardrailRegistry


@dataclass
class AppState:
    config: Config
    backends: BackendRegistry
    redis: aioredis.Redis
    bucket: TokenBucket
    ledger: Ledger
    estimator: TokenEstimator
    pricing: PricingTable
    guardrails: GuardrailRegistry
    cache: SemanticCache | None
    embedder: Embedder | None  # None when both cache+jailbreak disabled
