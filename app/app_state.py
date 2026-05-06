from dataclasses import dataclass

import redis.asyncio as aioredis

from app.accounting import Ledger, PricingTable, TokenBucket, TokenEstimator
from app.backends import BackendRegistry
from app.config import Config


@dataclass
class AppState:
    config: Config
    backends: BackendRegistry
    redis: aioredis.Redis
    bucket: TokenBucket
    ledger: Ledger
    estimator: TokenEstimator
    pricing: PricingTable
