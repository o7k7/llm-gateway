from fastapi.params import Depends
from redis.asyncio import Redis

from app.redis.redis_client import get_redis
from app.services.semantic_cache import SemanticCache


def get_semantic_cache(redis: Redis = Depends(get_redis)) -> SemanticCache:
    return SemanticCache(redis)
