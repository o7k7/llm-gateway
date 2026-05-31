import logging
import urllib.parse

from app.config import config
from async_lru import alru_cache
from redis.asyncio import Redis, from_url

logger = logging.getLogger("RedisClient")


def get_redis_url():
    encoded_pwd = urllib.parse.quote_plus(config.REDIS_PASSWORD)
    return f"redis://:{encoded_pwd}@{config.REDIS_HOST}:{config.REDIS_PORT}/0"


@alru_cache(maxsize=1)
async def get_redis() -> Redis:
    """
    Creates and returns a singleton Redis instance.
    The @alru_cache ensures the connection pool is only created once.
    """
    logger.info("Initializing Redis client...")
    client = from_url(get_redis_url(), encoding="utf-8", decode_responses=True, max_connections=100)
    return client


async def dispose_redis():
    """
    Closes the cached Redis client if it has been initialized.
    """
    # .cache_info().hits or .cache_parameters() can be used to check state,
    # but calling the function directly to get the instance is cleanest.
    # Note: We await it to get the singleton, then close it.
    client = await get_redis()
    await client.close()
    get_redis.cache_clear()
    logger.info("Redis client disposed")
