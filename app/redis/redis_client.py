from redis.asyncio import Redis, from_url

from app.config import Config

redis_client: Redis | None = None


def get_redis_url():
    return f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0"


async def init_redis():
    global redis_client
    redis_client = from_url(get_redis_url(), encoding="utf-8", decode_responses=True)
    print("Redis client initialized")


async def dispose_redis():
    global redis_client
    if redis_client:
        await redis_client.close()


async def get_redis():
    if redis_client is None:
        await init_redis()
    return redis_client
