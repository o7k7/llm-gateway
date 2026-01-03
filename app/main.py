from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter

from app.core.mini_lm_sentence_transformer import get_model_instance
from app.middleware.pii_middleware import PIIMiddleware
from app.redis.redis_client import init_redis, dispose_redis, get_redis
from app.routers import chat
from app.security.semantic_security_middleware import SemanticSecurityMiddleware
from app.services.semantic_cache import SemanticCache


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_redis()

    await FastAPILimiter.init(await get_redis())

    model = get_model_instance()
    cache_service = SemanticCache(redis_client=await get_redis(), sentence_transformer=model)

    await cache_service.initialize_cache_index()
    yield
    await dispose_redis()

app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.add_middleware(SemanticSecurityMiddleware)
app.add_middleware(PIIMiddleware)

app.include_router(chat.chat_router, prefix="/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}