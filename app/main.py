from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.middleware.pii_middleware import PIIMiddleware
from app.redis.redis_client import init_redis, dispose_redis
from app.routers import chat

@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_redis()
    yield
    await dispose_redis()

app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.add_middleware(PIIMiddleware)

app.include_router(chat.chat_router, prefix="/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}