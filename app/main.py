import logging
from contextlib import asynccontextmanager

import litellm
from fastapi import FastAPI, Request
from fastapi_limiter import FastAPILimiter

from app.app_state import AppState
from app.backends import BackendRegistry, LiteLLMBackend, VLLMBackend
from app.config import Config, get_config
from app.core.mini_lm_sentence_transformer import get_model_instance
from app.dependencies import get_app_state
from app.redis.redis_client import dispose_redis, get_redis
from app.routers import chat, chat_v2
from app.services.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = get_config()
    _configure_logging(config)
    _configure_litellm_globals(config)
    backends = _build_backends(config)
    await _probe_backends(backends)
    app.state.app_state = AppState(config=config, backends=backends)

    logger.info("Gateway ready: env=%s backends=%s", config.env, backends.names())

    await get_redis()
    await FastAPILimiter.init(await get_redis())
    model = get_model_instance()
    cache_service = SemanticCache(redis_client=await get_redis(), sentence_transformer=model)

    await cache_service.initialize_cache_index()
    try:
        yield
    finally:
        logger.info("Shutting down gateway")
        await backends.aclose()
        await dispose_redis()


def _configure_logging(config: Config):
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def _configure_litellm_globals(config: Config):
    if config.langfuse_secret_key and config.langfuse_secret_key:
        litellm.callbacks = ["langfuse_otel"]
        logger.info("LiteLLM Langfuse enabled")
    else:
        logger.info("Langfuse keys not set")


def _build_backends(config: Config):
    registry = BackendRegistry()

    registry.register(
        VLLMBackend(
            name="small",
            base_url=str(config.vllm_small_url),
            model=config.vllm_small_model,
        )
    )

    registry.register(
        VLLMBackend(
            name="large",
            base_url=str(config.vllm_large_url),
            model=config.vllm_large_model,
        )
    )

    fallback_key = config.LLM_API_KEY

    if fallback_key:
        registry.register(
            LiteLLMBackend(
                name="fallback",
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL,
                api_key=fallback_key,
            )
        )
        logger.info(
            "LiteLLM Fallback backend registered: %s/%s", config.LLM_PROVIDER, config.LLM_MODEL
        )
    else:
        logger.info("No LLM_API_KEY configured. Skipping fallback backend")

    return registry


async def _probe_backends(backends: BackendRegistry) -> None:
    for backend in backends.all():
        try:
            success = await backend.health()
        except Exception as e:
            logger.exception(e)
            success = False
        logger.info(f"Probe backend: {backend.name} success: {success}")


app = FastAPI(title="LLM Gateway", version="0.2.0", lifespan=lifespan)

app.include_router(chat.chat_router, prefix="/v1")
app.include_router(chat_v2.chat_route_v2, prefix="/v2")


@app.get("/health")
async def health(request: Request):
    state = get_app_state(request)
    backend_status: dict[str, bool] = {}

    for b in state.backends.all():
        try:
            backend_status[b.name] = await b.health()
        except Exception as e:
            backend_status[b.name] = False
            logger.error(e)

    overall_ok = all(backend_status.values()) if backend_status else True
    return {
        "status": "ok" if overall_ok else "fail",
        "env": state.config.env,
        "backends": backend_status,
    }
