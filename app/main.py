import logging
from contextlib import asynccontextmanager

import litellm
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from presidio_analyzer import AnalyzerEngine
from sentence_transformers import SentenceTransformer

from app.accounting import Ledger, TokenBucket, TokenEstimator, get_pricing_table
from app.app_state import AppState
from app.backends import BackendRegistry, LiteLLMBackend, VLLMBackend
from app.cache import Embedder, SemanticCache
from app.config import Config, get_config
from app.dependencies import get_app_state
from app.guardrails import (
    GuardrailRegistry,
    JailbreakGuardrail,
    PIIConfig,
    PIIPolicy,
    PresidioPIIGuardrail,
)
from app.redis.redis_client import dispose_redis, get_redis
from app.routers import chat, chat_v2

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = get_config()
    _configure_logging(config)
    _configure_litellm_globals(config)
    backends = _build_backends(config)
    await _probe_backends(backends)

    redis_client = await get_redis()

    bucket = TokenBucket(client=redis_client)
    ledger = Ledger(client=redis_client)
    estimator = TokenEstimator(encoding_name=config.tokenizer_encoding_name)
    pricing = get_pricing_table()

    embedder = _build_embedder(config=config) if _needs_embedder(config) else None
    cache = (
        _build_cache(config, redis_client=redis_client, embedder=embedder)
        if config.cache_enabled and embedder is not None
        else None
    )
    guardrails = await _build_guardrails(config=config, embedder=embedder)

    app.state.app_state = AppState(
        config=config,
        backends=backends,
        ledger=ledger,
        estimator=estimator,
        pricing=pricing,
        redis=redis_client,
        bucket=bucket,
        embedder=embedder,
        guardrails=guardrails,
        cache=cache,
    )
    logger.info("Gateway ready: env=%s backends=%s", config.env, backends.names())

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


def _needs_embedder(config: Config) -> bool:
    return config.cache_enabled or config.jailbreak_enabled


def _build_embedder(config: Config) -> Embedder:
    """Load the sentence-transformer model. Done lazily"""
    logger.info("Loading sentence-transformer: %s", config.cache_embedder_model)
    model = SentenceTransformer(config.cache_embedder_model)
    return Embedder(model, lru_capacity=config.cache_embedder_lru_capacity)


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


def _build_cache(config: Config, redis_client: aioredis.Redis, embedder: Embedder) -> SemanticCache:
    return SemanticCache(
        redis_client=redis_client,
        embedder=embedder,
        distance_threshold=config.cache_distance_threshold,
        ttl_s=config.cache_ttl_s,
    )


async def _build_guardrails(config: Config, embedder: Embedder | None) -> GuardrailRegistry:
    """Build guardrails in pipeline order: PII first (so jailbreak sees
    scrubbed text), jailbreak second."""
    registry = GuardrailRegistry()

    if config.pii_enabled:
        logger.info("Loading Presidio AnalyzerEngine (this takes a moment...)")
        analyzer = AnalyzerEngine()
        pii_guardrail = PresidioPIIGuardrail(
            analyzer=analyzer,
            config=PIIConfig(
                policy=PIIPolicy(config.pii_policy),
                min_score=config.pii_min_score,
                entities=tuple(config.pii_entities),
            ),
        )
        registry.register(pii_guardrail)

    if config.jailbreak_enabled:
        if embedder is None:
            logger.warning("Jailbreak enabled but no embedder available; skipping")
        else:
            jailbreak_guardrail = JailbreakGuardrail(
                embedder=embedder,
                phrases=tuple(config.jailbreak_phrases),
                similarity_threshold=config.jailbreak_similarity_threshold,
            )
            registry.register(jailbreak_guardrail)

    return registry


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
