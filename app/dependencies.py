from typing import Annotated

from fastapi import Request
from fastapi.params import Depends
from redis.asyncio import Redis
from sentence_transformers import SentenceTransformer

from app.accounting import Ledger, PricingTable, TokenBucket, TokenEstimator
from app.app_state import AppState
from app.backends import BackendRegistry
from app.config import Config
from app.core.mini_lm_sentence_transformer import get_model_instance
from app.redis.redis_client import get_redis
from app.services.chat_completion_service import ChatCompletionService
from app.services.chat_completion_service_interface import IChatCompletionService
from app.services.lite_llm_service import LiteLLMService
from app.services.lite_llm_service_interface import ILiteLLMService
from app.services.semantic_cache import SemanticCache
from app.services.semantic_cache_interface import ISemanticCache


def get_semantic_cache(
    redis: Annotated[Redis, Depends(get_redis)],
    sentence_transformer: Annotated[SentenceTransformer, Depends(get_model_instance)],
) -> ISemanticCache:
    return SemanticCache(redis, sentence_transformer)


def get_lite_llm() -> ILiteLLMService:
    return LiteLLMService()


def get_chat_completion_service(
    sematic_cache: Annotated[ISemanticCache, Depends(get_semantic_cache)],
    lite_llm: Annotated[ILiteLLMService, Depends(get_lite_llm)],
) -> IChatCompletionService:
    return ChatCompletionService(semantic_cache=sematic_cache, llm_service=lite_llm)


def get_app_state(request: Request) -> AppState:
    state = request.app.state.app_state
    assert isinstance(state, AppState)
    return state


def get_config_dep(state: Annotated[AppState, Depends(get_app_state)]) -> Config:
    return state.config


def get_backends(state: Annotated[AppState, Depends(get_app_state)]) -> BackendRegistry:
    return state.backends


def get_bucket(state: Annotated[AppState, Depends(get_app_state)]) -> TokenBucket:
    return state.bucket


def get_ledger(state: Annotated[AppState, Depends(get_app_state)]) -> Ledger:
    return state.ledger


def get_estimator(state: Annotated[AppState, Depends(get_app_state)]) -> TokenEstimator:
    return state.estimator


def get_pricing(state: Annotated[AppState, Depends(get_app_state)]) -> PricingTable:
    return state.pricing


CurrentBackends = Annotated[BackendRegistry, Depends(get_backends)]
CurrentBucket = Annotated[TokenBucket, Depends(get_bucket)]
CurrentLedger = Annotated[Ledger, Depends(get_ledger)]
CurrentEstimator = Annotated[TokenEstimator, Depends(get_estimator)]
CurrentPricing = Annotated[PricingTable, Depends(get_pricing)]
