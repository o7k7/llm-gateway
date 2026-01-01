from fastapi.params import Depends
from redis.asyncio import Redis
from sentence_transformers import SentenceTransformer

from app.redis.redis_client import get_redis
from app.core.mini_lm_sentence_transformer import get_model_instance
from app.services.chat_completion_service import ChatCompletionService
from app.services.chat_completion_service_interface import IChatCompletionService
from app.services.lite_llm_service import LiteLLMService
from app.services.lite_llm_service_interface import ILiteLLMService
from app.services.semantic_cache import SemanticCache
from app.services.semantic_cache_interface import ISemanticCache


def get_semantic_cache(
        redis: Redis = Depends(get_redis),
        sentence_transformer: SentenceTransformer = Depends(get_model_instance)
) -> ISemanticCache:
    return SemanticCache(redis, sentence_transformer)

def get_lite_llm() -> ILiteLLMService:
    return LiteLLMService()

def get_chat_completion_service(
        sematic_cache: ISemanticCache = Depends(get_semantic_cache),
        lite_llm: ILiteLLMService = Depends(get_lite_llm)
) -> IChatCompletionService:
    return ChatCompletionService(semantic_cache=sematic_cache, llm_service=lite_llm)