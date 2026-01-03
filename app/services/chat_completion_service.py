from typing import Any

from app.services.chat_completion_service_interface import IChatCompletionService
from app.services.lite_llm_service_interface import ILiteLLMService
from app.services.semantic_cache_interface import ISemanticCache


class ChatCompletionService(IChatCompletionService):
    def __init__(self, semantic_cache: ISemanticCache, llm_service: ILiteLLMService):
        self.sematic_cache = semantic_cache
        self.llm_service = llm_service

    async def process_query(self, query: str, query_vector: list[Any] | None) -> str:
        cache_response = await self.sematic_cache.process_query(query, query_vector)

        if cache_response.response is not None:
            return cache_response.response
        else:
            llm_result = await self.llm_service.process_query(query)
            llm_response = llm_result.content

            await self.sematic_cache.create_cache_for_query(query, llm_response, query_vector)

            return llm_response