from typing import Any, Protocol, runtime_checkable

from app.models.semantic_cache_response import SemanticCacheResponse


@runtime_checkable
class ISemanticCache(Protocol):
    async def initialize_cache_index(self):
        pass

    async def process_query(
        self, query: str, query_vector: list[Any] | None
    ) -> SemanticCacheResponse:
        pass

    async def create_cache_for_query(
        self, query: str, llm_response: str, query_vector: list[Any] | None
    ) -> SemanticCacheResponse:
        pass
