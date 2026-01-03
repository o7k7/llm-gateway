from abc import ABC, abstractmethod
from typing import Any

from app.models.semantic_cache_response import SemanticCacheResponse


class ISemanticCache(ABC):
    @abstractmethod
    async def initialize_cache_index(self):
        pass

    @abstractmethod
    async def process_query(self, query: str, query_vector: list[Any] | None) -> SemanticCacheResponse:
        pass

    @abstractmethod
    async def create_cache_for_query(self, query: str, llm_response: str, query_vector: list[Any] | None) -> SemanticCacheResponse:
        pass