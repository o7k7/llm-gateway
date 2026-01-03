from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.models.llm_response import LLMResponse


class ILiteLLMService(ABC):
    @abstractmethod
    async def process_query(self, user_query: str) -> LLMResponse:
        pass

    @abstractmethod
    async def process_query_stream(self, user_query: str) -> AsyncGenerator[str, None]:
        pass