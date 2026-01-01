from abc import ABC, abstractmethod

from app.models.llm_response import LLMResponse


class ILiteLLMService(ABC):
    @abstractmethod
    async def process_query(self, user_query: str) -> LLMResponse:
        pass