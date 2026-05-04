from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from app.models.llm_response import LLMResponse


@runtime_checkable
class ILiteLLMService(Protocol):
    async def process_query(self, user_query: str) -> LLMResponse:
        pass

    async def process_query_stream(self, user_query: str) -> AsyncGenerator[str, None]:
        pass
