from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class IChatCompletionService(ABC):
    @abstractmethod
    async def process_query(self, query: str, query_vector: list[Any] | None) -> str:
        pass

    @abstractmethod
    async def process_query_stream(self, query: str, query_vector: list[Any] | None) -> AsyncGenerator[str, None]:
        pass