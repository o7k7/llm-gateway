from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class IChatCompletionService(ABC):
    @abstractmethod
    async def process_query(self, query: str, query_vector: list[Any] | None) -> str:
        pass

    @abstractmethod
    async def process_query_stream(
        self, query: str, query_vector: list[Any] | None
    ) -> AsyncGenerator[str, None]:
        pass
