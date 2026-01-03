from abc import ABC, abstractmethod
from typing import Any

class IChatCompletionService(ABC):
    @abstractmethod
    async def process_query(self, query: str, query_vector: list[Any] | None) -> str:
        pass