from abc import ABC, abstractmethod


class IChatCompletionService(ABC):
    @abstractmethod
    async def process_query(self, query: str) -> str:
        pass