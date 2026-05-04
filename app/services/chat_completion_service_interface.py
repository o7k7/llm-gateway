from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IChatCompletionService(Protocol):
    async def process_query(self, query: str, query_vector: list[Any] | None) -> str:
        pass

    async def process_query_stream(
        self, query: str, query_vector: list[Any] | None
    ) -> AsyncGenerator[str, None]:
        pass
