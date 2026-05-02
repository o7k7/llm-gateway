"""The Backend Protocol — the seam between the gateway and any inference engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.schemas.chat import ChatChunk, ChatRequest


@runtime_checkable
class Backend(Protocol):
    """An inference backend that streams OpenAI-shaped chat completions."""

    name: str
    """Logical name used by the router and registry (e.g. 'small', 'large', 'fallback')."""

    model: str
    """The actual model identifier sent to the upstream engine."""

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield chat chunks for the given request. The last chunk must carry usage."""
        ...

    async def health(self) -> bool:
        """Return True if the backend is ready to serve requests."""
        ...

    async def aclose(self) -> None:
        """Release resources (connection pools, clients, etc.)."""
        ...
