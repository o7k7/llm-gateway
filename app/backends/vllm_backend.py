"""vLLM backend — talks to a vLLM OpenAI-compatible server over HTTP."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.backends.errors import (
    BackendAuthError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from app.schemas.chat import ChatChunk, ChatRequest

logger = logging.getLogger(__name__)

_SSE_DATA_PREFIX = "data: "
_SSE_DONE_MARKER = "[DONE]"


class VLLMBackend:
    """Async streaming client for a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        max_connections: int = 200,
    ) -> None:
        self.name = name
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=timeout_s,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max(max_connections // 4, 10),
            ),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    # Public API
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        payload = self._build_payload(req)
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                self._raise_for_status(response)
                async for chunk in self._iter_sse_chunks(response):
                    yield chunk
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"vLLM backend {self.name!r} timed out", backend=self.name
            ) from e
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"vLLM backend {self.name!r} unreachable: {e}", backend=self.name
            ) from e
        except httpx.HTTPError as e:
            raise BackendError(
                f"vLLM backend {self.name!r} HTTP error: {e}", backend=self.name
            ) from e

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_payload(self, req: ChatRequest) -> dict[str, Any]:
        """Serialize the request, forcing streaming + usage reporting + our model."""
        payload = req.model_dump(exclude_none=True)
        payload["model"] = self.model
        payload["stream"] = True
        stream_options = payload.get("stream_options") or {}
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map upstream HTTP status to typed BackendError subclasses."""
        status = response.status_code
        if status < 400:
            return
        if status in (401, 403):
            raise BackendAuthError(
                f"vLLM backend {self.name!r} auth failed ({status})", backend=self.name
            )
        if status == 429:
            raise BackendRateLimitError(
                f"vLLM backend {self.name!r} rate-limited upstream", backend=self.name
            )
        if 500 <= status < 600:
            raise BackendUnavailableError(
                f"vLLM backend {self.name!r} returned {status}", backend=self.name
            )
        raise BackendError(f"vLLM backend {self.name!r} returned {status}", backend=self.name)

    async def _iter_sse_chunks(self, response: httpx.Response) -> AsyncIterator[ChatChunk]:
        """Parse the SSE stream into ChatChunks."""
        async for line in response.aiter_lines():
            if not line or not line.startswith(_SSE_DATA_PREFIX):
                continue
            payload = line[len(_SSE_DATA_PREFIX) :].strip()
            if payload == _SSE_DONE_MARKER:
                break
            try:
                yield ChatChunk.model_validate_json(payload)
            except ValueError:
                logger.warning(
                    "vLLM backend %s: failed to parse SSE chunk: %s",
                    self.name,
                    payload[:200],
                )
                continue
