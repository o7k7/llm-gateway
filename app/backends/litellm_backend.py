"""LiteLLM backend — fallback to hosted providers (Anthropic, OpenAI, Groq, ...).

This is the v0.2.0 reshape of the old `LiteLLMService`. It speaks the same
`Backend` Protocol as `VLLMBackend` so the router treats them uniformly.

Use cases:
- Self-hosted vLLM is down → fall back to hosted model
- Model isn't self-hosted (e.g. Claude) → route via LiteLLM
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import litellm
from opentelemetry.trace import Status, StatusCode

from app.backends.errors import (
    BackendAuthError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
)
from app.observability import get_current_span
from app.schemas.chat import ChatChunk, ChatRequest, ChoiceChunk, Delta, Usage

logger = logging.getLogger(__name__)


class LiteLLMBackend:
    """Streams chat completions through LiteLLM (hosted providers)."""

    def __init__(
        self,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str,
    ) -> None:
        self.name = name
        self._provider = provider
        self._model = model
        self._full_model = f"{provider}/{model}"
        self._api_key = api_key

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:  # type: ignore[name-defined]
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        span = get_current_span()
        span.set_attribute("gateway.backend.type", "litellm")
        span.set_attribute("gateway.backend.provider", self._provider)
        span.set_attribute("gateway.backend.model", self._model)

        try:
            response = await litellm.acompletion(
                model=self._full_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                api_key=self._api_key,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                presence_penalty=req.presence_penalty,
                frequency_penalty=req.frequency_penalty,
                stop=req.stop,
                user=req.user,
            )
        except litellm.AuthenticationError as e:
            span.set_status(Status(StatusCode.ERROR, "auth_failed"))
            raise BackendAuthError(
                f"LiteLLM backend {self.name!r}: auth failed", backend=self.name
            ) from e
        except litellm.RateLimitError as e:
            span.set_status(Status(StatusCode.ERROR, "rate_limited"))
            raise BackendRateLimitError(
                f"LiteLLM backend {self.name!r}: upstream rate limited", backend=self.name
            ) from e
        except litellm.Timeout as e:
            span.set_status(Status(StatusCode.ERROR, "timeout"))
            raise BackendTimeoutError(
                f"LiteLLM backend {self.name!r}: timeout", backend=self.name
            ) from e
        except Exception as e:  # pragma: no cover — defensive
            span.set_status(Status(StatusCode.ERROR, str(type(e).__name__)))
            raise BackendError(
                f"LiteLLM backend {self.name!r}: {type(e).__name__}: {e}",
                backend=self.name,
            ) from e

        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        final_usage: Any = None
        ttft_recorded = False

        async for litellm_chunk in response:
            # Capture usage if present (LiteLLM attaches it to the final chunk)
            usage_attr = getattr(litellm_chunk, "usage", None)
            if usage_attr is not None:
                final_usage = usage_attr

            choices = getattr(litellm_chunk, "choices", None) or []
            if not choices:
                continue

            delta = choices[0].delta
            content = getattr(delta, "content", None)

            if not ttft_recorded and content:
                span.add_event("ttft")
                ttft_recorded = True

            yield ChatChunk(
                id=chunk_id,
                created=created,
                model=self._model,
                choices=[
                    ChoiceChunk(
                        index=0,
                        delta=Delta(role=getattr(delta, "role", None), content=content),
                        finish_reason=choices[0].finish_reason,
                    )
                ],
            )

        # Emit a final usage-only chunk if we captured one
        if final_usage is not None:
            span.set_attribute("gen_ai.usage.input_tokens", int(final_usage.prompt_tokens))
            span.set_attribute("gen_ai.usage.output_tokens", int(final_usage.completion_tokens))

            yield ChatChunk(
                id=chunk_id,
                created=created,
                model=self._model,
                choices=[],
                usage=Usage(
                    prompt_tokens=int(final_usage.prompt_tokens),
                    completion_tokens=int(final_usage.completion_tokens),
                    total_tokens=int(final_usage.total_tokens),
                ),
            )

    async def health(self) -> bool:
        """Hosted providers have no health endpoint; assume reachable."""
        return True

    async def aclose(self) -> None:
        """LiteLLM uses a global httpx client; nothing to close."""
        return None
