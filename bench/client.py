"""Async HTTP client that measures per-request latency precisely.

What makes this different from a generic httpx wrapper:

- TTFT is "time to first non-empty content delta", not "time to
  HTTP 200". For streaming, headers can arrive well before the first
  data chunk; for non-streaming, TTFT == total latency by definition.
- ITL is computed from per-chunk arrival times, not averaged from
  total/count. This catches stuttering that simple averaging hides.
- Each request is fully isolated — no connection pooling across
  benchmark runs.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import httpx


class Outcome(StrEnum):
    SUCCESS = "success"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    CONNECT_ERROR = "connect_error"
    PROTOCOL_ERROR = "protocol_error"


@dataclass(frozen=True, slots=True)
class RequestSample:
    """One request's measurements.

    All times are in seconds.
    """

    outcome: Outcome
    status_code: int | None
    ttft_s: float | None
    """Time from request send to first content delta. For non-streaming"""

    total_latency_s: float
    """Time from request send to last byte received."""

    completion_tokens: int | None
    prompt_tokens: int | None
    inter_token_latencies_s: list[float] = field(default_factory=list)
    """Time between each consecutive content delta."""

    error_message: str | None = None


async def stream_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, object],
    timeout_s: float = 60.0,
) -> RequestSample:
    """Send a streaming chat completion and measure TTFT + ITL.

    Streaming target must produce SSE chunks of the OpenAI.
    """
    start = time.monotonic()
    ttft: float | None = None
    last_chunk_time: float | None = None
    itls: list[float] = []
    completion_tokens: int | None = None
    prompt_tokens: int | None = None

    try:
        async with client.stream(
            "POST", url, json=payload, timeout=timeout_s, headers={"Content-Type": "application/json", "X-Tenant-Id": "benchmark"}
        ) as response:
            if response.status_code != 200:
                await response.aread()
                return RequestSample(
                    outcome=Outcome.HTTP_ERROR,
                    status_code=response.status_code,
                    ttft_s=None,
                    total_latency_s=time.monotonic() - start,
                    completion_tokens=None,
                    prompt_tokens=None,
                    error_message=f"HTTP {response.status_code}",
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Extract content delta if present
                content = _extract_content(chunk)
                if content:
                    now = time.monotonic()
                    if ttft is None:
                        ttft = now - start
                    elif last_chunk_time is not None:
                        itls.append(now - last_chunk_time)
                    last_chunk_time = now

                # Final usage chunk
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")

        total = time.monotonic() - start

        if ttft is None:
            return RequestSample(
                outcome=Outcome.PROTOCOL_ERROR,
                status_code=200,
                ttft_s=None,
                total_latency_s=total,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                error_message="No content deltas received",
            )

        return RequestSample(
            outcome=Outcome.SUCCESS,
            status_code=200,
            ttft_s=ttft,
            total_latency_s=total,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            inter_token_latencies_s=itls,
        )

    except httpx.TimeoutException as e:
        return RequestSample(
            outcome=Outcome.TIMEOUT,
            status_code=None,
            ttft_s=None,
            total_latency_s=time.monotonic() - start,
            completion_tokens=None,
            prompt_tokens=None,
            error_message=str(e),
        )
    except httpx.ConnectError as e:
        return RequestSample(
            outcome=Outcome.CONNECT_ERROR,
            status_code=None,
            ttft_s=None,
            total_latency_s=time.monotonic() - start,
            completion_tokens=None,
            prompt_tokens=None,
            error_message=str(e),
        )


async def blocking_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, object],
    timeout_s: float = 120.0,
) -> RequestSample:
    """Send a non-streaming completion and measure total latency
    """
    start = time.monotonic()
    try:
        response = await client.post(url, json=payload, timeout=timeout_s, headers={"Content-Type": "application/json", "X-Tenant-Id": "benchmark"})
        total = time.monotonic() - start

        if response.status_code != 200:
            return RequestSample(
                outcome=Outcome.HTTP_ERROR,
                status_code=response.status_code,
                ttft_s=None,
                total_latency_s=total,
                completion_tokens=None,
                prompt_tokens=None,
                error_message=f"HTTP {response.status_code}",
            )

        body = response.json()
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        return RequestSample(
            outcome=Outcome.SUCCESS,
            status_code=200,
            ttft_s=total,  # non-streaming: TTFT == total
            total_latency_s=total,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            inter_token_latencies_s=[],
        )

    except httpx.TimeoutException as e:
        return RequestSample(
            outcome=Outcome.TIMEOUT,
            status_code=None,
            ttft_s=None,
            total_latency_s=time.monotonic() - start,
            completion_tokens=None,
            prompt_tokens=None,
            error_message=str(e),
        )
    except httpx.ConnectError as e:
        return RequestSample(
            outcome=Outcome.CONNECT_ERROR,
            status_code=None,
            ttft_s=None,
            total_latency_s=time.monotonic() - start,
            completion_tokens=None,
            prompt_tokens=None,
            error_message=str(e),
        )

def _extract_content(chunk: dict[str, object]) -> str | None:
    """Pull the delta.content from an OpenAI streaming chunk
    """
    choices = chunk.get("choices")
    if not choices or not isinstance(choices, list):
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not delta or not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None
