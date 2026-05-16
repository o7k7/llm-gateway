"""Three load patterns: single-stream, steady, burst."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import httpx

from bench.client import RequestSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadConfig:
    target_url: str
    """Full URL like http://localhost:8000/v1/chat/completions"""

    payload: dict[str, object]
    """The chat request body to send each time."""

    request_fn: Callable[..., Awaitable[RequestSample]]
    """Either bench.client.stream_request or blocking_request."""

    concurrency: int
    duration_s: float
    """Bench duration, after warmup."""

    warmup_s: float = 5.0
    """Time to discard at the start; lets caches/JIT settle."""

    request_timeout_s: float = 120.0


async def run_single_stream(
    config: LoadConfig,
) -> AsyncIterator[RequestSample]:
    """Sends one request at a time, awaiting completion before the next.
    """
    async with httpx.AsyncClient() as client:
        # Warmup
        warmup_end = asyncio.get_event_loop().time() + config.warmup_s
        while asyncio.get_event_loop().time() < warmup_end:
            await config.request_fn(
                client=client,
                url=config.target_url,
                payload=config.payload,
                timeout_s=config.request_timeout_s,
            )

        # Measurement
        end = asyncio.get_event_loop().time() + config.duration_s
        while asyncio.get_event_loop().time() < end:
            sample = await config.request_fn(
                client=client,
                url=config.target_url,
                payload=config.payload,
                timeout_s=config.request_timeout_s,
            )
            yield sample



async def run_steady(config: LoadConfig) -> AsyncIterator[RequestSample]:
    """N workers, each in a tight loop sending requests for `duration_s`.
    """
    queue: asyncio.Queue[RequestSample | None] = asyncio.Queue(maxsize=1024)
    stop_event = asyncio.Event()

    async with httpx.AsyncClient() as client:
        warmup_end = asyncio.get_event_loop().time() + config.warmup_s
        while asyncio.get_event_loop().time() < warmup_end:
            await config.request_fn(
                client=client,
                url=config.target_url,
                payload=config.payload,
                timeout_s=config.request_timeout_s,
            )

        async def worker() -> None:
            while not stop_event.is_set():
                sample = await config.request_fn(
                    client=client,
                    url=config.target_url,
                    payload=config.payload,
                    timeout_s=config.request_timeout_s,
                )
                if not stop_event.is_set():
                    await queue.put(sample)

        workers = [
            asyncio.create_task(worker()) for _ in range(config.concurrency)
        ]

        async def stopper() -> None:
            await asyncio.sleep(config.duration_s)
            stop_event.set()

            await queue.put(None)

        stopper_task = asyncio.create_task(stopper())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            stop_event.set()
            stopper_task.cancel()
            for w in workers:
                w.cancel()
            await asyncio.gather(
                stopper_task, *workers, return_exceptions=True
            )


async def run_burst(config: LoadConfig) -> AsyncIterator[RequestSample]:
    """All `concurrency` requests fire at t=0; we collect responses as
    they arrive.
    """
    async with httpx.AsyncClient() as client:
        # Warmup (single sequential request to get past cold path)
        await config.request_fn(
            client=client,
            url=config.target_url,
            payload=config.payload,
            timeout_s=config.request_timeout_s,
        )

        tasks = [
            asyncio.create_task(
                config.request_fn(
                    client=client,
                    url=config.target_url,
                    payload=config.payload,
                    timeout_s=config.request_timeout_s,
                )
            )
            for _ in range(config.concurrency)
        ]

        for completed in asyncio.as_completed(tasks):
            sample = await completed
            yield sample
