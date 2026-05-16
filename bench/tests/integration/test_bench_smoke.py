"""Smoke test: bench harness against an in-process fake server.

This test exists to catch wire-level integration bugs (does the runner
actually emit JSON? do the load patterns correctly hit the request
function?) without needing real vLLM or HuggingFace infrastructure.

For real numbers against real systems, see Part 3's bench scripts.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from bench.client import blocking_request, stream_request
from bench.load import LoadConfig, run_single_stream
from bench.metrics import aggregate_samples


def _build_streaming_fake_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: dict) -> StreamingResponse:
        async def gen():
            chunk1 = {
                "id": "test",
                "choices": [{"index": 0, "delta": {"content": "hello "}}],
            }
            chunk2 = {
                "id": "test",
                "choices": [{"index": 0, "delta": {"content": "world"}}],
            }
            chunk3 = {
                "id": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
            yield f"data: {json.dumps(chunk1)}\n\n".encode()
            await asyncio.sleep(0.01)  # tiny gap for ITL measurement
            yield f"data: {json.dumps(chunk2)}\n\n".encode()
            await asyncio.sleep(0.01)
            yield f"data: {json.dumps(chunk3)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return app

def _build_blocking_fake_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: dict) -> JSONResponse:
        # Sleep to simulate generation time
        await asyncio.sleep(0.05)
        return JSONResponse(
            {
                "id": "test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )

    return app

@asynccontextmanager
async def _serve(app: FastAPI, port: int):

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    # Wait for the server to actually be listening
    for _ in range(50):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"http://127.0.0.1:{port}/openapi.json", timeout=0.5
                )
                if r.status_code == 200:
                    break
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        await asyncio.sleep(0.05)
    else:
        task.cancel()
        raise RuntimeError(f"Fake server didn't come up on port {port}")

    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)

class TestStreamingClient:
    async def test_stream_request_records_ttft_and_itl(self) -> None:
        app = _build_streaming_fake_app()
        async with _serve(app, port=18001) as url:
            async with httpx.AsyncClient() as client:
                sample = await stream_request(
                    client=client,
                    url=url,
                    payload={"model": "fake", "messages": [], "stream": True},
                )

        assert sample.outcome.value == "success"
        assert sample.status_code == 200
        # TTFT must be set and earlier than total_latency
        assert sample.ttft_s is not None
        assert sample.ttft_s < sample.total_latency_s
        # We had 2 content chunks → 1 ITL gap
        assert len(sample.inter_token_latencies_s) == 1
        # Usage extracted from final chunk
        assert sample.completion_tokens == 2
        assert sample.prompt_tokens == 10

    async def test_stream_request_handles_404(self) -> None:
        app = FastAPI()  # no routes
        async with _serve(app, port=18002) as url:
            async with httpx.AsyncClient() as client:
                sample = await stream_request(
                    client=client,
                    url=url,
                    payload={"model": "fake", "messages": []},
                )

        assert sample.outcome.value == "http_error"
        assert sample.status_code == 404
        assert sample.ttft_s is None


class TestBlockingClient:
    async def test_blocking_request_ttft_equals_total(self) -> None:
        """For non-streaming, TTFT == total_latency by definition."""
        app = _build_blocking_fake_app()
        async with _serve(app, port=18003) as url:
            async with httpx.AsyncClient() as client:
                sample = await blocking_request(
                    client=client,
                    url=url,
                    payload={"model": "fake", "messages": []},
                )

        assert sample.outcome.value == "success"
        assert sample.ttft_s == sample.total_latency_s
        assert sample.completion_tokens == 2

    async def test_blocking_request_no_itls(self) -> None:
        """Non-streaming targets have no per-token timing data."""
        app = _build_blocking_fake_app()
        async with _serve(app, port=18004) as url:
            async with httpx.AsyncClient() as client:
                sample = await blocking_request(
                    client=client,
                    url=url,
                    payload={"model": "fake", "messages": []},
                )

        assert sample.inter_token_latencies_s == []

class TestLoadPatterns:
    async def test_single_stream_runs_for_duration(self) -> None:
        app = _build_streaming_fake_app()
        async with _serve(app, port=18005) as url:
            config = LoadConfig(
                target_url=url,
                payload={"model": "fake", "messages": [], "stream": True},
                request_fn=stream_request,
                concurrency=1,
                duration_s=0.5,
                warmup_s=0.1,
                request_timeout_s=10.0,
            )

            samples = []
            async for sample in run_single_stream(config):
                samples.append(sample)

        # We should have collected SOME samples in 0.5s (each fake
        # request takes ~30ms including the 2 × 10ms ITL gaps)
        assert len(samples) > 0
        assert all(s.outcome.value == "success" for s in samples)

    async def test_aggregate_produces_sensible_metrics(self) -> None:
        app = _build_streaming_fake_app()
        async with _serve(app, port=18006) as url:
            config = LoadConfig(
                target_url=url,
                payload={"model": "fake", "messages": [], "stream": True},
                request_fn=stream_request,
                concurrency=1,
                duration_s=0.5,
                warmup_s=0.1,
                request_timeout_s=10.0,
            )

            samples = []
            wall_start = time.monotonic()
            async for sample in run_single_stream(config):
                samples.append(sample)
            wall_clock_s = time.monotonic() - wall_start

        agg = aggregate_samples(
            samples,
            target="fake",
            pattern="single",
            concurrency=1,
            wall_clock_s=wall_clock_s,
        )

        assert agg.success_rate == 1.0
        assert agg.ttft_p50 is not None
        assert agg.ttft_p50 < agg.total_p50  # streaming property
        assert agg.requests_per_sec > 0

class TestJsonOutput:
    async def test_aggregate_serializes_to_valid_ndjson(self) -> None:
        from bench.metrics import aggregate_to_json_line

        app = _build_streaming_fake_app()
        async with _serve(app, port=18007) as url:
            async with httpx.AsyncClient() as client:
                samples = [
                    await stream_request(
                        client=client,
                        url=url,
                        payload={"model": "fake", "messages": []},
                    )
                    for _ in range(3)
                ]

        agg = aggregate_samples(
            samples,
            target="fake",
            pattern="single",
            concurrency=1,
            wall_clock_s=1.0,
        )
        line = aggregate_to_json_line(agg)

        parsed = json.loads(line)
        assert parsed["target"] == "fake"
        assert parsed["pattern"] == "single"
        assert parsed["sample_count"] == 3
        # No newlines inside (NDJSON requirement)
        assert "\n" not in line

