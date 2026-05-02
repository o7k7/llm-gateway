"""Tests for VLLMBackend using respx to mock httpx."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from app.backends.vllm_backend import VLLMBackend
from app.schemas.chat import ChatChunk, ChatRequest


def _sse_line(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _done_line() -> bytes:
    return b"data: [DONE]\n\n"


def _content_chunk(text: str, model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ") -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}],
    }


def _final_usage_chunk(prompt: int, completion: int) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _sample_request() -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "small",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": True,
        }
    )


async def _collect(it: AsyncIterator[ChatChunk]) -> list[ChatChunk]:
    return [c async for c in it]


@pytest.fixture
async def backend() -> AsyncIterator[VLLMBackend]:
    b = VLLMBackend(
        name="small",
        base_url="http://vllm-small:8000",
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    )
    try:
        yield b
    finally:
        await b.aclose()


class TestStreamHappyPath:
    async def test_yields_content_then_usage(self, backend: VLLMBackend) -> None:
        body = (
            _sse_line(_content_chunk("Hel"))
            + _sse_line(_content_chunk("lo"))
            + _sse_line(_final_usage_chunk(prompt=5, completion=2))
            + _done_line()
        )
        with respx.mock(base_url="http://vllm-small:8000") as m:
            m.post("/v1/chat/completions").mock(return_value=httpx.Response(200, content=body))
            chunks = await _collect(backend.stream(_sample_request()))

        assert len(chunks) == 3
        assert chunks[0].choices[0].delta.content == "Hel"
        assert chunks[1].choices[0].delta.content == "lo"
        assert chunks[2].usage is not None
        assert chunks[2].usage.prompt_tokens == 5
        assert chunks[2].usage.completion_tokens == 2

    async def test_overrides_model_field_with_backend_model(self, backend: VLLMBackend) -> None:
        """Client sent model='small'; vLLM must receive the real model name."""
        body = _sse_line(_content_chunk("x")) + _done_line()
        with respx.mock(base_url="http://vllm-small:8000") as m:
            route = m.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, content=body)
            )
            await _collect(backend.stream(_sample_request()))

        sent_body = json.loads(route.calls.last.request.content)
        assert sent_body["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert sent_body["stream"] is True
        assert sent_body["stream_options"]["include_usage"] is True

    async def test_forwards_vendor_extensions(self, backend: VLLMBackend) -> None:
        """vLLM-specific fields like guided_json must reach the upstream server."""
        body = _sse_line(_content_chunk("x")) + _done_line()
        req = ChatRequest.model_validate(
            {
                "model": "small",
                "messages": [{"role": "user", "content": "hi"}],
                "guided_json": {"type": "object"},
                "repetition_penalty": 1.05,
            }
        )
        with respx.mock(base_url="http://vllm-small:8000") as m:
            route = m.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, content=body)
            )
            await _collect(backend.stream(req))

        sent = json.loads(route.calls.last.request.content)
        assert sent["guided_json"] == {"type": "object"}
        assert sent["repetition_penalty"] == 1.05

    async def test_skips_malformed_chunks(self, backend: VLLMBackend) -> None:
        """A single un-parseable chunk must not kill the stream."""
        body = (
            _sse_line(_content_chunk("ok1"))
            + b"data: {this is not json}\n\n"
            + _sse_line(_content_chunk("ok2"))
            + _done_line()
        )
        with respx.mock(base_url="http://vllm-small:8000") as m:
            m.post("/v1/chat/completions").mock(return_value=httpx.Response(200, content=body))
            chunks = await _collect(backend.stream(_sample_request()))

        # Bad chunk is skipped; good chunks still flow
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "ok1"
        assert chunks[1].choices[0].delta.content == "ok2"

    async def test_ignores_non_data_sse_lines(self, backend: VLLMBackend) -> None:
        """SSE comments (':...') and event/id lines must be ignored without errors."""
        body = (
            b": ping\n\n" + b"event: message\n\n" + _sse_line(_content_chunk("hi")) + _done_line()
        )
        with respx.mock(base_url="http://vllm-small:8000") as m:
            m.post("/v1/chat/completions").mock(return_value=httpx.Response(200, content=body))
            chunks = await _collect(backend.stream(_sample_request()))

        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "hi"

    async def test_done_marker_ends_stream_early(self, backend: VLLMBackend) -> None:
        """Anything after [DONE] must not be yielded."""
        body = (
            _sse_line(_content_chunk("before")) + _done_line() + _sse_line(_content_chunk("after"))
        )
        with respx.mock(base_url="http://vllm-small:8000") as m:
            m.post("/v1/chat/completions").mock(return_value=httpx.Response(200, content=body))
            chunks = await _collect(backend.stream(_sample_request()))

        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "before"
