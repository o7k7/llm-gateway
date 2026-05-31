import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline.config import BaselineConfig, get_config

logger = logging.getLogger(__name__)

class _Message(BaseModel):
    role: str
    content: str

class _ChatRequest(BaseModel):
    model: str
    messages: list[_Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False

class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class _Choice(BaseModel):
    index: int
    message: _Message
    finish_reason: str

class _ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[_Choice]
    usage: _Usage

class _BaselineState:
    def __init__(self) -> None:
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.config: BaselineConfig | None = None
        self.generate_lock = asyncio.Lock()

_state = _BaselineState()

@asynccontextmanager
async def _lifespan(app: FastAPI):
    config = get_config()
    _state.config = config

    logger.info("Loading model: %s", config.model_name)
    t0 = time.monotonic()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.dtype]

    _state.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    _state.model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    _state.model.eval()

    load_time_s = time.monotonic() - t0
    logger.info("Model loaded in %.1fs", load_time_s)

    try:
        yield
    finally:
        logger.info("Shutting down baseline")

app = FastAPI(title="HF Baseline", lifespan=_lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(req: _ChatRequest) -> _ChatResponse:
    if _state.model is None or _state.tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    # How many concurrent requests before degradation? In our case there is no scaling
    async with _state.generate_lock:
        return await _generate_one(req)

async def _generate_one(req: _ChatRequest) -> _ChatResponse:
    prompt = _format_chat_prompt(req.messages)
    max_new_tokens = req.max_tokens or _state.config.max_new_tokens

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _run_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=req.temperature or 1.0,
            top_p=req.top_p or 1.0,
        )
    )

    return _ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            _Choice(
                index=0,
                message=_Message(role="Assistant", content=result["text"]),
                finish_reason="stop"
            )
        ],
        usage=_Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        )
    )

def _run_generate(
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Synchronous model.generate(). Runs on the executor thread."""
    assert _state.model is not None
    assert _state.tokenizer is not None

    device = _state.model.device
    inputs = _state.tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    with torch.no_grad():
        output_ids = _state.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0 and temperature != 1.0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=_state.tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0, prompt_tokens:]
    completion_tokens = int(new_token_ids.shape[0])
    text = _state.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }

def _format_chat_prompt(messages: list[_Message]) -> str:
    assert _state.tokenizer is not None
    try:
        return _state.tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat_template
        parts = []
        for m in messages:
            parts.append(f"{m.role}: {m.content}")
        parts.append("assistant:")
        return "\n".join(parts)

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if _state.model is not None else "loading",
        "model": _state.config.model_name if _state.config else None,
    }