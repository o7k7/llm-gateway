"""Pre-flight token estimation using tiktoken.

Why pre-flight estimation
-------------------------
The rate limiter needs a token budget guess BEFORE calling the backend,
otherwise we'd pay for the GPU inference of rejected-after-the-fact requests.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from app.schemas.chat import (
    ChatRequest,
    ContentPart,
    ImagePart,
    TextPart,
    UserMessage,
)

# Per-message overhead for role/name tokens. OpenAI's formula for
# cl100k_base-era models is well-documented at ~4 tokens per message.
_PER_MESSAGE_OVERHEAD = 4

# Trailing priming tokens before the assistant's reply
_PRIMING_OVERHEAD = 3

# Conservative image token estimate
_IMAGE_PART_TOKENS = 85


@lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


class TokenEstimator:
    """Estimates input token count for a ChatRequest."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        _encoding(encoding_name)  # to warmup the cache

    def count(self, req: ChatRequest) -> int:
        """Total estimated input tokens (not counting the model's response)."""
        enc = _encoding(self._encoding_name)
        total = 0
        for m in req.messages:
            total += _PER_MESSAGE_OVERHEAD
            content = m.content
            if isinstance(content, str):
                total += len(enc.encode(content))
            elif isinstance(m, UserMessage) and isinstance(content, list):
                total += self._count_parts(content, enc)
        total += _PRIMING_OVERHEAD
        return total

    def estimate_budget(self, req: ChatRequest, *, default_max_tokens: int) -> int:
        """Convenience: input estimate + max_tokens = total budget to pre-charge.

        Callers use this to decide how many tokens to withdraw from the bucket
        before the request.
        """
        return self.count(req) + (req.max_tokens or default_max_tokens)

    def _count_parts(self, parts: list[ContentPart], enc: tiktoken.Encoding) -> int:
        total = 0
        for p in parts:
            if isinstance(p, TextPart):
                total += len(enc.encode(p.text))
            elif isinstance(p, ImagePart):
                total += _IMAGE_PART_TOKENS
        return total
