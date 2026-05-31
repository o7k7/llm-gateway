"""The fixed prompt used across all benchmark runs."""
from __future__ import annotations

FIXED_PROMPT = (
    "Explain the concept of distributed tracing in a software system. "
    "Cover the key ideas: spans, traces, parent-child relationships, "
    "and how context propagates across service boundaries. "
    "Keep your answer to roughly 200 words."
)

MAX_NEW_TOKENS = 256


def build_chat_request(
    *,
    model: str,
    stream: bool,
    temperature: float = 0.0,
) -> dict[str, object]:
    """Build the OpenAI-style request body for the fixed prompt.

    temperature defaults to 0 for deterministic generation — bench
    runs should be reproducible across invocations of the same target.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": FIXED_PROMPT}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": temperature,
        "stream": stream,
    }
