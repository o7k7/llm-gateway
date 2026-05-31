"""Backend abstractions for LLM inference.

All backends implement the `Backend` Protocol in `base.py` and yield
`ChatChunk`s from `app.schemas.chat`. Callers never import a concrete
backend — they resolve by name from the `BackendRegistry`.
"""

from __future__ import annotations

from app.backends.base import Backend
from app.backends.errors import (
    BackendAuthError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from app.backends.litellm_backend import LiteLLMBackend
from app.backends.registry import BackendRegistry
from app.backends.vllm_backend import VLLMBackend

__all__ = [
    "Backend",
    "BackendAuthError",
    "BackendError",
    "BackendRateLimitError",
    "BackendRegistry",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "LiteLLMBackend",
    "VLLMBackend",
]
