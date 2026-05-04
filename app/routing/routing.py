"""Request routing.

Clients send `model` as either:
- a logical backend name ("small", "large", "fallback") → used directly
- "auto" → the gateway picks based on prompt features

If routing grows complex enough to need multiple strategies
(classifier-based, cost-aware, LLM-judge),
we'll refactor into a package at that point .
"""

from __future__ import annotations

import logging

from app.backends import BackendRegistry
from app.schemas.chat import ChatRequest
from app.security.code_detection_service import CodeDetectionService

logger = logging.getLogger(__name__)

_AUTO_LONG_CHAR_THRESHOLD = 3_000
"""Prompts longer than this (in characters) escalate to the large backend."""

_AUTO_ALIASES = frozenset({"auto", "default"})
_LARGE_BACKEND = "large"
_SMALL_BACKEND = "small"


def resolve_backend(req: ChatRequest, backends: BackendRegistry) -> tuple[str, str]:
    """Return (backend_name, reason) for a given request.
    Raises:
        KeyError: when the requested model is neither an auto alias nor a
            registered backend name.
        RuntimeError: when no backends are registered (misconfiguration).

    Returns:
        (backend_name, reason_tag) — reason is a short observability tag:
        "explicit", "auto_code", "auto_long", "auto_short".
    """
    requested = req.model.strip().lower()

    if requested in _AUTO_ALIASES:
        return _auto_select(req, backends)

    if requested not in backends:
        raise KeyError(requested)

    return requested, "explicit"


def _auto_select(req: ChatRequest, backends: BackendRegistry) -> tuple[str, str]:
    text = req.text_for_routing()

    if CodeDetectionService().is_code(text):
        return _prefer(_LARGE_BACKEND, backends), "auto_code"

    if len(text) > _AUTO_LONG_CHAR_THRESHOLD:
        return _prefer(_LARGE_BACKEND, backends), "auto_long"

    return _prefer(_SMALL_BACKEND, backends), "auto_short"


def _prefer(preferred: str, backends: BackendRegistry) -> str:
    """Return preferred if registered, else any available backend.
    Graceful degradation: if the preferred backend is down (e.g. vLLM-large
    container failed), auto-routing should not 500 every request. It picks
    another registered backend and logs the fact. The route handler
    records what actually served the request in response headers.
    """
    if preferred in backends:
        return preferred
    available = backends.names()
    if not available:
        raise RuntimeError("No backends registered")
    chosen = available[0]
    logger.warning(
        "Auto-routing preferred %r not registered; falling back to %r",
        preferred,
        chosen,
    )
    return chosen
