"""Cache key derivation — tenant-scoped and parameter-aware.

Why parameter-aware
-------------------
Two identical prompts with different temperature/top_p/max_tokens should
not share a cache entry.

Why tenant-scoped
-----------------
Multi-tenant safety. Tenant A's cached response for "what's my balance?"
must never surface for Tenant B. We achieve this at two levels:
1. The semantic index name is per-tenant (`query_cache_idx:<tenant_id>`)
2. The per-entry Redis key prefix is per-tenant (`cache:<tenant_id>:<uuid>`)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.schemas.chat import ChatRequest


@dataclass(frozen=True, slots=True)
class CacheKey:
    """All the bits that identify a unique cached request."""

    tenant_id: str
    model: str
    """The resolved backend model (e.g. 'Qwen/Qwen2.5-7B-Instruct-AWQ')."""

    param_hash: str
    """Deterministic hash of sampling parameters (temperature, top_p, etc.)."""

    def index_name(self) -> str:
        """Redis search index name, per-tenant."""
        return f"query_cache_idx:{self.tenant_id}"

    def doc_prefix(self) -> str:
        """Key prefix for individual cache entries, per-tenant."""
        return f"cache:{self.tenant_id}:"


def cache_key_hash(req: ChatRequest, *, tenant_id: str, model: str) -> CacheKey:
    """Derive a CacheKey from a request.

    Sampling parameters included in the hash:
    - temperature (default 1.0 if None)
    - top_p (default 1.0 if None)
    - max_tokens (default 0 if None — sentinel for "no limit")
    - presence_penalty, frequency_penalty (default 0.0 if None)
    - stop (sorted tuple of strings, or empty tuple)
    """
    params = (
        f"t={req.temperature if req.temperature is not None else 1.0:.4f}"
        f"|p={req.top_p if req.top_p is not None else 1.0:.4f}"
        f"|m={req.max_tokens or 0}"
        f"|pp={req.presence_penalty if req.presence_penalty is not None else 0.0:.4f}"
        f"|fp={req.frequency_penalty if req.frequency_penalty is not None else 0.0:.4f}"
        f"|s={_stop_signature(req.stop)}"
    )
    param_hash = hashlib.blake2b(params.encode("utf-8"), digest_size=8).hexdigest()
    return CacheKey(tenant_id=tenant_id, model=model, param_hash=param_hash)


def _stop_signature(stop: str | list[str] | None) -> str:
    if stop is None:
        return ""
    if isinstance(stop, str):
        return stop
    return "\x1f".join(sorted(stop))  # \x1f is ASCII unit separator
