"""Semantic cache for chat completions.

The cache stores (prompt embedding, response + usage) tuples in Redis,
with tenant isolation via per-tenant indices and param-aware keys
"""

from __future__ import annotations

from app.cache.embedder import Embedder
from app.cache.key import CacheKey, cache_key_hash
from app.cache.semantic import CachedEntry, SemanticCache

__all__ = [
    "CacheKey",
    "CachedEntry",
    "Embedder",
    "SemanticCache",
    "cache_key_hash",
]
