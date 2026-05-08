"""Shared sentence-transformer embedder.
Design Decision
---
- One Embedder instance is built at startup and shared by the semantic
  cache + jailbreak guardrail.
- The synchronous `model.encode()` call is offloaded to the default
  thread executor so it doesn't block the event loop.
- An in-process LRU cache avoids re-encoding identical prompts (common
  for retry traffic and idempotent tests).
- The tensor dimension is exposed via `dim` so the Redis vector schema
  and guardrail blocklist can be sized correctly.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Async wrapper around a SentenceTransformer with LRU caching."""

    def __init__(
        self,
        model: SentenceTransformer,
        *,
        lru_capacity: int = 512,
    ) -> None:
        self._model = model
        self._lru: OrderedDict[str, list[float]] = OrderedDict()
        self._lru_capacity = lru_capacity
        # Get the dimension by doing a throwaway encode at startup
        probe = model.encode("warmup")
        self._dim = int(probe.shape[0])
        logger.info("Embedder ready: dim=%d, lru_capacity=%d", self._dim, lru_capacity)

    @property
    def dim(self) -> int:
        return self._dim

    async def encode(self, text: str) -> list[float]:
        """Return the embedding for `text`, using the LRU cache when possible."""
        cache_hit = self._lru_get(text)
        if cache_hit is not None:
            return cache_hit

        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, self._encode_sync, text)
        self._lru_put(text, vec)
        return vec

    async def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode. Used by the jailbreak guardrail at startup to
        precompute the blocklist."""
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(None, lambda: self._model.encode(texts).tolist())
        return vecs

    def _encode_sync(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def _lru_get(self, text: str) -> list[float] | None:
        key = self._cache_key(text)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        return None

    def _lru_put(self, text: str, vec: list[float]) -> None:
        key = self._cache_key(text)
        self._lru[key] = vec
        self._lru.move_to_end(key)
        if len(self._lru) > self._lru_capacity:
            self._lru.popitem(last=False)
