"""Tenant-scoped semantic cache backed by Redis Stack (RediSearch).

Storage layout (per tenant)
---------------------------
Index:  query_cache_idx:<tenant_id>
Keys:   cache:<tenant_id>:<uuid>
Fields: response   (string, the cached LLM response content)
        usage_json (string, JSON-serialized Usage object)
        model      (string, the backend model that produced this)
        param_hash (string, the sampling-parameter signature)
        embedding  (vector, FLOAT32 COSINE FLAT)

Lookup semantics
----------------
- Query by cosine distance; return the best match if distance < threshold
- Additional filter on param_hash: entries cached with different
  sampling params won't match even if the prompt is identical
- The client-provided tenant never sees other tenants' entries because
  the index itself is per-tenant

"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis.asyncio as aioredis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.cache.embedder import Embedder
from app.cache.key import CacheKey
from app.schemas.chat import Usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CachedEntry:
    """A successful cache hit, ready to be returned to the client."""

    content: str
    usage: Usage | None
    model: str


class SemanticCache:
    """Redis-backed semantic cache with per-tenant indices."""

    # Default matches v0.1.0 — a cosine *distance* threshold. Lower = more similar.
    DEFAULT_DISTANCE_THRESHOLD = 0.15

    def __init__(
        self,
        redis_client: aioredis.Redis,
        embedder: Embedder,
        *,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        ttl_s: int = 7200,
    ) -> None:
        self._redis = redis_client
        self._embedder = embedder
        self._threshold = distance_threshold
        self._ttl_s = ttl_s
        self._ensured_indices: set[str] = set()

    # -------- public API ---------------------------------------------------

    async def get(
        self,
        *,
        key: CacheKey,
        prompt: str,
        prompt_vector: list[float] | None = None,
    ) -> CachedEntry | None:
        """Look up a cached response for the given prompt.

        Args:
            key: tenant + param hash context
            prompt: the query text to semantic-match
            prompt_vector: optional pre-computed embedding (reuse if the
                jailbreak guardrail already encoded it)

        Returns:
            CachedEntry if a match is found within threshold, else None.
        """
        await self._ensure_index(key)

        if prompt_vector is None:
            prompt_vector = await self._embedder.encode(prompt)

        vec_bytes = _vec_bytes(prompt_vector)

        # Filter on param_hash so different sampling params never collide
        q = (
            Query(
                f"(@param_hash:{{{key.param_hash}}}) "  # Tenant match
                "=>[KNN 1 @embedding $vec AS score]"
            )
            .sort_by("score")
            .return_fields("response", "usage_json", "model", "score")
            .dialect(2)
        )
        try:
            results = await self._redis.ft(key.index_name()).search(
                q, query_params={"vec": vec_bytes}
            )
        except ResponseError as e:
            logger.warning("Cache search failed for tenant %s: %s", key.tenant_id, e)
            return None

        if not results.docs:
            return None

        best = results.docs[0]
        score = float(best.score)
        if score >= self._threshold:
            logger.debug(
                "Cache MISS (below threshold): tenant=%s score=%.4f",
                key.tenant_id,
                score,
            )
            return None

        logger.info(
            "Cache HIT: tenant=%s score=%.4f model=%s",
            key.tenant_id,
            score,
            best.model,
        )
        return CachedEntry(
            content=best.response,
            usage=_deserialize_usage(best.usage_json),
            model=best.model,
        )

    async def put(
        self,
        *,
        key: CacheKey,
        prompt: str,
        response: str,
        usage: Usage | None,
        prompt_vector: list[float] | None = None,
    ) -> None:
        """Store a completed response in the cache."""
        await self._ensure_index(key)

        if prompt_vector is None:
            prompt_vector = await self._embedder.encode(prompt)

        doc_key = f"{key.doc_prefix()}{uuid.uuid4().hex}"
        try:
            await self._redis.hset(
                doc_key,
                mapping={
                    "response": response,
                    "usage_json": _serialize_usage(usage),
                    "model": key.model,
                    "param_hash": key.param_hash,
                    "embedding": _vec_bytes(prompt_vector),
                },
            )
            await self._redis.expire(doc_key, self._ttl_s)
        except Exception as e:
            # Cache write failures must never break the request
            logger.warning("Cache put failed for tenant %s: %s", key.tenant_id, e)

    async def _ensure_index(self, key: CacheKey) -> None:
        """Lazily create the per-tenant index on first use.

        We create indices on-demand rather than up-front because
        """
        index = key.index_name()
        if index in self._ensured_indices:
            return

        schema = (
            TextField("response"),
            TextField("usage_json"),
            TextField("model"),
            TagField("param_hash"),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._embedder.dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(
            prefix=[key.doc_prefix()],
            index_type=IndexType.HASH,
        )
        try:
            await self._redis.ft(index).create_index(schema, definition=definition)
            logger.info("Created cache index: %s", index)
        except ResponseError as e:
            # "Index already exists" is expected after restarts
            if "already exists" not in str(e).lower():
                logger.warning("Index create failed %s: %s", index, e)
                # Still mark as ensured — further retries won't help
        self._ensured_indices.add(index)


def _vec_bytes(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _serialize_usage(usage: Usage | None) -> str:
    if usage is None:
        return ""
    return json.dumps(usage.model_dump())


def _deserialize_usage(s: Any) -> Usage | None:
    if not s:
        return None
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    try:
        return Usage.model_validate(json.loads(s))
    except Exception:
        return None
