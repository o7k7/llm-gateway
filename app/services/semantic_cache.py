import asyncio
import logging
import uuid
from typing import Any

import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from app.core.mini_lm_sentence_transformer import get_model_instance_tensor_dim
from app.models.semantic_cache_response import SemanticCacheResponse
from app.services.semantic_cache_interface import ISemanticCache


class SemanticCache(ISemanticCache):
    logger = logging.getLogger(__name__)

    def __init__(self, redis_client: Redis, sentence_transformer: SentenceTransformer):
        self.redis_client = redis_client
        self.sentence_transformer = sentence_transformer
        self.vector_dim = get_model_instance_tensor_dim()
        self.index_name = "query_cache_idx"
        self.similarity_threshold = 0.15

    async def initialize_cache_index(self):
        try:
            # check whether if it exists or not
            cache_point = await self.redis_client.ft(self.index_name).info()
            self.logger.info(f"Semantic Cache Index already exists. {cache_point}")
        except Exception as e:
            self.logger.error(f"Creating cache index: {e}")
            self.logger.info("Index not found. Creating new Semantic Cache Index...")
            schema = [
                TextField("response"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                )
            ]

            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            await self.redis_client.ft(self.index_name).create_index(schema, definition=definition)

            self.logger.info("SemanticCache initialized successfully")

    async def _get_embedding(self, text: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sentence_transformer.encode(text).tolist()
        )

    async def process_query(self, query: str, query_vector: list[Any] | None) -> SemanticCacheResponse:
        if query_vector is None:
            query_vector = await self._get_embedding(query)

        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        # https://redis.io/kb/doc/153ae27nuz/how-to-perform-vector-search-and-find-the-semantic-similarity-of-documents-in-python
        q = (
            Query(f"*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("response", "score")
            .dialect(2)
        )

        params = {"vec": query_bytes}

        results = await self.redis_client.ft(self.index_name).search(q, query_params=params)

        if results and results.docs:
            best_match = results.docs[0]
            score = float(best_match.score)

            if score < self.similarity_threshold:
                self.logger.info("Cache HIT")
                return SemanticCacheResponse(source="cache", response=best_match.response)

        self.logger.info("Cache MISS")
        return SemanticCacheResponse(source="cache", response=None)

    async def create_cache_for_query(self, query: str, llm_response: str, query_vector: list[Any] | None):
        try:
            if query_vector is None:
                query_vector = await self._get_embedding(query)
            query_bytes = np.array(query_vector, dtype=np.float32).tobytes()

            key = f"cache:{uuid.uuid4()}"
            await self.redis_client.hset(
                key,
                mapping={
                    "response": llm_response,
                    "embedding": query_bytes,
                },
            )

            await self.redis_client.expire(key, 7200)
        except Exception as e:
            self.logger.error(f"Error creating cache for query: {query}.\n{e}")
