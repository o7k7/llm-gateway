import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.semantic_cache import SemanticCache


@pytest.mark.asyncio
async def test_cache_hit():
    mock_redis = AsyncMock()

    mock_redis.ft = MagicMock()

    cached_answer = "Cached Answer"
    mock_doc = MagicMock()
    mock_doc.score = 0.1
    mock_doc.response = cached_answer

    mock_search_result = MagicMock()
    mock_search_result.docs = [mock_doc]

    mock_redis.ft.return_value.search = AsyncMock(return_value=mock_search_result)

    mock_transformer = MagicMock()
    service = SemanticCache(mock_redis, mock_transformer)

    result = await service.process_query("test query", query_vector=None)

    assert result.source == "cache"
    assert result.response == cached_answer