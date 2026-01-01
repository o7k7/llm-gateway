from pydantic import BaseModel


class SemanticCacheResponse(BaseModel):
    source: str
    response: str | None