from fastapi import APIRouter, Depends, Request
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel

from app.dependencies import get_chat_completion_service
from app.services.chat_completion_service_interface import IChatCompletionService

chat_router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

class ChatRequest(BaseModel):
    query: str

@chat_router.post("/completions", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def chat(request: Request, chat_request: ChatRequest, service: IChatCompletionService = Depends(get_chat_completion_service)):
    query_vector = None
    if hasattr(request.state, "query_vector"):
        query_vector = request.state.query_vector

    return await service.process_query(query=chat_request.query, query_vector=query_vector)