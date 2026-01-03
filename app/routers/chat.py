from fastapi import APIRouter, Depends, Request
from fastapi_limiter.depends import RateLimiter
from starlette.responses import StreamingResponse

from app.dependencies import get_chat_completion_service
from app.models.chat_request import ChatRequest
from app.security.pii_service import pii_service
from app.security.semantic_security_service import semantic_security_service_singleton
from app.services.chat_completion_service_interface import IChatCompletionService

chat_router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

@chat_router.post("/completions", dependencies=[
    Depends(RateLimiter(times=5, seconds=60)),
    Depends(pii_service.check_pii),
    Depends(semantic_security_service_singleton.check_jailbreak),
])
async def chat(request: Request, chat_request: ChatRequest, service: IChatCompletionService = Depends(get_chat_completion_service)):
    query_vector = None
    if hasattr(request.state, "query_vector"):
        query_vector = request.state.query_vector

    return await service.process_query(query=chat_request.query, query_vector=query_vector)

@chat_router.post("/completions/stream", dependencies=[
    Depends(RateLimiter(times=5, seconds=60)),
    Depends(pii_service.check_pii),
    Depends(semantic_security_service_singleton.check_jailbreak),
])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    service: IChatCompletionService = Depends(get_chat_completion_service),
):
    query_vector = None
    if hasattr(request.state, "query_vector"):
        query_vector = request.state.query_vector

    return StreamingResponse(
        service.process_query_stream(query=chat_request.query, query_vector=query_vector),
        media_type="text/event-stream",
    )