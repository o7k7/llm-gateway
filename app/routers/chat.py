from fastapi import APIRouter

chat_router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

@chat_router.post("/completions")
async def chat():
    return {"status": "send request to llmlite"}