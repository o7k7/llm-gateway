from fastapi import APIRouter, Body
from pydantic import BaseModel

chat_router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

class Item(BaseModel):
    name: str

@chat_router.post("/completions")
async def chat(it: Item):
    # TODO Integrate semantic cache service
    return {"status": f"send request to llmlite {it.name}"}