from openai import BaseModel


class LLMResponse(BaseModel):
    content: str
    role: str = "assistant"
    usage: dict | None = None
    model: str | None = None