"""OpenAI-compatible chat schemas using Pydantic v2 discriminated unions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


# Content parts (multimodal-ready, discriminated by `type`)
class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


# Messages (discriminated by `role`)
class SystemMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system"] = "system"
    content: str
    name: str | None = None


class UserMessage(BaseModel):
    """A user turn. Content may be a plain string or a list of multimodal parts."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str | list[ContentPart]
    name: str | None = None


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    name: str | None = None


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


# Request
class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = True


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message] = Field(min_length=1)

    max_tokens: int | None = Field(default=None, ge=1, le=32_768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    stream: bool = False
    stream_options: StreamOptions | None = None

    stop: str | list[str] | None = None
    user: str | None = None

    def text_for_routing(self) -> str:
        """Flatten all message text into one string for routing heuristics."""
        parts: list[str] = []
        for m in self.messages:
            content = m.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(m, UserMessage) and isinstance(content, list):
                for p in content:
                    if isinstance(p, TextPart):
                        parts.append(p.text)
        return "\n".join(parts)

    def has_images(self) -> bool:
        """True if any user message contains an image part."""
        return any(
            isinstance(m, UserMessage)
            and isinstance(m.content, list)
            and any(isinstance(p, ImagePart) for p in m.content)
            for m in self.messages
        )


# Streaming response chunks
class Delta(BaseModel):
    """Incremental update inside a streaming choice."""

    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChoiceChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    delta: Delta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None


class Usage(BaseModel):
    """Token usage. vLLM emits this in the final SSE chunk when `include_usage=True`."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatChunk(BaseModel):
    """A single SSE chunk from a streaming chat completion response."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChoiceChunk] = Field(default_factory=list)
    usage: Usage | None = None
