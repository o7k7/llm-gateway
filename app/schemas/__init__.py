"""Wire-level schemas shared between gateway clients, backends, and internal services.

All schemas here are OpenAI-compatible by design so that:
- Clients using the openai/litellm SDKs work against us without changes
"""

from __future__ import annotations

from app.schemas.chat import (
    AssistantMessage,
    ChatChunk,
    ChatRequest,
    ChoiceChunk,
    ContentPart,
    Delta,
    ImagePart,
    ImageUrl,
    Message,
    StreamOptions,
    SystemMessage,
    TextPart,
    ToolMessage,
    Usage,
    UserMessage,
)
from app.schemas.tenant import (
    Pricing,
    Tenant,
    TenantLimits,
)

__all__ = [
    "AssistantMessage",
    "ChatChunk",
    "ChatRequest",
    "ChoiceChunk",
    "ContentPart",
    "Delta",
    "ImagePart",
    "ImageUrl",
    "Message",
    "Pricing",
    "StreamOptions",
    "SystemMessage",
    "Tenant",
    "TenantLimits",
    "TextPart",
    "ToolMessage",
    "Usage",
    "UserMessage",
]
