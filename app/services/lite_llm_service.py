import logging
from typing import AsyncGenerator

import litellm
from fastapi import HTTPException
from litellm import acompletion

from app.config import config
from app.core.configure_llm_environment import configure_llm_environment
from app.models.llm_response import LLMResponse
from app.services.lite_llm_service_interface import ILiteLLMService

litellm.callbacks = ["langfuse_otel"]

class LiteLLMService(ILiteLLMService):
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.provider, self.model_name = configure_llm_environment()

    async def process_query(self, user_query: str) -> LLMResponse:
        try:
            response = await acompletion(
                model=f"{self.provider}/{self.model_name}",
                messages=[
                    {"content": user_query, "role": "user"}
                ],
                stream=False
            )

            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model
            )
        except litellm.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=500, detail="LLM Provider Auth Failed")
        except litellm.RateLimitError as e:
            self.logger.error(f"Rate limit error: {e}")
            raise HTTPException(status_code=429, detail="LLM Rate Limit Exceeded")
        except Exception as e:
            self.logger.error(f"Unknown error: {e}")
            raise HTTPException(status_code=500, detail="Internal LLM Error")

    async def process_query_stream(self, user_query: str) -> AsyncGenerator[str, None]:
        try:
            response = await litellm.acompletion(
                model=f"{self.provider}/{self.model_name}",
                messages=[{"content": user_query, "role": "user"}],
                stream=True
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except litellm.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=500, detail="LLM Provider Auth Failed")
        except litellm.RateLimitError as e:
            self.logger.error(f"Rate limit error: {e}")
            raise HTTPException(status_code=429, detail="LLM Rate Limit Exceeded")
        except Exception as e:
            self.logger.error(f"Unknown error: {e}")
            raise HTTPException(status_code=500, detail="Internal LLM Error")