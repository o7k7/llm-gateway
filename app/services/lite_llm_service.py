import logging

import litellm
from fastapi import HTTPException
from litellm import completion

from app.config import config
from app.models.llm_response import LLMResponse
from app.services.lite_llm_service_interface import ILiteLLMService


class LiteLLMService(ILiteLLMService):
    logger = logging.getLogger(__name__)

    async def process_query(self, user_query: str) -> LLMResponse:
        try:
            response = completion(
                model=config.LLM_MODEL,
                messages=[
                    {"content": user_query, "role": "user"}
                ],
                # TODO Current caching implementation doesnt support stream
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