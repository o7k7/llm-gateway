import asyncio
import logging

from starlette.requests import Request
from fastapi import HTTPException
from app.core.mini_lm_sentence_transformer import get_model_instance
from app.models.chat_request import ChatRequest


class SemanticSecurityService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.blacklisted_phrases = [
            "fail safe mode",
            "act as a developer",
            "system override",
            "you are a linux terminal",
            "ignore all previous instructions"
        ]
        self.blacklisted_embeddings = None


    async def check_jailbreak(self, chat_request: ChatRequest, request: Request):
        model = get_model_instance()
        if self.blacklisted_embeddings is None:
            loop = asyncio.get_running_loop()
            self.blacklisted_embeddings = await loop.run_in_executor(
                None, lambda: model.encode(self.blacklisted_phrases)
            )

        user_query = chat_request.query

        if user_query:
            loop = asyncio.get_running_loop()
            is_unsafe = await loop.run_in_executor(None, lambda: self._calculate_similarity(model, user_query, request=request))
            if is_unsafe:
                self.logger.warning(f"Jailbreak attempt blocked: {user_query[:50]}...")
                raise HTTPException(
                    status_code=403,
                    detail="Security Violation: Unsafe prompt detected."
                )

        return True

    def _calculate_similarity(self, model, query: str, request: Request) -> bool:
        query_vec = model.encode(query)
        similarities = model.similarity([query_vec], self.blacklisted_embeddings)

        request.state.query_vector = query_vec.tolist()

        return similarities.max().item() > 0.75

semantic_security_service_singleton = SemanticSecurityService()