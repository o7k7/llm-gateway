import asyncio
from fastapi import HTTPException
from app.core.pii_analyzer import get_analyzer
from app.routers.chat import ChatRequest


class PIIService:
    def __init__(self):
        self.analyzer = get_analyzer()
        self.pii_entities = [
            "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON",
            "CREDIT_CARD", "IBAN_CODE", "CRYPTO"
        ]

    async def check_pii(self, chat_request: ChatRequest):
        content_to_scan = chat_request.query

        if not content_to_scan:
            return True

        loop = asyncio.get_running_loop()
        has_pii = await loop.run_in_executor(
            None,
            lambda: self._contains_pii(content_to_scan)
        )

        if has_pii:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PII_DETECTED",
                    "message": "Your request contains sensitive PII (e.g., Email, Phone, Name)."
                }
            )

        return True

    def _contains_pii(self, data) -> bool:
        if isinstance(data, str):
            results = self.analyzer.analyze(
                text=data,
                language="en",
                entities=self.pii_entities
            )
            if any(r.score > 0.5 for r in results):
                return True

        elif isinstance(data, dict):
            for value in data.values():
                if self._contains_pii(value):
                    return True

        elif isinstance(data, list):
            for item in data:
                if self._contains_pii(item):
                    return True

        return False

pii_service = PIIService()