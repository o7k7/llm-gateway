import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.pii_analyzer import get_analyzer


class PIIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method not in ['PUT', 'PATCH', 'POST']:
            return await call_next(request)

        content_type = request.headers.get('content-type', "")
        if "application/json" not in content_type:
            return await call_next(request)

        body_bytes = await request.body()

        try:
            body_json = json.loads(body_bytes)
        except json.decoder.JSONDecodeError:
            return await call_next(request)

        if self._contains_pii(body_json):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "PII_DETECTED"
                }
            )

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request._receive = receive
        
        return await call_next(request)

    def _contains_pii(self, body_json) -> bool:
        analyzer = get_analyzer()

        if isinstance(body_json, str):
            results = analyzer.analyze(
                text=body_json,
                language="en",
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "CREDIT_CARD", "IBAN_CODE", "CRYPTO"]
            )
            if any(r.score > 0.5 for r in results):
                return True

        elif isinstance(body_json, dict):
            for value in body_json.values():
                if self._contains_pii(value):
                    return True


        elif isinstance(body_json, list):
            for item in body_json:
                if self._contains_pii(item):
                    return True


        return False