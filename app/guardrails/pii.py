"""Presidio-based PII detection and redaction guardrail.

Ports the v0.1.0 PIIService into the Guardrail protocol.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from app.guardrails.base import GuardrailOutcome, GuardrailResult
from app.schemas.chat import (
    ChatRequest,
    ContentPart,
    Message,
    SystemMessage,
    TextPart,
    ToolMessage,
    UserMessage,
)
from app.schemas.tenant import Tenant

logger = logging.getLogger(__name__)


class PIIPolicy(StrEnum):
    REDACT = "redact"
    """Replace detected PII with entity-type placeholders (e.g. [EMAIL])."""

    BLOCK = "block"
    """Reject the request if any PII is detected."""


@dataclass(frozen=True, slots=True)
class PIIConfig:
    policy: PIIPolicy = PIIPolicy.REDACT
    min_score: float = 0.5
    """Presidio score threshold: recognizers below this are ignored."""

    entities: tuple[str, ...] = (
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_SSN",
        "IP_ADDRESS",
        "CRYPTO",
    )

    language: str = "en"


class PresidioPIIGuardrail:
    """PII detection using Microsoft Presidio."""

    name = "pii"

    def __init__(
        self,
        *,
        analyzer: AnalyzerEngine,
        config: PIIConfig | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._config = config or PIIConfig()

    async def check(self, req: ChatRequest, tenant: Tenant) -> GuardrailResult:
        """Scan all text content for PII, redact or block per policy."""
        texts_by_ref = list(self._enumerate_text_refs(req.messages))
        if not texts_by_ref:
            return GuardrailResult(outcome=GuardrailOutcome.PASSED, request=req)

        all_findings = await asyncio.gather(*(self._analyze(text) for _, text in texts_by_ref))

        total_entities: list[str] = []
        transformed_any = False
        new_messages: list[Message] = list(req.messages)

        for (ref, original_text), findings in zip(texts_by_ref, all_findings, strict=True):
            if not findings:
                continue

            entity_types = sorted({f.entity_type for f in findings})
            total_entities.extend(entity_types)

            if self._config.policy is PIIPolicy.BLOCK:
                return GuardrailResult(
                    outcome=GuardrailOutcome.BLOCKED,
                    request=req,
                    reason=f"PII detected: {entity_types}",
                    metadata={"entity_types": entity_types},
                )

            redacted = self._redact(original_text, findings)
            transformed_any = True
            new_messages[ref.message_index] = self._replace_text(
                new_messages[ref.message_index], ref, redacted
            )

        if not transformed_any:
            return GuardrailResult(outcome=GuardrailOutcome.PASSED, request=req)

        new_req = req.model_copy(update={"messages": new_messages})
        return GuardrailResult(
            outcome=GuardrailOutcome.TRANSFORMED,
            request=new_req,
            reason=f"Redacted PII entity types: {sorted(set(total_entities))}",
            metadata={
                "entity_types": sorted(set(total_entities)),
                "redaction_count": len(total_entities),
            },
        )

    async def _analyze(self, text: str) -> list[RecognizerResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._analyzer.analyze(
                text=text,
                entities=list(self._config.entities),
                language=self._config.language,
                score_threshold=self._config.min_score,
            ),
        )

    @staticmethod
    def _redact(text: str, findings: Iterable[RecognizerResult]) -> str:
        """Replace each finding's span with [ENTITY_TYPE]"""
        sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)
        result = text
        for f in sorted_findings:
            result = result[: f.start] + f"[{f.entity_type}]" + result[f.end :]
        return result

    @staticmethod
    def _enumerate_text_refs(
        messages: list[Message],
    ) -> Iterable[tuple[_TextRef, str]]:
        """Yield (reference, text) for every text-bearing content in messages.

        For UserMessage with list content, yields one ref per TextPart.
        For other messages, yields one ref for the whole content string.
        """
        for i, m in enumerate(messages):
            if isinstance(m, (SystemMessage, ToolMessage)):
                yield _TextRef(message_index=i, part_index=None), m.content
            elif isinstance(m, UserMessage):
                content = m.content
                if isinstance(content, str):
                    yield _TextRef(message_index=i, part_index=None), content
                else:
                    for j, part in enumerate(content):
                        if isinstance(part, TextPart):
                            yield _TextRef(message_index=i, part_index=j), part.text

    @staticmethod
    def _replace_text(message: Message, ref: _TextRef, new_text: str) -> Message:
        """Return a new message with the referenced text replaced."""
        if ref.part_index is None:
            return message.model_copy(update={"content": new_text})

        assert isinstance(message, UserMessage)
        assert isinstance(message.content, list)
        new_parts: list[ContentPart] = list(message.content)
        original_part = new_parts[ref.part_index]
        assert isinstance(original_part, TextPart)
        new_parts[ref.part_index] = TextPart(text=new_text)
        return message.model_copy(update={"content": new_parts})


@dataclass(frozen=True, slots=True)
class _TextRef:
    message_index: int
    part_index: int | None
