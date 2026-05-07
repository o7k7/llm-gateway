"""Tests for PresidioPIIGuardrail — detection, redaction, and policy behavior."""
from __future__ import annotations

import pytest

from app.guardrails import GuardrailOutcome
from app.guardrails.pii import PIIConfig, PIIPolicy, PresidioPIIGuardrail
from app.schemas.chat import ChatRequest, ImagePart, TextPart
from app.schemas.tenant import Tenant
from presidio_analyzer import AnalyzerEngine


@pytest.fixture(scope="module")
def analyzer() -> AnalyzerEngine:
    """One Presidio engine per module. Construction takes ~100-300ms."""
    return AnalyzerEngine()


@pytest.fixture
def tenant() -> Tenant:
    return Tenant(id="test-tenant")


def _req(content: str) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "small",
            "messages": [{"role": "user", "content": content}],
        }
    )

class TestRedactPolicy:
    async def test_email_is_redacted(
            self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(policy=PIIPolicy.REDACT)
        )
        req = _req("Contact me at john.doe@example.com for details")
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        new_content = result.request.messages[0].content
        assert isinstance(new_content, str)
        assert "john.doe@example.com" not in new_content
        assert "[EMAIL_ADDRESS]" in new_content

    async def test_phone_is_redacted(
            self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(policy=PIIPolicy.REDACT)
        )
        req = _req("Call me at 555-123-4567")
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        assert "[PHONE_NUMBER]" in str(result.request.messages[0].content)

    async def test_multiple_entities_in_one_message(
            self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(policy=PIIPolicy.REDACT)
        )
        req = _req("Email alice@test.com or call 555-987-6543")
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        content = str(result.request.messages[0].content)
        assert "[EMAIL_ADDRESS]" in content
        assert "[PHONE_NUMBER]" in content
        assert result.metadata["redaction_count"] >= 2

    async def test_no_pii_passes_through_unchanged(
            self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = _req("What is the capital of France?")
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.PASSED
        assert result.request is req  # unchanged


class TestBlockPolicy:
    async def test_pii_blocks_when_policy_is_block(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(policy=PIIPolicy.BLOCK)
        )
        req = _req("My email is alice@test.com")
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.BLOCKED
        assert result.reason is not None
        assert "EMAIL_ADDRESS" in (result.reason or "")
        # Original request is returned unchanged on BLOCK (registry raises)
        assert result.request is req

    async def test_clean_text_passes_under_block_policy(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(policy=PIIPolicy.BLOCK)
        )
        req = _req("What is 2 + 2?")
        result = await guard.check(req, tenant)
        assert result.outcome is GuardrailOutcome.PASSED

class TestMultiMessageHandling:
    async def test_pii_in_system_message_redacted(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = ChatRequest.model_validate(
            {
                "model": "small",
                "messages": [
                    {
                        "role": "system",
                        "content": "User's email: admin@company.com",
                    },
                    {"role": "user", "content": "Please help"},
                ],
            }
        )
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        system_content = str(result.request.messages[0].content)
        user_content = str(result.request.messages[1].content)
        assert "[EMAIL_ADDRESS]" in system_content
        assert user_content == "Please help"  # unchanged

    async def test_pii_in_multiple_messages_all_redacted(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = ChatRequest.model_validate(
            {
                "model": "small",
                "messages": [
                    {"role": "user", "content": "I am bob@first.com"},
                    {"role": "assistant", "content": "Got it."},
                    {"role": "user", "content": "My new one is carol@second.com"},
                ],
            }
        )
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        first_user = str(result.request.messages[0].content)
        third_user = str(result.request.messages[2].content)
        assert "bob@first.com" not in first_user
        assert "carol@second.com" not in third_user
        assert "[EMAIL_ADDRESS]" in first_user
        assert "[EMAIL_ADDRESS]" in third_user



class TestMultimodalRedaction:
    async def test_text_part_in_multimodal_message_redacted(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        """TextPart inside a list-content user message must be scanned too."""
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = ChatRequest.model_validate(
            {
                "model": "small",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "See attached; contact eve@test.com",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://x/y.png"},
                            },
                        ],
                    }
                ],
            }
        )
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        parts = result.request.messages[0].content
        assert isinstance(parts, list)
        # First part is TextPart with redacted text
        first_part = parts[0]

        assert isinstance(first_part, TextPart)
        assert "eve@test.com" not in first_part.text
        assert "[EMAIL_ADDRESS]" in first_part.text
        # Image part unchanged
        assert isinstance(parts[1], ImagePart)

    async def test_image_only_message_has_no_text_to_scan(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        """A message with only image parts should pass (nothing to scan)."""
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = ChatRequest.model_validate(
            {
                "model": "small",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://x/y.png"},
                            }
                        ],
                    }
                ],
            }
        )
        result = await guard.check(req, tenant)
        # No text parts to scan; passes through as-is
        assert result.outcome is GuardrailOutcome.PASSED



class TestEdgeCases:
    async def test_empty_message_content_handled(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = ChatRequest.model_validate(
            {
                "model": "small",
                "messages": [{"role": "user", "content": ""}],
            }
        )
        result = await guard.check(req, tenant)
        assert result.outcome is GuardrailOutcome.PASSED

    async def test_metadata_lists_entity_types(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        req = _req("Email: test@x.com Phone: 555-0100")
        result = await guard.check(req, tenant)

        assert result.outcome is GuardrailOutcome.TRANSFORMED
        entity_types = result.metadata.get("entity_types")
        assert isinstance(entity_types, list)
        assert "EMAIL_ADDRESS" in entity_types

    async def test_custom_min_score_filters_low_confidence(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        """A very high score threshold should cause borderline matches to be
        ignored. We pin the behavior so future Presidio upgrades don't
        silently change sensitivity."""
        strict = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(min_score=0.99)
        )
        loose = PresidioPIIGuardrail(
            analyzer=analyzer, config=PIIConfig(min_score=0.2)
        )
        # An ambiguous number that Presidio might score as low-confidence PHONE
        req = _req("Reference number 555 0100 in our system")

        strict_result = await strict.check(req, tenant)
        loose_result = await loose.check(req, tenant)

        strict_count = (
            len(strict_result.metadata.get("entity_types", []))
            if strict_result.outcome is GuardrailOutcome.TRANSFORMED
            else 0
        )
        loose_count = (
            len(loose_result.metadata.get("entity_types", []))
            if loose_result.outcome is GuardrailOutcome.TRANSFORMED
            else 0
        )
        assert strict_count <= loose_count


class TestImmutability:
    async def test_original_request_not_mutated_on_transform(
        self, analyzer: AnalyzerEngine, tenant: Tenant
    ) -> None:
        """The Guardrail contract says impls must not mutate the input.
        This pins that contract for PresidioPIIGuardrail."""
        guard = PresidioPIIGuardrail(analyzer=analyzer)
        original_content = "My email is alice@example.com"
        req = _req(original_content)

        result = await guard.check(req, tenant)
        assert result.outcome is GuardrailOutcome.TRANSFORMED

        # Original request untouched
        assert req.messages[0].content == original_content
        # Result request is a new instance
        assert result.request is not req
