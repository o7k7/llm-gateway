from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.chat import (
    ChatRequest,
    ContentPart,
    ImagePart,
    TextPart,
)


class TestContentPartDiscriminator:
    def test_parses_text_part(self) -> None:
        part = TypeAdapter(ContentPart).validate_python({"type": "text", "text": "hi"})
        assert isinstance(part, TextPart)

    def test_parses_image_part(self) -> None:
        part = TypeAdapter(ContentPart).validate_python(
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
        )
        assert isinstance(part, ImagePart)
        assert part.image_url.detail == "auto"

    def test_rejects_unknown_part_type(self) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(ContentPart).validate_python({"type": "audio", "audio": "..."})


class TestChatRequest:
    def _sample_payload(self) -> dict[str, object]:
        return {
            "model": "small",
            "messages": [
                {"role": "system", "content": "Y."},
                {"role": "user", "content": "X"},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def test_round_trip_json(self) -> None:
        payload = self._sample_payload()
        req = ChatRequest.model_validate(payload)

        # Serialize back and re-parse — should be idempotent
        re_serialized = req.model_dump(exclude_none=True)
        re_parsed = ChatRequest.model_validate(re_serialized)
        assert re_parsed == req

    def test_rejects_empty_messages(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest.model_validate({"model": "x", "messages": []})

    def test_allows_vendor_extensions(self) -> None:
        """vLLM-specific fields like `guided_json` must pass through unchanged."""
        payload = self._sample_payload()
        payload["guided_json"] = {"type": "object"}
        payload["repetition_penalty"] = 1.05

        req = ChatRequest.model_validate(payload)
        dumped = req.model_dump(exclude_none=True)
        assert dumped["guided_json"] == {"type": "object"}
        assert dumped["repetition_penalty"] == 1.05

    def test_temperature_bounds(self) -> None:
        payload = self._sample_payload()
        payload["temperature"] = 2.5
        with pytest.raises(ValidationError):
            ChatRequest.model_validate(payload)

    def test_max_tokens_must_be_positive(self) -> None:
        payload = self._sample_payload()
        payload["max_tokens"] = 0
        with pytest.raises(ValidationError):
            ChatRequest.model_validate(payload)

    def test_text_for_routing_plain_strings(self) -> None:
        req = ChatRequest.model_validate(self._sample_payload())
        text = req.text_for_routing()
        assert "X" in text
        assert "Y" in text

    def test_text_for_routing_multimodal_extracts_text_only(self) -> None:
        payload = {
            "model": "small",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ],
        }
        req = ChatRequest.model_validate(payload)
        assert req.text_for_routing() == "Describe this:"

    def test_has_images_detects_image_part(self) -> None:
        payload = {
            "model": "small",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ],
        }
        req = ChatRequest.model_validate(payload)
        assert req.has_images() is True

    def test_has_images_false_for_text_only(self) -> None:
        req = ChatRequest.model_validate(self._sample_payload())
        assert req.has_images() is False
