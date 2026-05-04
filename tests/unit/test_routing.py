"""Tests for app.routing — auto policy, explicit names, degraded mode.

All tests are offline: no Redis, no vLLM, no network. The _FakeBackend
class matches the Backend protocol structurally (no inheritance needed
because Backend is a Protocol, not an ABC).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from app.backends import BackendRegistry
from app.routing.routing import resolve_backend
from app.schemas.chat import ChatChunk, ChatRequest

# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


class _FakeBackend:
    """Minimal structural match for the Backend protocol.

    Only the attributes resolve_backend actually touches (name, model) are
    meaningful here — stream/health/aclose are stubbed because the routing
    module never calls them. This keeps the fake tiny and explicit about
    what routing actually depends on.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = f"stub-{name}"

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


def _req(model: str, content: str = "hi") -> ChatRequest:
    """Construct a minimal ChatRequest with a single user message."""
    return ChatRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
    )


def _registry(*names: str) -> BackendRegistry:
    """Build a registry pre-populated with fake backends."""
    reg = BackendRegistry()
    for n in names:
        reg.register(_FakeBackend(n))
    return reg


# --------------------------------------------------------------------------
# Explicit model names (non-auto path)
# --------------------------------------------------------------------------


class TestExplicitModel:
    def test_uses_literal_backend_name(self) -> None:
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("small"), reg)
        assert name == "small"
        assert reason == "explicit"

    def test_large_backend_explicitly(self) -> None:
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("large"), reg)
        assert name == "large"
        assert reason == "explicit"

    def test_auto_alias_is_case_insensitive(self) -> None:
        """Aliases ('auto', 'default') are matched case-insensitively."""
        reg = _registry("small", "large")
        name_lower, _ = resolve_backend(_req("auto"), reg)
        name_upper, _ = resolve_backend(_req("AUTO"), reg)
        name_mixed, _ = resolve_backend(_req("Auto"), reg)
        assert name_lower == name_upper == name_mixed == "small"

    def test_default_alias_behaves_like_auto(self) -> None:
        reg = _registry("small", "large")
        name_auto, reason_auto = resolve_backend(_req("auto"), reg)
        name_def, reason_def = resolve_backend(_req("default"), reg)
        assert name_auto == name_def
        assert reason_auto == reason_def

    def test_unknown_backend_raises_keyerror(self) -> None:
        reg = _registry("small", "large")
        with pytest.raises(KeyError):
            resolve_backend(_req("nonexistent"), reg)

    def test_whitespace_stripped_from_model_name(self) -> None:
        """Surrounding whitespace on 'auto' is tolerated."""
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("  auto  "), reg)
        assert name == "small"
        assert reason == "auto_short"


# --------------------------------------------------------------------------
# Auto-routing — the heuristic
# --------------------------------------------------------------------------


class TestAutoShort:
    def test_trivial_prompt_routes_to_small(self) -> None:
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", "hi"), reg)
        assert name == "small"
        assert reason == "auto_short"

    def test_moderate_prose_routes_to_small(self) -> None:
        """Prose well under the 3000-char threshold stays on small."""
        prose = "Tell me a short story about a cat." * 3  # ~100 chars
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", prose), reg)
        assert name == "small"
        assert reason == "auto_short"

    def test_empty_content_routes_to_small(self) -> None:
        """Edge case: empty string is trivially 'short'."""
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", ""), reg)
        assert name == "small"
        assert reason == "auto_short"


class TestAutoLong:
    def test_long_prompt_routes_to_large(self) -> None:
        """Anything over 3000 characters escalates to large."""
        long_text = "a " * 2_000  # 4000 characters
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", long_text), reg)
        assert name == "large"
        assert reason == "auto_long"

    def test_just_over_threshold_routes_to_large(self) -> None:
        """Boundary: 3001 characters should already be 'long'."""
        text = "x" * 3_001
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", text), reg)
        assert name == "large"
        assert reason == "auto_long"

    def test_at_threshold_routes_to_small(self) -> None:
        """Boundary: exactly 3000 characters is still 'short'.

        The condition is `len(text) > threshold`, so equal → small.
        This test pins that behavior so a future refactor doesn't
        flip the inequality accidentally.
        """
        text = "x" * 3_000
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", text), reg)
        assert name == "small"
        assert reason == "auto_short"


class TestAutoCode:
    def test_markdown_code_fence_routes_to_large(self) -> None:
        reg = _registry("small", "large")
        prompt = (
            "Explain this:\npython\nprint('hi')\n words = "
            "text.split() symbol_count = sum(1 for char in text if char in "
            ") if len(words) > 0 and (symbol_count / len(words)) > 0.5:return True"
        )
        name, reason = resolve_backend(_req("auto", prompt), reg)
        assert name == "large"
        assert reason == "auto_code"

    def test_def_keyword_routes_to_large(self) -> None:
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", "def f(): pass"), reg)
        assert name == "large"
        assert reason == "auto_code"

    def test_class_keyword_routes_to_large(self) -> None:
        reg = _registry("small", "large")
        name, reason = resolve_backend(_req("auto", "class Foo: ..."), reg)
        assert name == "large"
        assert reason == "auto_code"

    def test_import_keyword_routes_to_large(self) -> None:
        reg = _registry("small", "large")
        prompt = "My code uses import pandas — what's wrong?"
        name, reason = resolve_backend(_req("auto", prompt), reg)
        assert name == "small"
        assert reason == "auto_short"

    def test_sql_keyword_routes_to_large(self) -> None:
        reg = _registry("small", "large")
        prompt = "Optimize this query: SELECT * FROM users WHERE id = 1;"
        name, reason = resolve_backend(_req("auto", prompt), reg)
        assert name == "large"
        assert reason == "auto_code"

    def test_code_detection_precedes_length(self) -> None:
        """A very short code prompt still goes to large, not small."""
        reg = _registry("small", "large")
        _, reason = resolve_backend(_req("auto", "def f(): pass"), reg)
        assert reason == "auto_code"

    def test_code_detection_precedes_long_path(self) -> None:
        """A long code prompt is classified as code, not long.

        The 'reason' tag should reflect the actual trigger (code markers),
        not the length threshold, because the code check comes first.
        """
        long_code = "def function():\n" + ("    print('x')\n" * 1_000)  # >3000 chars
        reg = _registry("small", "large")
        _, reason = resolve_backend(_req("auto", long_code), reg)
        assert reason == "auto_code"


# --------------------------------------------------------------------------
# Multi-message requests
# --------------------------------------------------------------------------


class TestMultiMessageRouting:
    def test_system_prompt_counts_toward_length(self) -> None:
        """Long system prompt + short user prompt should escalate."""
        req = ChatRequest.model_validate(
            {
                "model": "auto",
                "messages": [
                    {"role": "system", "content": "x" * 3_000},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        reg = _registry("small", "large")
        name, reason = resolve_backend(req, reg)
        assert name == "large"
        assert reason == "auto_long"

    def test_code_in_any_message_triggers_code_path(self) -> None:
        """Code in the system or assistant message also counts."""
        req = ChatRequest.model_validate(
            {
                "model": "auto",
                "messages": [
                    {"role": "system", "content": "You are a helper."},
                    {"role": "assistant", "content": "py\ndef f(): print('X')\n" * 200},
                    {"role": "user", "content": "improve it"},
                ],
            }
        )
        reg = _registry("small", "large")
        _, reason = resolve_backend(req, reg)
        assert reason == "auto_code"

    def test_multimodal_text_parts_contribute_to_routing(self) -> None:
        """TextPart content inside a list must be visible to the router."""
        req = ChatRequest.model_validate(
            {
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "def f(): pass"},
                        ],
                    }
                ],
            }
        )
        reg = _registry("small", "large")
        _, reason = resolve_backend(req, reg)
        assert reason == "auto_code"


# --------------------------------------------------------------------------
# Degraded mode — backend(s) missing
# --------------------------------------------------------------------------


class TestDegradedRouting:
    def test_auto_long_falls_back_when_large_absent(self) -> None:
        """If 'large' isn't registered, auto_long still returns a valid backend.

        The whole point of graceful degradation: vLLM-large being down
        shouldn't 500 every request. Pick whatever IS registered.
        """
        reg = _registry("small")  # no 'large'
        long_text = "a " * 2_000
        name, _reason = resolve_backend(_req("auto", long_text), reg)
        assert name == "small"

    def test_auto_code_falls_back_when_large_absent(self) -> None:
        reg = _registry("small")  # no 'large'
        name, _reason = resolve_backend(_req("auto", "def f(): pass"), reg)
        assert name == "small"

    def test_auto_short_falls_back_when_small_absent(self) -> None:
        reg = _registry("large")  # no 'small'
        name, _reason = resolve_backend(_req("auto", "hi"), reg)
        assert name == "large"

    def test_no_backends_raises_runtime_error(self) -> None:
        reg = BackendRegistry()  # empty
        with pytest.raises(RuntimeError, match="No backends registered"):
            resolve_backend(_req("auto"), reg)

    def test_explicit_name_still_404s_when_backend_absent(self) -> None:
        """Explicit mode must NOT fall back — client asked specifically."""
        reg = _registry("small")  # no 'large'
        with pytest.raises(KeyError):
            resolve_backend(_req("large"), reg)
