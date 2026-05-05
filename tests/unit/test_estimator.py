"""Tests for app.accounting.estimator — token counting logic."""
from __future__ import annotations

import pytest
from app.accounting.estimator import TokenEstimator
from app.schemas.chat import ChatRequest


@pytest.fixture(scope="module")
def estimator() -> TokenEstimator:
    return TokenEstimator()


def _req(**extra: object) -> ChatRequest:
    base: dict[str, object] = {
        "model": "small",
        "messages": [{"role": "user", "content": "hello"}],
    }
    base.update(extra)
    return ChatRequest.model_validate(base)


class TestBasicCounting:
    def test_trivial_prompt_positive_and_small(
        self, estimator: TokenEstimator
    ) -> None:
        n = estimator.count(_req())
        # Message overhead + priming + 1 token for "hello" = 4 + 3 + 1 = 8ish
        assert 5 <= n <= 15

    def test_longer_prompt_counts_more(self, estimator: TokenEstimator) -> None:
        short = estimator.count(_req(messages=[{"role": "user", "content": "hi"}]))
        long = estimator.count(
            _req(
                messages=[
                    {"role": "user", "content": "The quick brown fox " * 50}
                ]
            )
        )
        assert long > short * 10

    def test_multi_message_accumulates(self, estimator: TokenEstimator) -> None:
        one_msg = estimator.count(
            _req(messages=[{"role": "user", "content": "test " * 20}])
        )
        three_msgs = estimator.count(
            _req(
                messages=[
                    {"role": "system", "content": "test " * 20},
                    {"role": "user", "content": "test " * 20},
                    {"role": "assistant", "content": "test " * 20},
                ]
            )
        )
        # Roughly 3x plus per-message overhead
        assert three_msgs > one_msg * 2


class TestMultimodal:
    def test_text_parts_counted(self, estimator: TokenEstimator) -> None:
        req = _req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello world"},
                    ],
                }
            ]
        )
        n = estimator.count(req)
        assert n >= 5

    def test_image_parts_add_fixed_estimate(
        self, estimator: TokenEstimator
    ) -> None:
        """An image part should add ~85 tokens on top of text."""
        text_only = estimator.count(
            _req(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "describe"}],
                    }
                ]
            )
        )
        with_image = estimator.count(
            _req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://x/y.png"},
                            },
                        ],
                    }
                ]
            )
        )
        # Image should bump count by ~85
        assert 70 <= (with_image - text_only) <= 100


class TestBudgetEstimate:
    def test_budget_equals_input_plus_max_tokens(
        self, estimator: TokenEstimator
    ) -> None:
        req = _req(max_tokens=256)
        budget = estimator.estimate_budget(req, default_max_tokens=512)
        assert budget == estimator.count(req) + 256

    def test_budget_uses_default_when_max_tokens_absent(
        self, estimator: TokenEstimator
    ) -> None:
        req = _req()  # no max_tokens
        budget = estimator.estimate_budget(req, default_max_tokens=512)
        assert budget == estimator.count(req) + 512
