"""Tests for app.accounting.pricing."""

from __future__ import annotations

import pytest
from app.accounting.pricing import PricingTable
from app.schemas.tenant import Pricing


class TestPricingLookup:
    def test_known_model_returns_exact_entry(self) -> None:
        entry = Pricing(model="m1", input_per_1m=1.0, output_per_1m=2.0)
        table = PricingTable(entries=(entry,))
        assert table.get("m1") is entry

    def test_unknown_model_returns_fallback(self) -> None:
        fallback = Pricing(model="__fb__", input_per_1m=10.0, output_per_1m=30.0)
        table = PricingTable(entries=(), unknown_model_fallback=fallback)
        assert table.get("mystery") is fallback

    def test_unknown_model_warning_is_once_per_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Don't spam logs — warn once per unseen model."""
        table = PricingTable(entries=())
        with caplog.at_level("WARNING"):
            table.get("mystery")
            table.get("mystery")
            table.get("mystery")
        warnings = [r for r in caplog.records if "mystery" in r.message]
        assert len(warnings) == 1

    def test_register_overrides_at_runtime(self) -> None:
        table = PricingTable(entries=())
        table.register(Pricing(model="new", input_per_1m=0.5, output_per_1m=1.5))
        assert table.get("new").input_per_1m == 0.5


class TestCostMath:
    def test_cost_usd_computes_correctly(self) -> None:
        entry = Pricing(model="m1", input_per_1m=0.20, output_per_1m=0.60)
        table = PricingTable(entries=(entry,))
        # 1000 input @ 0.20/1M + 500 output @ 0.60/1M
        # = 0.0002 + 0.0003 = 0.0005
        cost = table.cost_usd("m1", prompt_tokens=1000, completion_tokens=500)
        assert cost == pytest.approx(0.0005)

    def test_zero_tokens_zero_cost(self) -> None:
        entry = Pricing(model="m1", input_per_1m=1.0, output_per_1m=1.0)
        table = PricingTable(entries=(entry,))
        assert table.cost_usd("m1", 0, 0) == 0.0

    def test_fallback_pricing_applied_to_unknown_model(self) -> None:
        fallback = Pricing(model="__fb__", input_per_1m=10.0, output_per_1m=30.0)
        table = PricingTable(entries=(), unknown_model_fallback=fallback)
        # 100 input @ 10/1M + 100 output @ 30/1M = 0.001 + 0.003 = 0.004
        assert table.cost_usd("mystery", 100, 100) == pytest.approx(0.004)
