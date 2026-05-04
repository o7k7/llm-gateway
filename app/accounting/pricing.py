"""Pricing table: per-model USD cost per 1M tokens.

For self-hosted vLLM models, these numbers reflect your amortized cost
(GPU rental, electricity, etc.) spread across expected throughput. For
hosted fallback providers (Groq, GPT), use their published rates.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.schemas.tenant import Pricing

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: tuple[Pricing, ...] = (
    Pricing(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        input_per_1m=0.05,
        output_per_1m=0.15,
    ),
    Pricing(
        model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        input_per_1m=0.08,
        output_per_1m=0.24,
    ),
    # Hosted fallback providers (example — update to current rates).
    Pricing(
        model="llama-3.1-8b-instant",
        input_per_1m=0.05,
        output_per_1m=0.08,
    ),
)


class PricingTable:
    """Resolves a model name to its Pricing entry.

    Model lookup is exact-match on the model identifier the backend reports
    in the usage chunk. If an unknown model shows up, we log once per model
    and fall back to a configurable "unknown model" rate so billing never
    silently zeroes out.
    """

    def __init__(
        self,
        entries: tuple[Pricing, ...] = _DEFAULT_PRICING,
        *,
        unknown_model_fallback: Pricing | None = None,
    ) -> None:
        self._by_model: dict[str, Pricing] = {p.model: p for p in entries}
        self._unknown_fallback = unknown_model_fallback or Pricing(
            model="__unknown__", input_per_1m=1.0, output_per_1m=3.0
        )
        self._warned_models: set[str] = set()

    def get(self, model: str) -> Pricing:
        if model in self._by_model:
            return self._by_model[model]
        if model not in self._warned_models:
            logger.warning("No pricing configured for model %r; using unknown fallback", model)
            self._warned_models.add(model)
        return self._unknown_fallback

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return self.get(model).cost_usd(prompt_tokens, completion_tokens)

    def register(self, pricing: Pricing) -> None:
        """Register or override pricing for a model at runtime."""
        self._by_model[pricing.model] = pricing


@lru_cache
def get_pricing_table() -> PricingTable:
    """Singleton pricing table. Override by calling PricingTable() directly
    in tests or when constructing a custom table."""
    return PricingTable()
