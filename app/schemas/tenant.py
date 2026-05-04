"""Tenant-related schemas.

Used by:
- JWT middleware to populate `CurrentTenant` in request scope
- Token bucket for per-tenant rate limits
- Ledger/pricing for cost accounting
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TenantLimits(BaseModel):
    """Per-tenant rate limits. Applied at middleware time before routing."""

    model_config = ConfigDict(extra="forbid")

    requests_per_min: int = Field(default=60, ge=1)
    tokens_per_min: int = Field(default=100_000, ge=1)
    daily_budget_usd: float = Field(default=10.0, ge=0.0)


class Pricing(BaseModel):
    """Per-model pricing in USD per 1M tokens."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_1m / 1_000_000
            + completion_tokens * self.output_per_1m / 1_000_000
        )


class Tenant(BaseModel):
    """Authenticated tenant extracted from JWT or API key."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str | None = None
    limits: TenantLimits = Field(default_factory=TenantLimits)

    @classmethod
    def from_jwt_claims(
        cls,
        claims: dict[str, object],
        *,
        default_limits: TenantLimits | None = None,
    ) -> Tenant:
        """Construct a Tenant from decoded JWT claims."""
        tenant_id = claims.get("sub")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("JWT missing or invalid 'sub' claim")

        name_raw = claims.get("name")
        name = name_raw if isinstance(name_raw, str) else None

        limits_claim = claims.get("limits")
        if isinstance(limits_claim, dict):
            limits = TenantLimits.model_validate(limits_claim)
        elif default_limits is not None:
            limits = default_limits
        else:
            limits = TenantLimits()

        return cls(id=tenant_id, name=name, limits=limits)
