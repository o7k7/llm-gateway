"""JWT-based tenant resolution.

Two modes:
1. Production (jwt_public_key configured) — requires a valid Bearer token,
   verifies signature + claims, constructs Tenant from the payload.
2. Dev (jwt_public_key absent) — accepts X-Tenant-Id header. Never leave
   this mode on in production; lifespan logs a warning at startup.
"""

from __future__ import annotations

import logging
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Config
from app.dependencies import get_app_state
from app.schemas.tenant import Tenant, TenantLimits

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def resolve_tenant_from_jwt(
    token: str,
    *,
    config: Config,
    default_limits: TenantLimits,
) -> Tenant:
    """Decode and validate a JWT, returning the Tenant it represents.

    Raises:
        HTTPException(401): token missing, malformed, expired, or bad signature.
    """
    public_key = config.jwt_public_key
    if public_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT public key not configured",
        )

    try:
        claims = jwt.decode(
            token,
            public_key.get_secret_value(),
            algorithms=[config.jwt_algorithm],
            audience=config.jwt_audience,
            issuer=config.jwt_issuer,
            options={"require": ["sub", "exp"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Token expired", "type": "auth"}},
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid token", "type": "auth"}},
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    try:
        return Tenant.from_jwt_claims(claims, default_limits=default_limits)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": str(e), "type": "auth"}},
        ) from e


async def get_current_tenant(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> Tenant:
    """FastAPI dependency: resolves the tenant for the current request.

    Production mode (JWT key configured): requires a valid Bearer token.
    Dev mode (no JWT key): accepts X-Tenant-Id header, builds a default
    Tenant with baseline limits.
    """
    state = get_app_state(request)
    config = state.config

    default_limits = TenantLimits(
        requests_per_min=config.default_tenant_rpm,
        tokens_per_min=config.default_tenant_tpm,
        daily_budget_usd=config.default_tenant_daily_budget_usd,
    )

    if config.jwt_public_key is not None:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"message": "Bearer token required", "type": "auth"}},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return resolve_tenant_from_jwt(
            credentials.credentials,
            config=config,
            default_limits=default_limits,
        )

    # Dev mode
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "X-Tenant-Id header required in dev mode",
                    "type": "auth",
                }
            },
        )
    return Tenant(id=x_tenant_id, limits=default_limits)


CurrentTenant = Annotated[Tenant, Depends(get_current_tenant)]
