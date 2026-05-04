"""Tests for app.auth.tenant — JWT decode, dev mode, claim handling."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import jwt as pyjwt
import pytest
from app.auth.tenant import resolve_tenant_from_jwt
from app.schemas.tenant import Tenant, TenantLimits
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from pydantic import SecretStr


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[bytes, bytes]:
    """Generate an RSA keypair once per module for JWT signing."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _config(public_pem: bytes, **overrides: Any) -> Any:
    """Minimal config stub with just the fields resolve_tenant_from_jwt reads."""

    defaults = {
        "jwt_public_key": SecretStr(public_pem.decode()),
        "jwt_algorithm": "RS256",
        "jwt_issuer": None,
        "jwt_audience": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mint(private_pem: bytes, claims: dict[str, object]) -> str:
    return pyjwt.encode(claims, private_pem, algorithm="RS256")


class TestJwtDecode:
    def test_valid_token_returns_tenant(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        priv, pub = rsa_keypair
        token = _mint(
            priv,
            {
                "sub": "acme-corp",
                "name": "Acme Corporation",
                "exp": int(time.time()) + 60,
            },
        )
        tenant = resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert isinstance(tenant, Tenant)
        assert tenant.id == "acme-corp"
        assert tenant.name == "Acme Corporation"

    def test_custom_limits_in_claims_are_applied(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        priv, pub = rsa_keypair
        token = _mint(
            priv,
            {
                "sub": "acme-corp",
                "exp": int(time.time()) + 60,
                "limits": {
                    "requests_per_min": 600,
                    "tokens_per_min": 500_000,
                    "daily_budget_usd": 50.0,
                },
            },
        )
        tenant = resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert tenant.limits.requests_per_min == 600
        assert tenant.limits.daily_budget_usd == 50.0


class TestJwtRejection:
    def test_expired_token_raises_401(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        priv, pub = rsa_keypair
        token = _mint(
            priv,
            {"sub": "acme", "exp": int(time.time()) - 10},  # expired 10s ago
        )
        with pytest.raises(HTTPException) as exc:
            resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert exc.value.status_code == 401
        assert exc.value.headers is not None
        assert exc.value.headers["WWW-Authenticate"] == "Bearer"

    def test_malformed_token_raises_401(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        _, pub = rsa_keypair
        with pytest.raises(HTTPException) as exc:
            resolve_tenant_from_jwt(
                "not-a-real-jwt",
                config=_config(pub),
                default_limits=TenantLimits(),
            )
        assert exc.value.status_code == 401

    def test_bad_signature_raises_401(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        """Token signed with a different key must be rejected."""
        _, pub = rsa_keypair
        # Mint with an unrelated keypair
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other_priv = other_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        token = _mint(other_priv, {"sub": "acme", "exp": int(time.time()) + 60})

        with pytest.raises(HTTPException) as exc:
            resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert exc.value.status_code == 401

    def test_missing_sub_claim_raises_401(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        """PyJWT enforces `require=['sub','exp']` so missing 'sub' rejects at decode."""
        priv, pub = rsa_keypair
        token = _mint(priv, {"exp": int(time.time()) + 60})  # no 'sub'
        with pytest.raises(HTTPException) as exc:
            resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert exc.value.status_code == 401

    def test_empty_sub_claim_raises_401(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        """Even if 'sub' is present but empty, we reject."""
        priv, pub = rsa_keypair
        token = _mint(priv, {"sub": "", "exp": int(time.time()) + 60})
        with pytest.raises(HTTPException) as exc:
            resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert exc.value.status_code == 401

    def test_wrong_algorithm_raises_401(self, rsa_keypair: tuple[bytes, bytes]) -> None:
        """HS256 token against RS256 config must be rejected."""
        _, pub = rsa_keypair
        token = pyjwt.encode(
            {"sub": "acme", "exp": int(time.time()) + 60},
            "shared-secret",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException) as exc:
            resolve_tenant_from_jwt(token, config=_config(pub), default_limits=TenantLimits())
        assert exc.value.status_code == 401
