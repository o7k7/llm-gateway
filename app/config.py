from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str

    # Legacy LLM provider (LiteLLM fallback)
    LLM_PROVIDER: str
    LLM_MODEL: str
    LLM_API_KEY: str

    vllm_small_url: HttpUrl = Field(
        default=HttpUrl("http://localhost:8001"), alias="VLLM_SMALL_URL"
    )
    vllm_small_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct-AWQ", alias="VLLM_SMALL_MODEL")
    vllm_large_url: HttpUrl = Field(
        default=HttpUrl("http://localhost:8002"), alias="VLLM_LARGE_URL"
    )
    vllm_large_model: str = Field(
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        alias="VLLM_LARGE_MODEL",
    )

    env: Literal["dev", "staging", "prod"] = Field(default="dev", alias="ENV")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    langfuse_pub_key: SecretStr | None = Field(default=None, alias="LANGFUSE_PUB_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")

    small_model_token_threshold: int = 1000
    # Auth (JWT)
    jwt_public_key: SecretStr | None = Field(default=None, alias="JWT_PUBLIC_KEY")
    """PEM-encoded public key. If unset, auth runs in 'dev' mode and accepts a
        raw tenant id via `X-Tenant-Id` header."""
    jwt_algorithm: str = Field(default="RS256", alias="JWT_ALGORITHM")
    jwt_issuer: str | None = Field(default=None, alias="JWT_ISSUER")
    jwt_audience: str | None = Field(default=None, alias="JWT_AUDIENCE")

    # Rate limiting defaults
    default_tenant_rpm: int = Field(default=60, alias="DEFAULT_TENANT_RPM")
    default_tenant_tpm: int = Field(default=100_000, alias="DEFAULT_TENANT_TPM")
    default_tenant_daily_budget_usd: float = Field(
        default=10.0, alias="DEFAULT_TENANT_DAILY_BUDGET_USD"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_config() -> Config:
    return Config()


# Backward-Compat
config = get_config()
