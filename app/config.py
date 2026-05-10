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
    otel_service_name: str = Field(
        default="llm-gateway", alias="OTEL_SERVICE_NAME"
    )

    service_version: str = Field(default="0.2.0", alias="SERVICE_VERSION")

    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_OTEL_HOST"
    )

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

    tokenizer_encoding_name: str = Field(default="cl100k_base", alias="TOKENIZER_ENCODING_NAME")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Cache
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    """Feature flag. Set False in debugging scenarios to bypass the cache
    without removing infrastructure."""

    cache_ttl_s: int = Field(default=7200, alias="CACHE_TTL_S")
    """Per-entry TTL in seconds. Default 2 hours preserves v0.1.0 behavior."""

    cache_distance_threshold: float = Field(default=0.15, alias="CACHE_DISTANCE_THRESHOLD")
    """Cosine distance threshold for cache hits. Lower = more similar.
    Default 0.15 preserves v0.1.0 threshold."""

    cache_embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="CACHE_EMBEDDER_MODEL",
    )
    """HuggingFace model id for the sentence transformer. Shared between
    the semantic cache and the jailbreak guardrail."""

    cache_embedder_lru_capacity: int = Field(default=512, alias="CACHE_EMBEDDER_LRU_CAPACITY")
    """In-process LRU size for deduplicating encode calls. 512 prompts
    @ ~1.5KB each = under 1MB memory."""

    # PII guardrail
    pii_enabled: bool = Field(default=True, alias="PII_ENABLED")

    pii_policy: str = Field(default="REDACT", alias="PII_POLICY")

    pii_min_score: float = Field(default=0.5, alias="PII_MIN_SCORE")
    """Presidio confidence threshold; matches below this score are ignored."""

    pii_entities: list[str] = Field(
        default=[
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "US_SSN",
            "IP_ADDRESS",
        ],
        alias="PII_ENTITIES",
    )

    # Jailbreak guardrail
    jailbreak_enabled: bool = Field(default=True, alias="JAILBREAK_ENABLED")

    jailbreak_similarity_threshold: float = Field(
        default=0.75, alias="JAILBREAK_SIMILARITY_THRESHOLD"
    )
    """Cosine similarity threshold above which a prompt is classified as
    a jailbreak attempt. Default 0.75 preserves v0.1.0 tuning."""

    jailbreak_phrases: list[str] = Field(
        default=[
            "fail safe mode",
            "act as a developer",
            "system override",
            "you are a linux terminal",
            "ignore all previous instructions",
        ],
        alias="JAILBREAK_PHRASES",
    )


@lru_cache
def get_config() -> Config:
    return Config()


# Backward-Compat
config = get_config()
