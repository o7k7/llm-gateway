from typing import Literal

from pydantic import Field, HttpUrl
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
    vllm_small_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-AWQ", alias="VLLM_SMALL_MODEL"
    )
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

    small_model_token_threshold: int = 1000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


config = Config()
