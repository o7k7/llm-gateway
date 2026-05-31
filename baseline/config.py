from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaselineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        alias="BASELINE_MODEL",
    )
    max_new_tokens: int = Field(default=512, alias="BASELINE_MAX_NEW_TOKENS")
    port: int = Field(default=8100, alias="BASELINE_PORT")
    dtype: str = Field(default="float16", alias="BASELINE_DTYPE")


def get_config() -> BaselineConfig:
    return BaselineConfig()
