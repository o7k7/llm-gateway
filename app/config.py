from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str
    LLM_API_KEY: str
    LLM_MODEL: str = "groq/llama-3.1-8b-instant"

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding = "utf-8",
        extra = "ignore"
    )


config = Config()
