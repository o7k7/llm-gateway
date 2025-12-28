from pydantic_settings import BaseSettings


class Config(BaseSettings):
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str
    OPENAI_API_KEY: str

    class Config:
        env_file = "../.env"


config = Config()
