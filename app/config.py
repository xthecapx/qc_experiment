from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    token: str
    QBRAID_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()