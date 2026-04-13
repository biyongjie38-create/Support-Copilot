from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Support Copilot"
    environment: str = "development"
    database_url: str = "postgresql://support:support@localhost:5432/support_copilot"
    llm_provider: str = "ollama"
    llm_chat_model: str = "qwen2.5:7b"
    llm_embedding_model: str = "text-embedding-3-small"
    llm_api_key: str | None = None
    llm_base_url: str | None = "http://localhost:11434/v1"
    llm_enable_calls: bool = False
    rag_min_score: float = 0.18
    rag_max_context_chars: int = 5000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
