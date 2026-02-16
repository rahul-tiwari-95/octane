"""Octane configuration — loaded from .env via pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class OctaneSettings(BaseSettings):
    """All Octane configuration. Reads from .env file and environment variables."""

    # --- Bodega Inference Engine (local LLM) ---
    bodega_inference_url: str = Field(
        default="http://localhost:44468",
        description="Bodega Inference Engine base URL",
    )

    # --- Bodega Intelligence (external data APIs) ---
    bodega_search_url: str = Field(
        default="http://localhost:1111",
        description="Beru Search API base URL",
    )
    bodega_finance_url: str = Field(
        default="http://localhost:8030",
        description="Finance API base URL",
    )
    bodega_entertainment_url: str = Field(
        default="http://localhost:8031",
        description="Entertainment API base URL",
    )
    bodega_news_url: str = Field(
        default="http://localhost:8032",
        description="News API base URL",
    )

    # --- PostgreSQL + pgVector ---
    postgres_url: str = Field(
        default="postgresql://localhost:5432/octane",
        description="PostgreSQL connection string",
    )

    # --- Redis ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for Shadows + cache",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="console",
        description="Log format: 'console' for dev, 'json' for production",
    )

    # --- User ---
    default_user_id: str = Field(default="default", description="Default user ID")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — import this everywhere
settings = OctaneSettings()
