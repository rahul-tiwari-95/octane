"""Octane configuration — loaded from .env via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

# Anchor .env to the repo root (one level above this file) so Octane works
# regardless of the user's cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _REPO_ROOT / ".env"


class OctaneSettings(BaseSettings):
    """All Octane configuration. Reads from .env file and environment variables."""

    # --- Bodega Inference Engine (local LLM :44468) ---
    bodega_inference_url: str = Field(
        default="http://localhost:44468",
        description="Bodega Inference Engine base URL",
    )
    bodega_topology: str = Field(
        default="auto",
        description="Model topology: auto|compact|balanced|power",
    )

    # --- Bodega Intelligence — consolidated server (:44469) ---
    bodega_intel_url: str = Field(
        default="http://localhost:44469",
        description="Bodega Intelligence consolidated API base URL",
    )
    bodega_intel_api_key: str = Field(
        default="",
        description="API key for Bodega Intelligence (Bearer token)",
    )

    # --- Legacy per-service URLs (kept for fallback reference) ---
    bodega_search_url: str = Field(default="http://localhost:44469")
    bodega_finance_url: str = Field(default="http://localhost:44469")
    bodega_entertainment_url: str = Field(default="http://localhost:44469")
    bodega_news_url: str = Field(default="http://localhost:44469")

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
    log_level: str = Field(default="WARNING", description="Log level (default WARNING for clean CLI output; use --verbose or LOG_LEVEL=DEBUG for diagnostics)")
    log_format: str = Field(
        default="console",
        description="Log format: 'console' for dev, 'json' for production",
    )

    # --- User ---
    default_user_id: str = Field(default="default", description="Default user ID")

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — import this everywhere
settings = OctaneSettings()
